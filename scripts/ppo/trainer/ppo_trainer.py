from accelerate import Accelerator
from accelerate.utils import broadcast, gather_object
import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
from transformers import (
  Trainer,
  DataCollatorWithPadding,
  TrainerControl,
  GenerationConfig,
  is_wandb_available
)
if is_wandb_available():
    import wandb
from transformers.integrations import get_reporting_integration_callbacks
from transformers.trainer import DEFAULT_CALLBACKS, DEFAULT_PROGRESS_CALLBACK
from transformers.trainer_callback import CallbackHandler, ExportableState, PrinterCallback
from transformers.utils import is_peft_available
if is_peft_available():
  from peft import PeftModel, get_peft_model
from trl.core import masked_whiten, masked_mean
from trl.models import create_reference_model
from trl.models.utils import unwrap_model_for_generation
from trl.trainer.utils import (
  OnlineTrainerState,
  peft_module_casting_to_bf16,
  exact_div,
  disable_dropout_in_model,
  prepare_deepspeed,
  batch_generation,
  forward,
  truncate_response,
  first_true_indices,
  get_reward,
  print_rich_table,
  generate_model_card
)

from collections import defaultdict
from contextlib import contextmanager, nullcontext
import gc
import math
import os
import textwrap
import time

from .utils import get_o1_reward, PolicyAndValueWrapper

# Modified version of HuggingFace's PPOTrainer
#   - Uses a o1 reward function
class PPOTrainer(Trainer):
  def __init__(
    self,
    config,
    processing_class,
    reward_processing_class,
    policy,
    ref_policy,
    reward_model,
    train_dataset,
    value_model,
    data_collator,
    eval_dataset,
    optimizers,
    callbacks,
    peft_config
  ):
    if ref_policy is policy:
      raise ValueError(
        "`policy` and `ref_policy` cannot be the same object. If you want `ref_policy` to be the "
        "same as `policy`, you must make a copy of it, or `None` if you use peft."
      )
    
    self.args = config
    args = config
    self.processing_class = processing_class
    self.reward_processing_class = reward_processing_class
    self.policy = policy

    if data_collator is None:
      data_collator = DataCollatorWithPadding(self.processing_class)
    self.policy.generation_config.eos_token_id = None
    self.policy.generation_config.pad_token_id = None

    if not is_peft_available() and peft_config is not None:
      raise ImportError(
        "PEFT is not installed and you passed a `peft_config` in the trainer's kwargs, please install it to use the PEFT models"
      )
    else:
      if isinstance(self.policy, PeftModel):
        self.policy = self.policy.merge_and_unload()

      self.policy = get_peft_model(self.policy, peft_config)
      if args.bf16 and getattr(self.policy, "is_loaded_in_4bit", False):
        peft_module_casting_to_bf16(self.policy)

    self.is_peft_model = is_peft_available() and isinstance(self.policy, PeftModel)
    self.model_adapter_name = args.model_adapter_name
    self.ref_adapter_name = args.ref_adapter_name

    if ref_policy:
      self.ref_policy = ref_policy
    elif self.is_peft_model:
      self.ref_policy = None
    else:
      self.ref_policy = create_reference_model(self.policy)

    self.reward_model = reward_model
    self.train_dataset = train_dataset
    self.train_dataset_len = len(train_dataset)
    self.value_model = value_model
    self.data_collator = data_collator
    self.eval_dataset = eval_dataset
    self.optimizer, self.lr_scheduler = optimizers
    self.optimizer_cls_and_kwargs = None

    if args.total_episodes is None:
      args.total_episodes = int(args.num_train_epochs * self.train_dataset_len)
    accelerator = Accelerator(gradient_accumulation_steps=args.gradient_accumulation_steps)
    self.accelerator = accelerator
    args.world_size = accelerator.num_processes
    args.local_batch_size = (
      args.per_device_train_batch_size * args.gradient_accumulation_steps * args.num_mini_batches
    )
    args.micro_batch_size = int(args.per_device_train_batch_size * args.world_size)
    args.batch_size = int(args.local_batch_size * args.world_size)
    args.mini_batch_size = exact_div(
      args.batch_size, args.num_mini_batches, "`batch_size` must be a multiple of `num_mini_batches`"
    )
    args.local_mini_batch_size = exact_div(
      args.local_batch_size, args.num_mini_batches, "`local_batch_size` must be a multiple of `num_mini_batches`"
    )
    if args.whiten_rewards:
      assert args.local_mini_batch_size >= 8, \
        f"Per-rank minibatch size {args.local_mini_batch_size} is insufficient for whitening"
    args.num_total_batches = math.ceil(args.total_episodes / args.batch_size)
    time_tensor = torch.tensor(int(time.time()), device=accelerator.device)
    time_int = broadcast(time_tensor, 0).item()
    args.run_name = f"{args.exp_name}__{args.seed}__{time_int}"
    self.local_seed = args.seed + accelerator.process_index * 100003
    if args.num_sample_generations > 0:
      self.sample_generations_freq = max(1, args.num_total_batches // args.num_sample_generations)
    self.local_dataloader_batch_size = args.local_batch_size

    for module in [self.policy_model, self.ref_model, self.value_model, self.reward_model]:
      if module is not None:
        disable_dropout_in_model(module)
    self.model = PolicyAndValueWrapper(self.policy_model, self.value_model)
    self.model.config = self.policy_model.config
    self.create_optimizer_and_scheduler(num_training_steps=args.num_total_batches)

    default_callbacks = DEFAULT_CALLBACKS + get_reporting_integration_callbacks(self.args.report_to)
    self.callbacks = default_callbacks if callbacks is None else default_callbacks + callbacks
    self.callback_handler = CallbackHandler(
      self.callbacks, self.model, self.processing_class, self.optimizer, self.lr_scheduler
    )
    self.add_callback(PrinterCallback if self.args.disable_tqdm else DEFAULT_PROGRESS_CALLBACK)
    self.control = TrainerControl()
    self.state = OnlineTrainerState(
      is_local_process_zero=self.is_local_process_zero(),
      is_world_process_zero=self.is_world_process_zero(),
      stateful_callbacks=[
        cb for cb in self.callback_handler.callbacks + [self.control] if isinstance(cb, ExportableState)
      ],
    )
    self.current_flos = 0
    self.hp_search_backend = None
    self.is_deepspeed_enabled = getattr(self.accelerator.state, "deepspeed_plugin", None) is not None
    self.is_fsdp_enabled = getattr(self.accelerator.state, "fsdp_plugin", None) is not None

    self.hub_model_id = None
    if self.args.push_to_hub:
      self.init_hf_repo()
    if self.args.should_save:
      os.makedirs(self.args.output_dir, exist_ok=True)

    if hasattr(self.model, "add_model_tags"):
      self.model.add_model_tags(self._tag_names)

    self.dataloader = DataLoader(
      self.train_dataset,
      batch_size=self.local_dataloader_batch_size,
      shuffle=True,
      collate_fn=self.data_collator,
      drop_last=True
    )

    torch.manual_seed(args.seed)
    self.model, self.optimizer, self.dataloader = accelerator.prepare(self.model, self.optimizer, self.dataloader)
    torch.manual_seed(self.local_seed)

    self.eval_dataloader = DataLoader(
      self.eval_dataset,
      batch_size=args.per_device_eval_batch_size,
      collate_fn=self.data_collator,
      drop_last=True
    )
    self.eval_dataloader = accelerator.prepare(self.eval_dataloader)

    if self.is_deepspeed_enabled:
      self.reward_model = prepare_deepspeed(
        self.reward_model, args.per_device_train_batch_size, args.fp16, args.bf16
      )

      if self.ref_policy is None:
        if not self.is_peft_model:
          raise ValueError("No reference model and model is not a Peft model.")
      else:
        self.ref_policy = prepare_deepspeed(
            self.ref_policy, args.per_device_train_batch_size, args.fp16, args.bf16
        )
    else:
      if self.ref_policy is None:
        if not self.is_peft_model:
          raise ValueError("No reference model and model is not a Peft model.")
      else:
        self.ref_policy = self.ref_policy.to(self.accelerator.device)
      self.reward_model = self.reward_model.to(self.accelerator.device)

  def get_train_dataloder(self):
    return self.dataloader
  
  def get_eval_dataloader(self):
    return self.eval_dataloader
  
  @contextmanager
  def null_ref_context(self):
    """Context manager for handling null reference model (that is, peft adapter manipulation)."""
    with self.accelerator.unwrap_model(
        self.model.policy
    ).disable_adapter() if self.is_peft_model and not self.ref_adapter_name else nullcontext():
      if self.ref_adapter_name:
        self.model.policy.set_adapter(self.ref_adapter_name)
      yield
      if self.ref_adapter_name:
        self.model.policy.set_adapter(self.model_adapter_name or "default")

  def save_model(self, output_dir, _internal_call=False):
    backup_model = self.model
    self.model = self.model.policy
    Trainer.save_model(self, output_dir, _internal_call)
    self.model = backup_model
  
  def _save(self, output_dir=None, state_dict=None):
    if self.is_deepspeed_enabled:
      state_dict = {
        name.removeprefix('policy.'): param for name, param in state_dict.items()
          if name.startswith('policy.')
    }
    super()._save(output_dir, state_dict)

  def train(self):
    args = self.args
    accelerator = self.accelerator
    optimizer = self.optimizer
    model = self.model
    ref_policy = self.ref_policy
    reward_model = self.reward_model
    processing_class = self.processing_class
    dataloader = self.dataloader
    device = accelerator.device

    def repeat_generator():
      while True:
        yield from dataloader

    iter_dataloader = iter(repeat_generator())
    generation_config = GenerationConfig(
      max_new_tokens=args.response_length,
      temperature=(args.temperature + 1e-7),
      top_k=0.0,
      top_p=1.0,
      do_sample=True
    )

    accelerator.print("===training policy===")
    start_time = time.time()
    stats_shape = (args.num_ppo_epochs, args.num_mini_batches, args.gradient_accumulation_steps)
    approxkl_stats = torch.zeros(stats_shape, device=device)
    pg_clipfrac_stats = torch.zeros(stats_shape, device=device)
    pg_loss_stats = torch.zeros(stats_shape, device=device)
    vf_loss_stats = torch.zeros(stats_shape, device=device)
    vf_clipfrac_stats = torch.zeros(stats_shape, device=device)
    entropy_stats = torch.zeros(stats_shape, device=device)
    ratio_stats = torch.zeros(stats_shape, device=device)
    model.train()

    self.state.global_step = 0
    self.state.episode = 0
    self.state.max_steps = args.num_total_batches * args.num_mini_batches
    self.state.num_train_epochs = args.total_episodes / self.train_dataset_len
    if args.logging_steps is not None:
      if args.logging_steps < 1:
        self.state.logging_steps = math.ceil(self.state.max_steps * args.logging_steps)
      else:
        self.state.logging_steps = args.logging_steps
    if args.eval_steps is not None:
      if args.eval_steps < 1:
        self.state.eval_steps = math.ceil(self.state.max_steps * args.eval_steps)
      else:
        self.state.eval_steps = args.eval_steps
    if args.save_steps is not None:
      if args.save_steps < 1:
        self.state.save_steps = math.ceil(self.state.max_steps * args.save_steps)
      else:
        self.state.save_steps = args.save_steps
    self.control = self.callback_handler.on_train_begin(args, self.state, self.control)

    if self.is_deepspeed_enabled:
      self.deepspeed = self.model
      self.model_wrapped = self.model

    for update in range(1, args.num_total_batches + 1):
      self.state.episode += 1 * args.batch_size
      data = next(iter_dataloader)
      with torch.no_grad():
        queries = data["input_ids"].to(device)
        allanswer = data["answer"] 
        context_length = queries.shape[1]
        responses = []
        postprocessed_responses = []
        logprobs = []
        ref_logprobs = []
        scores = []
        sequence_lengths = []
        values = []
        with unwrap_model_for_generation(model, self.accelerator) as unwrapped_model:
          query_responses, logitss = batch_generation(
            unwrapped_model.policy,
            queries,
            args.local_rollout_forward_batch_size,
            processing_class.pad_token_id,
            generation_config
          )

        for i in range(0, queries.shape[0], args.local_rollout_forward_batch_size):
          query = queries[i : i + args.local_rollout_forward_batch_size]
          sub_answer = allanswer[i : i + args.local_rollout_forward_batch_size]
          query_response = query_responses[i : i + args.local_rollout_forward_batch_size]
          response = query_response[:, context_length:]
          logits = logitss[i : i + args.local_rollout_forward_batch_size]
          all_logprob = F.log_softmax(logits, dim=-1)
          logprob = torch.gather(all_logprob, 2, response.unsqueeze(-1)).squeeze(-1)
          del logits, all_logprob
          torch.cuda.empty_cache()

          if ref_policy is None:
            with self.null_ref_context():
              ref_output = forward(model.policy, query_response, processing_class.pad_token_id)
          else:
            ref_output = forward(ref_policy, query_response, processing_class.pad_token_id)
          ref_logits = ref_output.logits[:, context_length - 1 : -1]
          ref_logits /= args.temperature + 1e-7
          ref_all_logprob = F.log_softmax(ref_logits, dim=-1)
          ref_logprob = torch.gather(ref_all_logprob, 2, response.unsqueeze(-1)).squeeze(-1)
          del ref_output, ref_logits, ref_all_logprob
          torch.cuda.empty_cache()

          postprocessed_response = response
          if args.stop_token_id is not None:
            postprocessed_response = truncate_response(
              args.stop_token_id, processing_class.pad_token_id, response
            )

          postprocessed_query_response = torch.cat((query, postprocessed_response), 1)
          sequence_length = first_true_indices(postprocessed_response == processing_class.pad_token_id) - 1
          unwrapped_value_model = accelerator.unwrap_model(model).value_model
          full_value, _, _ = get_reward(
            unwrapped_value_model, query_response, processing_class.pad_token_id, context_length
          )
          value = full_value[:, context_length - 1 : -1].squeeze(-1)
          score = get_o1_reward(
            reward_model, postprocessed_response, processing_class, self.reward_processing_class, processing_class.pad_token_id, sub_answer
          )
          responses.append(response)
          postprocessed_responses.append(postprocessed_response)
          logprobs.append(logprob)
          ref_logprobs.append(ref_logprob)
          sequence_lengths.append(sequence_length)
          scores.append(score)
          values.append(value)
        responses = torch.cat(responses, 0)
        postprocessed_responses = torch.cat(postprocessed_responses, 0)
        logprobs = torch.cat(logprobs, 0)
        ref_logprobs = torch.cat(ref_logprobs, 0)
        sequence_lengths = torch.cat(sequence_lengths, 0)
        scores = torch.cat(scores, 0)
        values = torch.cat(values, 0)
        del logprob, ref_logprob, full_value, value, score, unwrapped_model
        torch.cuda.empty_cache()
        gc.collect()

        contain_eos_token = torch.any(postprocessed_responses == self.processing_class.eos_token_id, dim=-1)
        if self.args.missing_eos_penalty is not None:
          scores[~contain_eos_token] -= self.args.missing_eos_penalty

        response_idxs = torch.arange(responses.shape[1], device=responses.device).repeat(responses.shape[0], 1)
        padding_mask = response_idxs > sequence_lengths.unsqueeze(1)
        logprobs = torch.masked_fill(logprobs, padding_mask, 1.0)
        ref_logprobs = torch.masked_fill(ref_logprobs, padding_mask, 1.0)
        sequence_lengths_p1 = sequence_lengths + 1
        padding_mask_p1 = response_idxs > (sequence_lengths_p1.unsqueeze(1))
        values = torch.masked_fill(values, padding_mask_p1, 0)

        kl = logprobs - ref_logprobs
        non_score_reward = -args.kl_coef * kl
        rewards = non_score_reward.clone()
        actual_start = torch.arange(rewards.size(0), device=rewards.device)
        actual_end = torch.where(sequence_lengths_p1 < rewards.size(1), sequence_lengths_p1, sequence_lengths)
        rewards[[actual_start, actual_end]] += scores

        if args.whiten_rewards:
          rewards = masked_whiten(rewards, mask=~padding_mask_p1, shift_mean=False)
          rewards = torch.masked_fill(rewards, padding_mask_p1, 0)

        lastgaelam = 0
        advantages_reversed = []
        gen_length = responses.shape[1]
        for t in reversed(range(gen_length)):
          nextvalues = values[:, t + 1] if t < gen_length - 1 else 0.0
          delta = rewards[:, t] + args.gamma * nextvalues - values[:, t]
          lastgaelam = delta + args.gamma * args.lam * lastgaelam
          advantages_reversed.append(lastgaelam)
        advantages = torch.stack(advantages_reversed[::-1], axis=1)
        returns = advantages + values
        advantages = masked_whiten(advantages, ~padding_mask)
        advantages = torch.masked_fill(advantages, padding_mask, 0)
        torch.cuda.empty_cache()
    
    for ppo_epoch_idx in range(args.num_ppo_epochs):
      b_inds = np.random.permutation(args.local_batch_size)
      minibatch_idx = 0
      for mini_batch_start in range(0, args.local_batch_size, args.local_mini_batch_size):
        mini_batch_end = mini_batch_start + args.local_mini_batch_size
        mini_batch_inds = b_inds[mini_batch_start:mini_batch_end]
        gradient_accumulation_idx = 0
        for micro_batch_start in range(0, args.local_mini_batch_size, args.per_device_train_batch_size):
          with accelerator.accumulate(model):
            micro_batch_end = micro_batch_start + args.per_device_train_batch_size
            micro_batch_inds = mini_batch_inds[micro_batch_start:micro_batch_end]
            mb_advantage = advantages[micro_batch_inds]
            mb_responses = responses[micro_batch_inds]
            mb_query_responses = query_responses[micro_batch_inds]
            mb_logprobs = logprobs[micro_batch_inds]
            mb_return = returns[micro_batch_inds]
            mb_values = values[micro_batch_inds]

            output, vpred_temp = forward(model, mb_query_responses, processing_class.pad_token_id)
            logits = output.logits[:, context_length - 1 : -1]
            logits /= args.temperature + 1e-7
            new_all_logprobs = F.log_softmax(logits, dim=-1)
            new_logprobs = torch.gather(new_all_logprobs, 2, mb_responses.unsqueeze(-1)).squeeze(-1)
            new_logprobs = torch.masked_fill(
              new_logprobs, padding_mask[micro_batch_inds], 1.0
            )
            vpred = vpred_temp[:, context_length - 1 : -1].squeeze(-1)
            vpred = torch.masked_fill(vpred, padding_mask_p1[micro_batch_inds], 0)
            vpredclipped = torch.clamp(
              vpred,
              mb_values - args.cliprange_value,
              mb_values + args.cliprange_value,
            )
            vf_losses1 = torch.square(vpred - mb_return)
            vf_losses2 = torch.square(vpredclipped - mb_return)
            vf_loss_max = torch.max(vf_losses1, vf_losses2)
            vf_loss = 0.5 * masked_mean(vf_loss_max, ~padding_mask_p1[micro_batch_inds])
            vf_clipfrac = masked_mean(
              (vf_losses2 > vf_losses1).float(), ~padding_mask_p1[micro_batch_inds]
            )
            logprobs_diff = new_logprobs - mb_logprobs
            ratio = torch.exp(logprobs_diff)
            pg_losses = -mb_advantage * ratio
            pg_losses2 = -mb_advantage * torch.clamp(ratio, 1.0 - args.cliprange, 1.0 + args.cliprange)
            pg_loss_max = torch.max(pg_losses, pg_losses2)
            pg_loss = masked_mean(pg_loss_max, ~padding_mask[micro_batch_inds])
            loss = pg_loss + args.vf_coef * vf_loss
            accelerator.backward(loss)
            optimizer.step()
            optimizer.zero_grad()
            with torch.no_grad():
              pg_clipfrac = masked_mean(
                (pg_losses2 > pg_losses).float(), ~padding_mask[micro_batch_inds]
              )
              prob_dist = torch.nn.functional.softmax(logits, dim=-1)
              entropy = torch.logsumexp(logits, dim=-1) - torch.sum(prob_dist * logits, dim=-1)
              approxkl = 0.5 * (logprobs_diff**2).mean()
              approxkl_stats[ppo_epoch_idx, minibatch_idx, gradient_accumulation_idx] = approxkl
              pg_clipfrac_stats[ppo_epoch_idx, minibatch_idx, gradient_accumulation_idx] = pg_clipfrac
              pg_loss_stats[ppo_epoch_idx, minibatch_idx, gradient_accumulation_idx] = pg_loss
              vf_loss_stats[ppo_epoch_idx, minibatch_idx, gradient_accumulation_idx] = vf_loss
              vf_clipfrac_stats[ppo_epoch_idx, minibatch_idx, gradient_accumulation_idx] = vf_clipfrac
              entropy_stats[ppo_epoch_idx, minibatch_idx, gradient_accumulation_idx] = entropy.mean()
              ratio_stats[ppo_epoch_idx, minibatch_idx, gradient_accumulation_idx] = ratio.mean()
          gradient_accumulation_idx += 1
        minibatch_idx += 1
        del (
          output, vpred_temp, logits, new_all_logprobs, new_logprobs, vpred, vpredclipped,
          vf_losses1, vf_losses2, vf_loss, vf_clipfrac, logprobs_diff, ratio, pg_losses, pg_losses2, pg_loss_max,
          pg_loss, loss, pg_clipfrac, prob_dist, entropy, approxkl, mb_return,
          mb_advantage, mb_values, mb_responses, mb_query_responses, mb_logprobs
        )
        torch.cuda.empty_cache()

    with torch.no_grad():
      mean_kl = kl.sum(1).mean()
      mean_entropy = (-logprobs).sum(1).mean()
      mean_non_score_reward = non_score_reward.sum(1).mean()
      rlhf_reward = mean_non_score_reward + scores.mean()
      eps = int(self.state.episode / (time.time() - start_time))
      metrics = {}
      metrics["eps"] = eps
      metrics["objective/kl"] = self.accelerator.gather(mean_kl).mean().item()
      metrics["objective/entropy"] = self.accelerator.gather(mean_entropy).mean().item()
      metrics["objective/non_score_reward"] = self.accelerator.gather(mean_non_score_reward).mean().item()
      metrics["objective/rlhf_reward"] = self.accelerator.gather(rlhf_reward).mean().item()
      metrics["objective/scores"] = self.accelerator.gather(scores.mean()).mean().item()
      metrics["policy/approxkl_avg"] = self.accelerator.gather(approxkl_stats).mean().item()
      metrics["policy/clipfrac_avg"] = self.accelerator.gather(pg_clipfrac_stats).mean().item()
      metrics["loss/policy_avg"] = self.accelerator.gather(pg_loss_stats).mean().item()
      metrics["loss/value_avg"] = self.accelerator.gather(vf_loss_stats).mean().item()
      metrics["val/clipfrac_avg"] = self.accelerator.gather(vf_clipfrac_stats).mean().item()
      metrics["policy/entropy_avg"] = self.accelerator.gather(entropy_stats).mean().item()
      metrics["val/ratio"] = self.accelerator.gather(ratio_stats).mean().item()
      metrics["val/ratio_var"] = self.accelerator.gather(ratio_stats).var().item()
      metrics["val/num_eos_tokens"] = (responses == processing_class.eos_token_id).sum().item()
      metrics["lr"] = self.lr_scheduler.get_last_lr()[0]
      metrics["episode"] = self.state.episode
      self.state.epoch = self.state.episode / self.train_dataset_len
      self.state.global_step += 1
      self.log(metrics)

    self.lr_scheduler.step()
    self.control = self.callback_handler.on_step_end(args, self.state, self.control)
    if self.control.should_save:
      self._save_checkpoint(model, trial=None)
      self.control = self.callback_handler.on_save(self.args, self.state, self.control)
    del kl, mean_kl, mean_entropy, mean_non_score_reward, scores, metrics, non_score_reward
    torch.cuda.empty_cache()
    gc.collect()

    if args.num_sample_generations > 0 and (update - 1) % self.sample_generations_freq == 0:
      self.generate_completions(sampling=True)
      torch.cuda.empty_cache()
    del (
      query_responses, responses, postprocessed_responses, logprobs, ref_logprobs,
      values, sequence_lengths, contain_eos_token, sequence_lengths_p1, response_idxs,
      padding_mask, padding_mask_p1, rewards, actual_start, actual_end, advantages,
      returns
    )
    torch.cuda.empty_cache()

    self.control = self.callback_handler.on_train_end(args, self.state, self.control)
    if self.control.should_save:
      self._save_checkpoint(model, trial=None, metrics=None)
      self.control = self.callback_handler.on_save(self.args, self.state, self.control)
  
  def generate_completions(self, sampling=False):
    args = self.args
    processing_class = self.processing_class
    generation_config = GenerationConfig(
      max_new_tokens=self.args.response_length,
      temperature=(0.01 + 1e-7),
      top_k=0.0,
      top_p=1.0,
      do_sample=True
    )

    table = defaultdict(list)
    with unwrap_model_for_generation(self.model, self.accelerator) as unwrapped_model:
      for batch in self.eval_dataloader:
          query = batch["input_ids"]
          with torch.no_grad():
            context_length = query.shape[1]
            query_response, _ = batch_generation(
              unwrapped_model.policy,
              query,
              query.shape[0],
              processing_class.pad_token_id,
              generation_config
            )
            response = query_response[:, context_length:]
            postprocessed_response = response
            if args.stop_token_id is not None:
              postprocessed_response = truncate_response(
                args.stop_token_id, processing_class.pad_token_id, response
              )
            table["query"].extend(
              gather_object(processing_class.batch_decode(query, skip_special_tokens=True))
            )
            table["model response"].extend(
              gather_object(processing_class.batch_decode(postprocessed_response))
            )

            postprocessed_query_response = torch.cat((query, postprocessed_response), 1)
            _, score, _ = get_reward(
              self.reward_model, postprocessed_query_response, processing_class.pad_token_id, context_length
            )
            table["score"].extend(self.accelerator.gather(score).float().cpu().numpy())

          if sampling:
              break
      df = pd.DataFrame(table)
      if self.accelerator.is_main_process:
        print_rich_table(df.iloc[0 : 0 + 5])
        if "wandb" in args.report_to:
          import wandb

          if wandb.run is not None:
            wandb.log({"completions": wandb.Table(dataframe=df)})

  def create_model_card(self, model_name=None, dataset_name=None, tags=None):
    if not self.is_world_process_zero():
      return

    if hasattr(self.model.config, "_name_or_path") and not os.path.isdir(self.model.config._name_or_path):
      base_model = self.model.config._name_or_path
    else:
      base_model = None

    tags = tags or []
    if isinstance(tags, str):
      tags = [tags]

    if hasattr(self.model.config, "unsloth_version"):
      tags.append("unsloth")

    citation = textwrap.dedent("""\
    @article{mziegler2019fine-tuning,
        title        = {{Fine-Tuning Language Models from Human Preferences}},
        author       = {Daniel M. Ziegler and Nisan Stiennon and Jeffrey Wu and Tom B. Brown and Alec Radford and Dario Amodei and Paul F. Christiano and Geoffrey Irving},
        year         = 2019,
        eprint       = {arXiv:1909.08593}
    }""")

    model_card = generate_model_card(
      base_model=base_model,
      model_name=model_name,
      hub_model_id=self.hub_model_id,
      dataset_name=dataset_name,
      tags=tags,
      wandb_url=wandb.run.get_url() if is_wandb_available() and wandb.run is not None else None,
      trainer_name="PPO",
      trainer_citation=citation,
      paper_title="Fine-Tuning Language Models from Human Preferences",
      paper_id="1909.08593",
    )

    model_card.save(os.path.join(self.args.output_dir, "README.md"))
