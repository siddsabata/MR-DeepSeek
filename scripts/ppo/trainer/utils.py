import torch
import torch.nn as nn
import torch.nn.functional as F

import re

def get_o1_reward(model, input_ids, tokenizer, reward_tokenizer, answer, max_len=4000):
  response_template = '''<Model Response>
{}
</Model Response>

<Reference Answer>
{}
</Reference Answer>

Your task is to evaluate the model response by comparing it to the reference answer. If the model response is correct and aligns with the reference answer, output "True" . If it is incorrect or fails to select the correct option (if options are provided), output "False" . {}'''
  output_pattern = r'## Final Response\n\n(.*)'

  batch = []
  model_outputs = []
  for idx in range(len(answer)):
    response = tokenizer.decode(input_ids[idx], skip_special_tokens=True)
    final_response_count = response.count('## Final Response\n\n')
    thinking_count = response.count('## Thinking')

    if '## Final Response\n\n' in response and final_response_count == 1 and thinking_count == 1:
      model_output = re.search(output_pattern, response, re.S)
    else:
      model_output = None
    model_outputs.append(model_output)
    if model_output is not None:
      model_output = model_output.group(1).strip()
    else:
      model_output = 'I do not know the answer.'
    response = response_template.format(model_output, answer[idx], reward_tokenizer.eos_token)
    batch.append(response)

  input_batch = reward_tokenizer(batch, return_tensors="pt", add_special_tokens=False, max_length=max_len, padding=True, truncation=True).to(model.device)
  with torch.no_grad():
    logits = model(**input_batch, return_dict=True).logits
    probs = F.softmax(logits, dim=-1)

    rewards_list = []
    for idx in range(len(answer)):
      if model_outputs[idx] is None:
        rewards_list.append(0.0)
      else:
        p = probs[idx, 1].item()
        if p > 0.4:
          rewards_list.append(1.0)
        else:
          rewards_list.append(0.1)
    rewards = torch.tensor(rewards_list, device=model.device, dtype=probs.dtype)
    accumulated_reward = rewards.sum().item() / len(batch)

    return rewards, accumulated_reward

class PolicyAndValueWrapper(nn.Module):
  def __init__(self, policy, value_model):
    super().__init__()
    self.policy = policy
    self.value_model = value_model
    self.critic = getattr(value_model, value_model.base_model_prefix)

  def forward(self, **kwargs):
    out = self.critic(**kwargs)
    logits = self.value_model.score(out.hidden_states[-1])
    return self.policy(**kwargs), logits
