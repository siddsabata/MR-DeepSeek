from trl.trainer.utils import OnPolicyConfig

from dataclasses import dataclass
import os
from typing import Optional

# Slightly modified version of HuggingFace's PPOConfig
@dataclass
class PPOConfig(OnPolicyConfig):
  exp_name: str = os.path.basename(__file__)[: -len(".py")]
  reward_model_path: str = "EleutherAI/pythia-160m"
  value_model_path: str = "EleutherAI/pythia-160m"
  model_adapter_name: Optional[str] = None
  ref_adapter_name: Optional[str] = None
  num_ppo_epochs: int = 4
  whiten_rewards: bool = False
  kl_coef: float = 0.05
  cliprange: float = 0.2
  vf_coef: float = 0.1
  cliprange_value: float = 0.2
  gamma: float = 1.0
  lam: float = 0.95
