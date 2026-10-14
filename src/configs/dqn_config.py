from typing import Optional, Tuple

import gymnasium as gym
from gym.wrappers.record_episode_statistics import RecordEpisodeStatistics
from gym.wrappers.frame_stack import FrameStack 


import numpy as np
import torch 
import torch.nn as nn


def basic_dqn_config(
    env_name: str,
    exp_name: Optional[str] = None,
    hidden_size: int = 64,
    num_layers: int = 2,
    learning_rate: int = 1e-3,
    total_steps: int = 1000000,
    discount: float = 0.99,
    target_update_period: int = 1000,
    clip_grad_norm: Optional[float] = None,
    use_double_q: bool = False,
    learning_starts: int  = 20000,
    batch_size: int = 128,
    **kwargs
):
    def make_config():
        pass


configs = {
    "dqn_basic": basic_dqn_config,
    "dqn_atari": atari_dqn_config,
}