import torch
from torch import nn
import numpy as np

from infrastructure import pytorch_utils as ptu

class DQNCritic(nn.Module):
    """Critic network for DQN. Maps observations to Q-values for each action."""
    
    def __init__(
        self,
        observation_shape,
        num_actions,
        n_layers,
        size
    ):
        super().__init__()
        self.net = ptu.build_mlp(
            input_size=int(np.prod(observation_shape)),
            output_size=num_actions,
            n_layers=n_layers,
            size=size
        )
    
    def forward(self, obs):
        """
        Return Q-values for all actions.
        
        arguments:
            obs: (batch_size, *observation_shape) observations
        return:
            qa_values: (batch_size, num_actions) Q-values for each action

        """
        # Flatten observations if needed
        if obs.ndim > 2:
            obs = obs.reshape(obs.shape[0], -1)
        return self.net(obs)