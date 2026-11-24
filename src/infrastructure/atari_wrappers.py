import numpy as np
import gymnasium as gym
from gymnasium.wrappers import RecordEpisodeStatistics
from gymnasium.wrappers import FrameStackObservation
from gymnasium.wrappers import AtariPreprocessing

class FireResetEnv(gym.Wrapper):
    def __init__(self, env):
        """Take action on reset for environments that are fixed until firing."""
        gym.Wrapper.__init__(self, env)
        assert env.unwrapped.get_action_meanings()[1] == "FIRE"
        assert len(env.unwrapped.get_action_meanings()) >= 3
    
    def reset(self, **kwargs):
        self.env.reset(**kwargs)
        obs, _, terminated, truncated, info = self.env.step(1)
        if terminated or truncated:
            self.env.reset(**kwargs)
        obs, _, terminated, truncated, info = self.env.step(2)
        if terminated or truncated:
            self.env.reset(**kwargs)
        return obs
    
    def step(self, ac):
        return self.env.step(ac)
    

class ClipRewardEnv(gym.RewardWrapper):
    def __init__(self, env):
        gym.RewardWrapper.__init__(self, env)
        
    def reward(self, reward):
        """Bin reward to {+1, 0, -1} by its sign."""
        return np.sign(reward)

def wrap_deepmind(env: gym.Env):
    """ Configure environment for DeepMind-style Atari.
    """
    env = RecordEpisodeStatistics(env)
    
    # Standard Atari preprocessing
    env = AtariPreprocessing(
        env,
        frame_skip=4,
        screen_size=84,
        noop_max=30,
        grayscale_obs=True,
        terminal_on_life_loss=False,
    )
    
    env = FrameStackObservation(env, stack_size=4)
    return env