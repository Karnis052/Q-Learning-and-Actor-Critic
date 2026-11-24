from collections import OrderedDict
import numpy as np
import copy
import gymnasium as gym
import cv2
from infrastructure import pytorch_utils as ptu
from typing import Dict, Tuple, List

def sample_trajectory(
    env: gym.Env,
    policy, 
    max_length: int,
    render: bool = False
) -> Dict[str, np.ndarray]:
    """Sample a rollout in the environment from a policy."""
    ob, _ = env.reset()
    obs, acs, rewards, next_obs, terminals, image_obs = [], [], [], [], [], []
    steps = 0
    
    while True:
        # render an image
        if render:
            if hasattr(env, "sim"):
                img = env.sim.render(camera_name="track", height=500, width=500)[::-1]
            else:
                img = env.render()
                
            if isinstance(img, list):
                img = img[0]
                
            image_obs.append(
                cv2.resize(img, dsize=(250, 250), interpolation=cv2.INTER_CUBIC)
            )
        
        # Get action from policy
        ac = policy.get_action(ob)
        
        # Take action and get reward and next observation
        next_ob, rew, terminated, truncated, info = env.step(ac)
        
        #Rollout can end due to done, or due to max_length
        steps += 1
        rollout_done = terminated or truncated or steps>max_length
        
        # record result of taking that actiion 
        obs.append(ob)
        acs.append(ac)
        rewards.append(rew)
        next_obs.append(next_ob)
        terminals.append(rollout_done)
        
        ob = next_ob
        
        # end the rollout if the rollout ended
        if rollout_done:
            break
    episode_statistics = {"l": steps, "r": np.sum(rewards)}
    if "episode" in info:
        episode_statistics.update(info["episode"])
    
    return {
        "observation": np.array(obs, dtype=np.float32),
        "image_obs": np.array(image_obs, dtype=np.uint8),
        "reward": np.array(rewards, dtype=np.float32),
        "action": np.array(acs, dtype=np.float32),
        "next_observation": np.array(next_obs, dtype=np.float32),
        "terminal": np.array(terminals, dtype=np.float32),
        "episode_statistics": episode_statistics
    }
    

def sample_trajectories(
    env: gym.Env, 
    policy,
    min_timesteps_per_batch: int,
    max_length: int, 
    render: bool = False
) -> Tuple[List[Dict[str, np.ndarray]], int]:
    """Collect rollouts using policy until we have collected min_timesteps_per_batch steps."""
    timesteps_this_batch = 0
    trajs = []
    while timesteps_this_batch < min_timesteps_per_batch:
        # collect rollout
        traj = sample_trajectory(env, policy, max_length, render)
        trajs.append(traj)
        
        # Count steps
        timesteps_this_batch += get_traj_length(traj)
    return trajs, timesteps_this_batch
        

def sample_n_trajectories(
    env: gym.Env, policy, ntraj: int, max_length: int, render: bool = False
):
    """Collect ntraj rollouts."""
    trajs = []
    for _ in range(ntraj):
        # collect rollout
        traj = sample_trajectory(env, policy, max_length, render)
        trajs.append(traj)
    return trajs


def get_traj_length(traj):
    return len(traj["reward"])