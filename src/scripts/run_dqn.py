import time
import argparse
import yaml
import os

import gym
import numpy as np
import torch
import tqdm
from configs import dqn_config
from infrastructure.log_utils import Logger, setup_wandb
from infrastructure import pytorch_utils as ptu
from agents.dqn_agent import DQNAgent

MAX_VIDEO = 2

def run_training_loop(config: dict, logger: Logger, args: argparse.Namespace):
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    ptu.init_gpu(use_gpu=not args.no_gpu, gpu_id=args.which_gpu)
    
    env = config["make_env"](eval=False)
    eval_env = config["make_env"](eval=True)
    render_env = config["make_env"](eval=True, render=True)
    exploration_schedule = config["exploration_schedule"]
    discrete = isinstance(env.action_space, gym.space.Discrete)
    
    assert discrete, "DQN only supports discrete action space"
    
    agent = DQNAgent(
        env.observation_space.shape,
        env.action_space.n,
        **config["agent_kwargs"],
    )
    
    if "model" in dir(env):
        fps = 1 / env.model.opt.timestep
    elif "render_fps" in env.env.metadata:
        fps = env.env.metadata["render_fps"]
    else:
        fps = 4

    ep_len = env.spec.max_episode_steps
    
    observation = None
    
    #Replay buffer
    if len(env.observation_space.shape) == 3:
        stacked_frames = True
        frame_history_len = env.observation_space.shape[0]
        assert frame_history_len == 4, "Only support 4 stacked frames"
        replay_buffer = Me
        
    
    
    

def make_config(config_file: str) ->dict:
    with open(config_file, "r") as f:
        config_kwargs = yaml.safe_load(f)
    
    base_config_name = config_kwargs.pop("base_config")
    return dqn_config.configs[base_config_name](**config_kwargs)
    
    
def make_logger(config: dict, args: argparse.Namespace) -> Logger:
    logdir = "{}_sd{}_{}".format(
        config["log_name"], args.seed, time.strftime("%Y%m%d_%H%M%S")
    )
    logdir = os.path.join("exp", logdir)
    os.makedirs(logdir, exist_ok=True)
    
    wandb_config = {**config, **vars(args)}
    setup_wandb(
        entity=args.wandb_entity,
        project=args.wandb_project,
        group=config["log_name"],
        name=logdir.split("/")[-1],
        mode='online',
        config=wandb_config
    )
    return Logger(os.path.join(logdir, "log.csv"))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config_file", "-cfg", type=str, required=True)
    
    parser.add_argument("--eval_interval", "-ei", type=int, default=10000)
    parser.add_argument("--num_eval_trajectories", "-neval", type=int, default=10)
    parser.add_argument("--num_render_trajectories", "-nvid", type=int, default=0)
    
    parser.add_argument("--seed", type=int, default=1)
    parser.add_argument("--no_gpu", "-ngpu", action="store_true")
    parser.add_argument("--which_gpu", "-gpu_id", default=0)
    parser.add_argument("--log_interval", type=int, default=1000)
    
    parser.add_argument("--wandb_entity", type=str, default=None)
    parser.add_argument("--wandb_project", type=str, default="actor_critic")
    
    args = parser.parse_args()
    
    config = make_config(args.config_file)
    logger = make_logger(config, args)
    
    run_training_loop(config, logger, args)
    


if __name__ == "__main__":
    main()