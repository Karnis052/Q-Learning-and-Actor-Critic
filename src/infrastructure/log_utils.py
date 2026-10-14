import copy
import json
import os
import pickle
import tempfile

import numpy as np
import torch
import wandb
from torch import nn

class Logger:
    """LOogger for logging metrics"""
    
    def __init__(self, path):
        self.path = path
        self.header = None
        self.file = None
        self.disallowed_types = (wandb.Image, wandb.Video, wandb.Histogram)
        self.rows = []
        
    def log(self, row, step):
        pass
    
def setup_wandb(
    entity=None,
    project='project',
    group=None,
    name=None,
    mode='online',
    config=None
):
    wandb_output_dir = tempfile.mkdtemp()
    wandb_tag_max_characters = 64
    if group is not None and len(group)>wandb_tag_max_characters:
        tags = [group[:64]]
    else:
        tags = [group] if group is not None else None
        
    init_kwargs = dict(
        config=config,
        project=project,
        entity=entity,
        tags=tags,
        group=group,
        dir=wandb_output_dir,
        name=name,
        settings=wandb.Settings(
            start_method='thread',
            _disable_starts=False
        )
        mode=mode,
        save_code=True
    )
    
    run = wandb.init(**init_kwargs)
    return run

        
    