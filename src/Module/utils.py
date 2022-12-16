#%%
import pandas as pd
from pathlib import Path
import pickle
import yaml

def load_config():
    '''
    loads configuration
    '''
    project_path = Path(__file__).parents[2]
    config_path = project_path.joinpath('config/config.yaml')
    
    with open(config_path, 'rb') as f:
        config = yaml.load(f, yaml.SafeLoader)
        
    return config


# %%
