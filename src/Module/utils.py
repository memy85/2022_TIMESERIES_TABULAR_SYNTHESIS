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

def load_file(path, file_name, file_type):
    '''
    path : path to the file
    file_name : full name of the file
    file_type : csv, pickle, feather
    '''
    if file_type == 'csv':
        return pd.read_csv(Path(path).joinpath(file_name))
    
    elif file_type =='pickle':
        return pd.read_pickle(Path(path).joinpath(file_name))
    
    else : 
        return pd.read_feather(Path(path).joinpath(file_name))
    

    
    
