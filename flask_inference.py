import os
from dkt.dataloader import Preprocess
import flask_trainer as trainer
import torch
import pandas as pd
import numpy as np
import sys
import datetime as pydatetime
from train import YamlConfigManager

sys.path.append(os.path.dirname(os.path.abspath(os.path.dirname(__file__))))
from args import parse_args
from config import config

# Bring config manager
args = parse_args(mode='train')
cfg = YamlConfigManager(args.config_path, args.config)
config.set_args(args,cfg)

# Set device
device = "cuda" if torch.cuda.is_available() else "cpu"
args.device = device

# Set config
args.n_questions = len(np.load(os.path.join(args.asset_dir,'assessmentItemID_classes.npy')))
args.n_test = len(np.load(os.path.join(args.asset_dir,'testId_classes.npy')))
args.n_tag = len(np.load(os.path.join(args.asset_dir,'KnowledgeTag_classes.npy')))

# (Caution) Pre-load model in memory
model = trainer.load_model(args)
model.to(device)
args.model = model

def gen_data(data):
    """
        (Function)
            1. Read train/test/dummy .csv file for make frame columns 
            2. Make Dataframe 
    """
    # [To-Do] Refactor later for skip reading csv
    df = pd.read_csv("questions.csv")
    new_columns = df.columns.tolist()

    user_actions = pd.DataFrame(data, columns=new_columns)   

    user_actions['answerCode'].fillna(-1, inplace=True)
    user_actions['answerCode']=user_actions['answerCode'].astype(int)
    user_actions['KnowledgeTag']=user_actions['KnowledgeTag'].astype(str)
    
    # Save for next train 
    save_df = df.append(user_actions,ignore_index=True)
    save_df.to_csv('/opt/ml/code/questions.csv', index=False)
    return user_actions

    
def inference(data):
    """
        (Input)
            - data : `user data` (From Chrome Browser, Client choices were loaded.)
        (Output)
            - result : after inference, it might be the calculated score or probability.
    """
    # [To-Do] Generate data, firstly formatted by csv columns
    data = gen_data(data) # new_df
    
    # Pre-processing user data
    preprocess = Preprocess(args)
    preprocess.load_test_data(data)
    
    test_data = preprocess.get_test_data()

    result = trainer.inference(args, test_data)
    print('result : ',result)

    return result    