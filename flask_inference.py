import os
from flask_dataloader import Preprocess
import flask_trainer as trainer
import torch
import pandas as pd
import numpy as np
import sys
import datetime as pydatetime
from flask_train import YamlConfigManager
sys.path.append(os.path.dirname(os.path.abspath(os.path.dirname(__file__))))
from args import parse_args
from config_dir import config
# (추가된 부분) 모델 미리 로딩

args = parse_args(mode='train')
cfg = YamlConfigManager(args.config_path, args.config)
config.set_args(args,cfg)

device = "cuda" if torch.cuda.is_available() else "cpu"
args.device = device
args.n_questions = len(np.load(os.path.join(args.asset_dir,'assessmentItemID_classes.npy')))
args.n_test = len(np.load(os.path.join(args.asset_dir,'testId_classes.npy')))
args.n_tag = len(np.load(os.path.join(args.asset_dir,'KnowledgeTag_classes.npy')))

model = trainer.load_model(args)
model.to(device)
args.model = model


def gen_data(data):
    df = pd.read_csv("questions_1.csv")
    new_columns = df.columns.tolist()+['answerCode'] 
    new_df = pd.DataFrame([],columns=new_columns+['userID'])
    
    for index, row in df.iterrows():
        user_actions = pd.DataFrame(data, columns=new_columns)    
        user_actions['userID'] = index
        new_df=new_df.append(user_actions)
        row['userID'] = index
        
        new_df=new_df.append(row)
    
    new_df['answerCode'].fillna(-1, inplace=True)
    new_df['answerCode']=new_df['answerCode'].astype(int)
    new_df['KnowledgeTag']=new_df['KnowledgeTag'].astype(str)
    return new_df

    
def inference(data):
    
    data = gen_data(data)
    
    
    # (바뀐 부분)
    preprocess = Preprocess(args)
    preprocess.load_test_data(data)
    
    test_data = preprocess.get_test_data()

    result = trainer.inference(args, test_data)
    print('result : ',result)
    return result    

