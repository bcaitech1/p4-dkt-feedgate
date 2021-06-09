import os
import os.path as p

import torch
import torch.nn as nn
import pandas as pd
import numpy as np

from bentoml import env, artifacts, api, BentoService
from bentoml.adapters import DataframeInput
from bentoml.adapters import JsonInput, JsonOutput
from bentoml.frameworks.pytorch import PytorchModelArtifact
from bentoml.service.artifacts.pickle import PickleArtifact

import sys
# sys.path.append('/opt/ml/code')
print(os.getcwd())
from args import parse_args
from train import YamlConfigManager
from inference import main
from config_dir import config
sys.path.append('/opt/ml/code/dkt/')
print(os.getcwd())

from dataloader import Preprocess
import trainer


def gen_data(data):
    df = pd.read_csv("questions.csv")
    
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

@env(infer_pip_packages=True) # infer error 
@artifacts([PytorchModelArtifact('model')])
class LstmClassifier(BentoService):
    def __init__(self,args):
        super().__init__()
        self.args=args

    @api(input=DataframeInput(), batch=True)
    def inference(self, df: pd.DataFrame):
        """
        (Input)
            args
        (Output)
            after preprocess, return test data
        """

        # Pre-process first
        data = gen_data(df)
        
        preprocess = Preprocess(args=self.args)
        preprocess.load_test_data(data)
        
        test_data = preprocess.get_test_data()
        # result = trainer.inference(args, test_data)
        # return result
        return self.artifacts.model.inference(df)


# if __name__=="__main__":
    # load model - 이걸 저장된 애로 가져와야 함 (아직 해결 못함)
    # model = nn.LSTM(64,64,2,batch_first=True)
    # model_path = p.join('/opt/ml/code/models', 'lstm.pt')
    # model.load_state_dict(torch.load(model_path, map_location=torch.device('cpu')),strict=False)
    # model.eval()

    # packaging and save service 
    # lstm_classifier_service = LstmClassifier(args=args)
    # lstm_classifier_service.pack('model', model) # packaging with trained model
    # saved_path = lstm_classifier_service.save() # save service (check (ex. $bentoml serve LstmClassifier:latest))