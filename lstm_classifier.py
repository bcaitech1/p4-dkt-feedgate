import os
import flask_trainer as trainer
import torch
import pandas as pd
import numpy as np
import sys
import datetime as pydatetime
from train import YamlConfigManager

from bentoml import env, artifacts, api, BentoService
from bentoml.adapters import DataframeInput
from bentoml.adapters import JsonInput, JsonOutput
from bentoml.frameworks.pytorch import PytorchModelArtifact
from bentoml.service.artifacts.pickle import PickleArtifact

sys.path.append(os.path.dirname(os.path.abspath(os.path.dirname(__file__))))
from args import parse_args
from config_dir import config
from dkt.dataloader import Preprocess


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


@env(infer_pip_packages=True) # infer error 
@artifacts([PytorchModelArtifact('model')])
class LstmClassifier(BentoService):
    def __init__(self):
        super().__init__()
        self.args=None # (Warning) after generating object, args might be set in hand.
    
    # @api(
    #     input=DataframeInput(
    #         orient="records",
    #         columns=["userID","assessmentItemID","testId","answerCode","Timestamp","KnowledgeTag"],
    #         dtype={"userID":"int","assessmentItemID":"str","testId":"str","answerCode":"int","Timestamp":"str","KnowledgeTag":"int"},
    #     )
    # ,batch=True)
    @api(input=DataframeInput(), batch=True)
    def inference(self,data:pd.DataFrame):
        """
            (Input)
                - data : `user data` (From Chrome Browser, Client choices were loaded.)
            (Output)
                - result : after inference, it might be the calculated score or probability.
        """
        # [To-Do] Generate data, firstly formatted by csv columns
        data = gen_data(data) # new_df
        
        # Pre-processing user data
        preprocess = Preprocess(self.args)
        preprocess.load_test_data(data)
        
        test_data = preprocess.get_test_data()

        result = trainer.inference(self.args, test_data)
        print('Result : ',result)

        return result
        # return self.artifacts.model.inference(data)

# 민원 청구 : 
# 민원 창구 여자 이름 : 
# 자진신고 : 최대 감면 - 