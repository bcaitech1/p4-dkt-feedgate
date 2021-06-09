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
print('=======')

from args import parse_args
from train import YamlConfigManager
from inference import main
from config_dir import config
sys.path.append('/opt/ml/code/dkt/')
print(os.getcwd())

from dataloader import Preprocess
import trainer

args = parse_args(mode='train')
cfg = YamlConfigManager(args.config_path, args.config)
config.set_args(args,cfg)
print(args)
device = "cuda" if torch.cuda.is_available() else "cpu"
args.device = device
args.n_questions = len(np.load(os.path.join(args.asset_dir,'assessmentItemID_classes.npy')))
args.n_test = len(np.load(os.path.join(args.asset_dir,'testId_classes.npy')))
args.n_tag = len(np.load(os.path.join(args.asset_dir,'KnowledgeTag_classes.npy')))

model = trainer.load_model(args)
model.to(device)
args.model = model


lstm_classifier_service = LstmClassifier(args=args)
lstm_classifier_service.pack('model', model) # packaging with trained model
saved_path = lstm_classifier_service.save() # save service (check (ex. $bentoml serve LstmClassifier:latest))