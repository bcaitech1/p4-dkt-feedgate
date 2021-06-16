import os
import os.path as p
import torch
import torch.nn as nn
import flask_trainer as trainer
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
from config import config
from dkt.dataloader import Preprocess

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

# [ Bentoml serving ]
from lstm_classifier import LstmClassifier

lstm_classifier_service = LstmClassifier()
lstm_classifier_service.args=args

# packaging
lstm_classifier_service.pack('model', model)

# save 
saved_path = lstm_classifier_service.save()