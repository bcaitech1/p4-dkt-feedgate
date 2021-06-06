import torch
import sys
import os.path as p
import numpy as np
from config import config
from torchvision import transforms
from flask import Flask, jsonify, request
from args import parse_args
from dkt.model import LSTM
from dkt.trainer import load_model, get_model
from train import YamlConfigManager
from inference import main
import torch.nn as nn


app = Flask(__name__)
@app.route('/inference', methods=['POST'])
def inference():
    print('Execute inference....')
    data = request.json
    # 이 로직이 비어있음. data랑 result를 연관시키는 법 
    result = main(args)
    print('Ready to response....')
    return result


if __name__ == '__main__':
    args = parse_args(mode='test')
    cfg = YamlConfigManager(args.config_path, args.config)
    config.set_args(args,cfg)

    # # load model
    # model = nn.LSTM(64,64,2,batch_first=True)
    # model_path = p.join('/opt/ml/code/models', 'lstm.pt')
    # model.load_state_dict(torch.load(model_path),strict=False)
    # model.eval()

    app.run(host='0.0.0.0', port=2431, threaded=False)





