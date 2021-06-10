import os
import torch
import sys
import os.path as p
import numpy as np
from torchvision import transforms
from flask import Flask, jsonify, request
import datetime
# from dkt.model import LSTM
from flask_trainer import load_model, get_model
import torch.nn as nn
from flask import Flask, render_template
from flask import request
import flask_inference
import sys
sys.path.append(os.path.dirname(os.path.abspath(os.path.dirname(__file__))))
from config_dir import config
from args import parse_args
from train import YamlConfigManager

app = Flask(__name__)

@app.route('/', methods=['GET'])
def index():
    return render_template('index.html')


# @app.route('/inference', methods=['POST'])
# def inference():
#     print('Execute inference....')
#     data = request.json
#     # 이 로직이 비어있음. data랑 result를 연관시키는 법 
#     result = main(args)
#     print('Ready to response....')
#     return result


@app.route('/get_score', methods=['POST'])
def get_score():
    data = request.json
    user_data = []
    print(data)
    t = str(datetime.datetime.now())[:-7]#.strftime('%Y-%m-%d %H:%M:%S') # plus
    
    for d in data:
        if 'answer' in d:
            row = [d['assess_id'], d['test_id'],d['tag'], t, d['answer']]
            user_data.append(row)
     
    print(user_data)
    score = flask_inference.inference(user_data)
    score = int(score)
    return str(score)

if __name__ == '__main__':
    # args = parse_args(mode='test')
    # cfg = YamlConfigManager(args.config_path, args.config)
    # config.set_args(args,cfg)

    # # load model
    # model = nn.LSTM(64,64,2,batch_first=True)
    # model_path = p.join('/opt/ml/code/models', 'lstm.pt')
    # model.load_state_dict(torch.load(model_path, map_location=torch.device('cpu')),strict=False)
    # model.eval()

    app.run(host='0.0.0.0', port=2431, threaded=False)





