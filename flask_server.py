import os
import torch
import sys
import os.path as p
import numpy as np
from torchvision import transforms
from flask import Flask, jsonify, request
import datetime
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

@app.route('/get_score', methods=['POST'])
def get_score():
    data = request.json
    user_data = []
    print(data)
    t = str(datetime.datetime.now())[:-7] # make current time-stamp for str type
    
    for d in data:
        if 'answer' in d:
            row = [d['assess_id'], d['test_id'],d['tag'], t, d['answer']]
            user_data.append(row)
     
    print(user_data)
    score = flask_inference.inference(user_data)
    score = int(score)
    return str(score)

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=2431, threaded=False)





