# import torch
# import sys
# import os.path as p
# import numpy as np
# from config import config
# from torchvision import transforms
# from flask import Flask, jsonify, request
# from args import parse_args
# from dkt.model import LSTM
# from dkt.trainer import load_model, get_model
# from train import YamlConfigManager
# from inference import main
# import torch.nn as nn


# app = Flask(__name__)
# @app.route('/inference', methods=['POST'])
# def inference():
#     print('inference....!')
#     data = request.json
#     _, result = model.forward(normalize(np.array(data['images'], dtype=np.uint8)).unsqueeze(0)).max(1)
#     return str(result.item())


# if __name__ == '__main__':
#     args = parse_args(mode='test')
#     print(args)
#     cfg = YamlConfigManager(args.config_path, args.config)
#     config.set_args(args,cfg)

#     model = nn.LSTM()
#     model_path = p.join('/opt/ml/code/models', 'lstm.pt')
#     load_state = torch.load(model_path)
#     print(model)
#     model.load_state_dict(torch.load(model_path), strict=True)
   
    
#     app.run(host='0.0.0.0', port=2431, threaded=False)





