import os
import yaml
from args import parse_args
from dkt.dataloader import Preprocess
from dkt import trainer
import torch
from dkt.utils import setSeeds
from easydict import EasyDict
import wandb
from prettyprinter import cpprint
from config import config

# Set Config
class YamlConfigManager:
    def __init__(self, config_file_path, config_name):
        super().__init__()
        self.values = EasyDict()        
        if config_file_path:
            self.config_file_path = config_file_path
            self.config_name = config_name
            self.reload()
    
    def reload(self):
        self.clear()
        if self.config_file_path:
            with open(self.config_file_path, 'r') as f:
                self.values.update(yaml.safe_load(f)[self.config_name])

    def clear(self):
        self.values.clear()
    
    def update(self, yml_dict):
        for (k1, v1) in yml_dict.items():
            if isinstance(v1, dict):
                for (k2, v2) in v1.items():
                    if isinstance(v2, dict):
                        for (k3, v3) in v2.items():
                            self.values[k1][k2][k3] = v3
                    else:
                        self.values[k1][k2] = v2
            else:
                self.values[k1] = v1

    def export(self, save_file_path):
        if save_file_path:
            with open(save_file_path, 'w') as f:
                yaml.dump(dict(self.values), f)

def main(args):
     
    setSeeds(42)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    args.device = device
    if torch.cuda.is_available():
        print(f'We will use the GPU : {torch.cuda.get_device_name(0)}')

    wandb.login()
    preprocess = Preprocess(args)
    preprocess.load_train_data(args.file_name)
    train_data = preprocess.get_train_data()
    train_data, valid_data = preprocess.split_data(train_data)
    # train_data, valid_data = preprocess.load_data_from_file_2(args.file_name)
    
    wandb.init(project='dkt', config=vars(args), name=args.name)
    trainer.run(args, train_data, valid_data)
    

if __name__ == "__main__":
    args = parse_args(mode='train')
    cfg = YamlConfigManager(args.config_path, args.config)
    config.set_args(args,cfg)
    os.makedirs(args.model_dir, exist_ok=True)
    main(args)