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

def set_args(args, cfg):
    args.seed = cfg.values.seed
    args.device = cfg.values.device
    args.data_dir = cfg.values.data_dir
    args.asset_dir = cfg.values.asset_dir
    args.file_name = cfg.values.file_name
    args.model_dir = cfg.values.model_dir
    args.model_name = cfg.values.model_name
    args.output_dir = cfg.values.output_dir
    args.test_file_name = cfg.values.test_file_name  
    args.max_seq_len = cfg.values.max_seq_len
    args.num_workers = cfg.values.num_workers
    #model arguments
    args.model = cfg.values.model_args.model
    args.hidden_dim = cfg.values.model_args.hidden_dim
    args.n_layers = cfg.values.model_args.n_layers
    args.n_heads =  cfg.values.model_args.n_heads
    args.drop_out = cfg.values.model_args.drop_out
    #train arguments
    args.n_epochs = cfg.values.train_args.n_epochs
    args.batch_size = cfg.values.train_args.batch_size
    args.lr = cfg.values.train_args.lr
    args.clip_grad = cfg.values.train_args.clip_grad
    args.patience = cfg.values.train_args.patience
    args.log_steps = cfg.values.train_args.log_steps
    args.optimizer = cfg.values.train_args.optimizer
    args.scheduler = cfg.values.train_args.scheduler
    #wandb argument
    args.name = cfg.values.wandb.name