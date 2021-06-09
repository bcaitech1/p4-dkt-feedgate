#!/usr/bin/env bash

seed=42

device=gpu
#data path
data_dir='/opt/ml/input/data/train_dataset'
asset_dir='asset/'
file_name='train_data.csv'
#model path
model_dir='models/'
model_name='model.pt'
#output path.
output_dir='output/'
test_file_name='test_data.csv'

max_seq_len=20
num_workers=1


hidden_dim=64
n_layers=2
n_heads=2
drop_out=0.2


n_epochs=20
batch_size=64
lr=0.001
clip_grad=10
patience=5

log_steps=50
model='lstm'
optimizer='adam'
scheduler='plateau'

#wandb
name='testmodel'

config_path='./config/config.yml'
config='lstm'

ARGS="
--seed $seed 

--device $device 

--data_dir $data_dir 
--asset_dir $asset_dir 
--file_name $file_name 

--model_dir $model_dir 
--model_name $model_name 

--output_dir $output_dir 
--test_file_name $test_file_name 

--max_seq_len $max_seq_len 
--num_workers $num_workers 

--hidden_dim $hidden_dim 
--n_layers $n_layers 
--n_heads $n_heads 

--n_epochs $n_epochs 
--batch_size $batch_size 
--clip_grad $clip_grad 
--patience $patience 

--log_steps $log_steps 
--model $model 
--optimizer $optimizer 
--scheduler $scheduler 

--name $name 

--config_path $config_path 
--config $config
"
#--lr= $lr \
#--drop_out $drop_out 

python train.py $ARGS
