# Config

## | Train / Inference
### - train.py
config 파일 위치 (default : ./config/config.yml)

`
python train.py --config_path {file_dir/file_name.yml}
`

#### model (default : lstm)

`
python train.py --config {root}
`
### - inference.py
#### config 파일 위치 (default : ./config/config.yml)

`
python inference.py --config_path {file_dir/file_name.yml}
`

#### model (default : lstm)

`
python inference.py --config {root}
`


## | Config.yml
- model_dir: model이 저장되는 directory
- model_name: model 저장명
- model: 'LSTM' load할 model class
### ex) LSTM
```
lstm: # root
  seed: 42
  device: 'cpu'
  data_dir: '/opt/ml/input/data/train_dataset'
  asset_dir: 'asset/'
  file_name: 'train_data.csv'
  model_dir: 'models/' # model directory
  model_name: 'lstm.pt' # save model as model_name
  output_dir: 'output/'
  test_file_name: 'test_data.csv'  
  max_seq_len: 20
  num_workers: 1
  model_args:
    model: 'LSTM' # model class name
    hidden_dim: 64
    n_layers: 2
    n_heads: 2
    drop_out: 0.2
  train_args:
    n_epochs: 20
    batch_size: 64
    lr: 0.0001
    clip_grad: 10
    patience: 5
    log_steps: 50
    optimizer: 'adam'
    scheduler: 'plateau'
  wandb:
    name: 'lstm' # wandb run name
```
