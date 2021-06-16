import numpy as np
import os
import torch
from sklearn.model_selection import StratifiedKFold

def get_fold_score(datas):
    targets = []
    for data in datas:
        targets.append(data[-1][-1]) # fold score 기준이 되는 열을 반환해주세요.
    return np.array(targets)


def seed_everything(seed=42):
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True


def evaluate( args, model, valid_data):
    """훈련된 모델과 validation 데이터셋을 제공하면 predict 반환"""
    pin_memory = False
    valset = DKTDataset(valid_data, args)
    valid_loader = torch.utils.data.DataLoader(valset, shuffle=False,
                                                batch_size=args.batch_size,
                                                pin_memory=pin_memory,
                                                collate_fn=collate)
    auc, acc, preds, _ = validate(valid_loader, model, args)
    print(f"AUC : {auc}, ACC : {acc}")
    return preds


kfold = StratifiedKFold(n_splits=5)
# data = train data
# train
oof = np.zeros(data.shape[0])
fold_models = []
target = get_fold_score(datas)
for i, (train_index, valid_index) in enumerate(kfold.split(data, target)):
    train_data, valid_data = data[train_index], data[valid_index]
    print(f'Calculating train oof {i + 1}')
    #torch.cuda.empty_cache()
    #gc.collect()
    # 모델 생성 및 훈련
    args.model_name = 'fold_' + str(i) + '.pt'
    run(args, train_data, valid_data)
    # 학습이 끝나면 return model 해서 넣으셔도 됩니다. -> # trained_mdoel =run(args, train_data, valid_data)
    trained_model = load_model(args)
    # fold별 oof 값 모으기
    predict = evaluate(args, train_mdoel, valid_data)
    oof[valid_index] = predict #<- valid_data을 model로 infer한 값이여야합니다.
    fold_models.append(trained_model)
    oof[valid_index] = predict
oof_val_np_path = './oof/np/val_infer.npy'
np.save(oof_val_np_path, oof)
predicts = np.zeros(test_data.shape[0])
for i, model in enumerate(fold_models):
    print(f'Calculating train oof {i + 1}')
    predict = test(args, model, test_data)#테스트 데이터 예측.
    predicts +=predict
predict_avg = predicts/ len(fold_mdoels)
oof_predict_np_path = './oof/np/test_infer.npy'
np.save(oof_predict_np_path, predict_avg)


