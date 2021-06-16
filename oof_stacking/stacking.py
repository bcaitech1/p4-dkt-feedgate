import os
import sys

import pandas as pd
import numpy as np
import torch

from sklearn.linear_model import LinearRegression
import warnings
warnings.filterwarnings("ignore")

GT_path = './answerCode.csv'
# ground truths
GT = pd.read_csv(GT_path)['answerCode'].to_numpy()

# 보미님
val = np.load('./val_infer.npy') 
test = np.load('./test_infer.npy')

# 평화님 
val1 = np.load('./tj_val_infer.npy') 
test1 = np.load('./test_infer.npy')

val_list = [val, val1]
test_list = [test, test1]

##########학습.
S_train = None       #학습에 쓸 데이터 [X]

for i, train_oof in enumerate(val_list):
    train_oof = train_oof.reshape(-1, 1)

    if not isinstance(S_train, np.ndarray):
        S_train = train_oof
    else:
        S_train = np.concatenate([S_train, train_oof], axis=1)
#학습 시작.
meta_model.fit(S_train, GT)

