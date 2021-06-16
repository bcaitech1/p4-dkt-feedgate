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

##########추론

S_test = None       #학습에 쓸 데이터 [X]
for i, test in enumerate(test_list):
    test = test.reshape(-1, 1)
#     print(test)
    if not isinstance(S_test, np.ndarray):
        S_test = test
    else:
        S_test = np.concatenate([S_test, test], axis=1)       

predict = meta_model.predict(S_test)

df = pd.DataFrame(predict)

write_path ="output.csv"
with open(write_path, 'w', encoding='utf8') as w:
    print("writing prediction : {}".format(write_path))
    w.write("id,prediction\n")
    for id, p in enumerate(predict):
        w.write('{},{}\n'.format(id,p))
