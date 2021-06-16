import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns; sns.set_theme(color_codes=True)
import os1


FILE_PATH = '../input/data/train_dataset/train_data.csv'

dtype = {
    'userID': 'int16',
    'answerCode': 'int8',
    'KnowledgeTag': 'int16'
}   

df = pd.read_csv(, dtype=dtype, parse_dates=['Timestamp'])
df = df.sort_values(by=['userID', 'Timestamp']).reset_index(drop=True)

def percentile(s):
    return np.sum(s) / len(s)

group = df.groupby('userID').agg({
    'assmtID': 'count',
    'answerCode': percentile
})

group.rename(columns = {'assmtID': 'assmt_cnt', 'answerCode': 'answer_rate'}, inplace = True)

def get_fold_score(df):
    score = df['assmt_cnt']*4 + df['answer_rate']*3
    return score


df = pd.merge(df,group,on='userID')
df.head()

#유저별 마지막 로우만 추출 
temp = pd.DataFrame(columns=list(df.columns))
for i in range(0, 7442):
    temp=temp.append(df[df.userID == i][len(df[df.userID == i])-1:])
temp=temp.astype({'assmt_cnt':'int'})


# 10분위수 구하기.
q2 = 0

for i in [0.1,0.2,0.3,0.4,0.5,0.6,0.7,0.8,0.9,1.0]:
    q1 = q2
    q2 = temp['fold_score'].quantile(i)
    print(f'{i}분위수의 정답률')
    print(temp[(temp['fold_score']>=q1) & (temp['fold_score']<q2)]['answerCode'][temp['answerCode']==1].count()/temp[(temp['fold_score']>=q1) & (temp['fold_score']<q2)]['answerCode'].count())

fold_quantile =[]
for i in [0,0.1,0.2,0.3,0.4,0.5,0.6,0.7,0.8,0.9,1.0]:
    fold_quantile.append(temp['fold_score'].quantile(i))

def get_fold_class(df):
    # 10분위수 구하기.
    for i, val in enumerate(fold_quantile):
        if df['fold_score']<val:
            return int(i)
    
    return 10

temp['fold_class'] = temp.apply(get_fold_class, axis=1)
temp = temp.astype({'fold_class':'int'})
output = temp.drop(['testID','assmtID','timestamp','knowledgeTag','answerCode','assmt_cnt','answer_rate','fold_score'],axis=1)

CHANGE_FILE_NAME = 'user_fold_class.csv'
# Write data.
path = os.path.join(DATA_DIR , CHANGE_FILE_NAME)
output.to_csv(path, index=False)

