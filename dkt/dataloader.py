import os
from datetime import datetime
import time
import tqdm
import pandas as pd
import random
from sklearn.preprocessing import LabelEncoder
import numpy as np
import torch
from torch.nn.utils.rnn import pad_sequence

class Preprocess:
    def __init__(self,args):
        self.args = args
        self.train_data = None
        self.test_data = None

    def get_train_data(self):
        return self.train_data

    def get_test_data(self):
        return self.test_data

    def split_data(self, data, ratio=0.9, shuffle=True, seed=0):
        """
        split data into two parts with a given ratio.
        """
        if shuffle:
            random.seed(seed) # fix to default seed 0
            random.shuffle(data)

        size = int(len(data) * ratio)
        data_1 = data[:size]
        data_2 = data[size:]

        return data_1, data_2

    def __save_labels(self, encoder, name):
        le_path = os.path.join(self.args.asset_dir, name + '_classes.npy')
        np.save(le_path, encoder.classes_)

<<<<<<< HEAD
    def __preprocessing(self, df, is_train=False):
        cate_cols = ['assessmentItemID', 'testId', 'KnowledgeTag']
=======
    def __preprocessing(self, df, is_train = True):
        cate_cols = ['assessmentItemID', 'testId', 'KnowledgeTag', 'category', 'number']
>>>>>>> 984057abd5c6e19299ffa8869d023aa341e19f04

        if not os.path.exists(self.args.asset_dir):
            os.makedirs(self.args.asset_dir)
            
        for col in cate_cols:
<<<<<<< HEAD
            
=======
>>>>>>> 984057abd5c6e19299ffa8869d023aa341e19f04
            le = LabelEncoder()
            
            if is_train:
                # for unknown class
                a = df[col].unique().tolist() + ['unknown']
                le.fit(a)
                self.__save_labels(le,col)
            else:
                label_path = os.path.join(self.args.asset_dir,col+'_classes.npy')
                le.classes_ = np.load(label_path)
                df[col] = df[col].apply(lambda x: x if str(x) in le.classes_ else 'unknown')

            #모든 컬럼이 범주형이라고 가정
            df[col]= df[col].astype(str)
            test = le.transform(df[col])
            df[col] = test
<<<<<<< HEAD

        def convert_time(s):
            timestamp = time.mktime(datetime.strptime(s, '%Y-%m-%d %H:%M:%S').timetuple())
            return int(timestamp)

        df['Timestamp'] = df['Timestamp'].apply(convert_time)
        print('Preprocess Done....')
        return df

    def __feature_engineering(self, df):
        #TODO
        return df
=======
            
        
        return df

    def __feature_engineering(self, df):
>>>>>>> 984057abd5c6e19299ffa8869d023aa341e19f04

    def load_data(self, df):
        """
            (Input)
                - df : dataframe of user data. 
            (Caution)
                - If you want to load csv (not dataframe), use `load_data_from_file()`. 
        """
        df = self.__feature_engineering(df)
        df = self.__preprocessing(df)
        
        self.args.n_questions = len(np.load(os.path.join(self.args.asset_dir,'assessmentItemID_classes.npy')))
        self.args.n_test = len(np.load(os.path.join(self.args.asset_dir,'testId_classes.npy')))
        self.args.n_tag = len(np.load(os.path.join(self.args.asset_dir,'KnowledgeTag_classes.npy')))
        
        columns = ['userID', 'assessmentItemID', 'testId', 'answerCode', 'KnowledgeTag']
        group = df[columns].groupby('userID').apply(
                lambda r: (
                    r['testId'].values, 
                    r['assessmentItemID'].values,
                    r['KnowledgeTag'].values,
                    r['answerCode'].values
                )
            )

        return group.values

    def load_data_from_file(self, file_name, is_train=True):
        csv_file_path = os.path.join(self.args.data_dir, file_name)
        df = pd.read_csv(csv_file_path)#, nrows=100000)
        df = self.__feature_engineering(df)
        df = self.__preprocessing(df, is_train)

<<<<<<< HEAD
        self.args.n_questions = len(np.load(os.path.join(self.args.asset_dir,'assessmentItemID_classes.npy')))
        self.args.n_test = len(np.load(os.path.join(self.args.asset_dir,'testId_classes.npy')))
        self.args.n_tag = len(np.load(os.path.join(self.args.asset_dir,'KnowledgeTag_classes.npy')))


        df = df.sort_values(by=['userID','Timestamp'], axis=0)
        columns = ['userID', 'assessmentItemID', 'testId', 'answerCode', 'KnowledgeTag']
        group = df[columns].groupby('userID').apply(
                lambda r: (
                    r['testId'].values, 
                    r['assessmentItemID'].values,
                    r['KnowledgeTag'].values,
=======
        # 추후 feature를 embedding할 시에 embedding_layer의 input 크기를 결정할때 사용
        self.args.n_questions = len(np.load(os.path.join(self.args.asset_dir,'assessmentItemID_classes.npy')))
        self.args.n_test = len(np.load(os.path.join(self.args.asset_dir,'testId_classes.npy')))
        self.args.n_tag = len(np.load(os.path.join(self.args.asset_dir,'KnowledgeTag_classes.npy')))
        self.args.n_category = len(np.load(os.path.join(self.args.asset_dir,'category_classes.npy')))
        self.args.n_number = len(np.load(os.path.join(self.args.asset_dir,'number_classes.npy')))


        df = df.sort_values(by=['userID','Timestamp'], axis=0)
        columns = ['userID', 'assessmentItemID', 'testId', 'answerCode', 'Timestamp','KnowledgeTag', 
                    'user_ans', 'user_cnt', 'elapsed_time', 'category','cate_ans', 'cate_cnt', 'number', 'test_ans', 'test_cnt', 'test_cumsum',
                    'tag_ans', 'tag_cnt','tag_mean']
        group = df[columns].groupby('userID').apply(
                lambda r: (
                    r['testId'].values, 
                    r['Timestamp'].values,
                    r['assessmentItemID'].values,
                    r['KnowledgeTag'].values,
                    r['elapsed_time'].values,
                    r['test_ans'].values,
                    r['user_ans'].values,
                    r['user_cnt'].values,
>>>>>>> 984057abd5c6e19299ffa8869d023aa341e19f04
                    r['answerCode'].values
                )
            )

        return group.values

    def load_test_data(self, data):
        self.test_data = self.load_data(data)

    def load_train_data_csv(self, file_name):
        self.train_data = self.load_data_from_file(file_name)

    def load_test_data_csv(self, file_name):
        self.train_data = self.load_data_from_file(file_name, is_train=False)
        
class DKTDataset(torch.utils.data.Dataset):
    def __init__(self, data, args):
        self.data = data
        self.args = args

    def __getitem__(self, index):
        row = self.data[index]

        # 각 data의 sequence length
        seq_len = len(row[0])

<<<<<<< HEAD
        test, question, tag, correct = row[0], row[1], row[2], row[3]
        

        cate_cols = [test, question, tag, correct]
=======
        test, time, question, tag, elapsed_time, test_ans, user_ans, user_cnt, correct = row[0], row[1], row[2], row[3], row[4], row[5], row[6], row[7], row[8]

        cate_cols = [test, time, question, tag, elapsed_time, test_ans,  user_ans, user_cnt, correct]
>>>>>>> 984057abd5c6e19299ffa8869d023aa341e19f04

        # max seq len을 고려하여서 이보다 길면 자르고 아닐 경우 그대로 냅둔다

        if seq_len > self.args.max_seq_len:
            for i, col in enumerate(cate_cols):
                cate_cols[i] = col[-self.args.max_seq_len:]
            mask = np.ones(self.args.max_seq_len, dtype=np.int16)
        else:
            mask = np.zeros(self.args.max_seq_len, dtype=np.int16)
            mask[-seq_len:] = 1
        
        # reg_time
        for i in range(len(cate_cols[1])):
            cate_cols[1][i] = cate_cols[1][len(cate_cols[1])-1] - cate_cols[1][i]

        # mask도 columns 목록에 포함시킴
        cate_cols.append(mask)


        # np.array -> torch.tensor 형변환
        for i, col in enumerate(cate_cols):
            cate_cols[i] = torch.tensor(col)


        return cate_cols

    def __len__(self):
        return len(self.data)



def collate(batch):
    col_n = len(batch[0])
    col_list = [[] for _ in range(col_n)]
    max_seq_len = len(batch[0][-1])
<<<<<<< HEAD
        
=======


>>>>>>> 984057abd5c6e19299ffa8869d023aa341e19f04
    # batch의 값들을 각 column끼리 그룹화
    for row in batch:
        for i, col in enumerate(row):
            pre_padded = torch.zeros(max_seq_len)
            pre_padded[-len(col):] = col
            col_list[i].append(pre_padded)


    for i, _ in enumerate(col_list):
        col_list[i] =torch.stack(col_list[i])
    
    return tuple(col_list)

def slidding_window(data, args):
    window_size = args.max_seq_len
    stride = 10
    args.shuffle = True

    augmented_datas = []
    for row in data:
        seq_len = len(row[0])

        # 만약 window 크기보다 seq len이 같거나 작으면 augmentation을 하지 않는다
        if seq_len <= window_size:
            augmented_datas.append(row)
        else:
            total_window = ((seq_len - window_size) // stride) + 1
            
            # 앞에서부터 slidding window 적용
            for window_i in range(total_window):
                # window로 잘린 데이터를 모으는 리스트
                window_data = []
                for col in row:
                    window_data.append(col[window_i*stride:window_i*stride + window_size])

                # Shuffle
                # 마지막 데이터의 경우 shuffle을 하지 않는다
                if args.shuffle and window_i + 1 != total_window:
                    shuffle_datas = shuffle(window_data, window_size, args)
                    augmented_datas += shuffle_datas
                else:
                    augmented_datas.append(tuple(window_data))

            # slidding window에서 뒷부분이 누락될 경우 추가
            total_len = window_size + (stride * (total_window - 1))
            if seq_len != total_len:
                window_data = []
                for col in row:
                    window_data.append(col[-window_size:])
                augmented_datas.append(tuple(window_data))


    return augmented_datas

def shuffle(data, data_size, args):
    shuffle_datas = []
    args.shuffle_n = 1
    for i in range(args.shuffle_n):
        # shuffle 횟수만큼 window를 랜덤하게 계속 섞어서 데이터로 추가
        shuffle_data = []
        random_index = np.random.permutation(data_size)
        for col in data:
            shuffle_data.append(col[random_index])
        shuffle_datas.append(tuple(shuffle_data))
    return shuffle_datas

def get_loaders(args, train, valid):

    pin_memory = True
    train_loader, valid_loader = None, None
    if train is not None:
        train = slidding_window(train, args)
        # print(type(train))
        trainset = DKTDataset(train, args)
        train_loader = torch.utils.data.DataLoader(trainset, num_workers=args.num_workers, shuffle=True,
                            batch_size=args.batch_size, pin_memory=pin_memory, collate_fn=collate)
    if valid is not None:
        valset = DKTDataset(valid, args)
        valid_loader = torch.utils.data.DataLoader(valset, num_workers=args.num_workers, shuffle=False,
                            batch_size=args.batch_size, pin_memory=pin_memory, collate_fn=collate)
    


    return train_loader, valid_loader