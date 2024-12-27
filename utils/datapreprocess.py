import os
import pandas as pd
import numpy as np
import torch.nn.functional as ff
import torch
from utils import function as fc
from sklearn.preprocessing import MinMaxScaler, StandardScaler

import re

from sklearn.utils import resample

def balance_data(train_x, train_y, target_count):
    """
    平衡数据集：对少数类进行过采样，对多数类进行欠采样。
    
    Args:
        train_x (torch.Tensor): 输入数据。
        train_y (torch.Tensor): 对应的标签。
        target_count (int): 每个类别的目标样本数量。
        
    Returns:
        balanced_x, balanced_y: 平衡后的数据和标签。
    """
    # 将数据转换为 numpy 格式，方便处理
    train_x = np.array(train_x)
    train_y = np.array(train_y)

    # 初始化存储平衡数据的列表
    balanced_x = []
    balanced_y = []

    # 获取所有类别及其索引
    unique_classes, class_counts = np.unique(train_y, return_counts=True)

    for cls in unique_classes:
        # 提取当前类别的样本
        class_indices = np.where(train_y == cls)[0]
        class_x = train_x[class_indices]
        class_y = train_y[class_indices]

        if len(class_indices) > target_count:
            # 对多数类进行欠采样
            if cls == 4:
                resampled_x, resampled_y = resample(
                    class_x, class_y, n_samples=target_count * 3, random_state=42, replace=False
                )
            else:
                resampled_x, resampled_y = resample(
                    class_x, class_y, n_samples=target_count * 2, random_state=42, replace=False
                )
        else:
            # 对少数类进行过采样
            resampled_x, resampled_y = resample(
                class_x, class_y, n_samples=target_count, random_state=42, replace=True
            )

        # 将平衡后的样本添加到列表中
        balanced_x.append(resampled_x)
        balanced_y.append(resampled_y)

    # 将所有类别的样本合并
    balanced_x = np.vstack(balanced_x)
    balanced_y = np.hstack(balanced_y)

    # 随机打乱数据
    shuffle_indices = np.random.permutation(len(balanced_x))
    balanced_x = balanced_x[shuffle_indices]
    balanced_y = balanced_y[shuffle_indices]

    return torch.tensor(balanced_x, dtype=torch.float32), torch.tensor(balanced_y, dtype=torch.long)


def loadPrediction(data_dir, input_list, input_len):
    train_x = []
    for path, file_dir, files in os.walk(data_dir):
        for file_name in files:
            df = pd.read_csv(os.path.join(path, file_name))
            try:
                df = df[input_list]
            except:
                print(file_name)

            if df.shape[0] < 100:  # 数据不够长的不要
                print('需要预测的数据长度小于100，请修改数据！！！')
                break

            if len(input_list) == 3:
                df = df[(df['Primus/MAC'] >= 0.3) & (df['BIS/BIS'] > 0) & (df['Solar8000/ART_MBP'] > 0)]
            elif len(input_list) == 5:
                df = df[(df['Primus/MAC'] >= 0.3) & (df['BIS/BIS'] > 0) & (df['Solar8000/ART_MBP'] > 0)
                        & (df[input_list[3]] > 0) & (df[input_list[4]] > 0)]
            elif len(input_list) == 6:
                df = df[(df['Primus/MAC'] >= 0.3) & (df['BIS/BIS'] > 0) & (df['Solar8000/ART_MBP'] > 0)
                        & (df[input_list[3]] > 0) & (df[input_list[4]] > 0) & (df[input_list[5]] > 0)]
            else:
                print("input_list!!!!!")

            df = df.iloc[-input_len:, :]
            train_x.append(df.values.tolist())

    # 归一化数据到0-1
    mm = MinMaxScaler(feature_range=(0, 1))

    train_x = np.array(train_x, dtype='float32')  # (1,100,3) 3/5/6
    i, j, k = train_x.shape
    train_x = train_x.reshape(-1, k)
    train_x = mm.fit_transform(train_x)
    train_x = train_x.reshape(i, j, k)

    train_x = torch.from_numpy(train_x)

    return train_x, mm




def loadData(data_dir, input_list, t, data_long=10, data_jiange=5, type='regression', state='train'):
    if type == 'regression':
        train_x = []
        train_y = []
        data_long = data_long
        data_jiange = data_jiange
        len_dataset = 0
        for path, file_dir, files in os.walk(data_dir):
            for file_name in files:
                df = pd.read_csv(os.path.join(path, file_name))
                if input_list[-1].endswith('CI'):
                    for s in df.columns:
                        if re.search('.*CI$', s):
                            input_list[-1] = s
                try:
                    df = df[input_list]
                except:
                    print(file_name)
                if df.shape[0] >= (data_long + t):  # 数据不够长的不要

                    if len(input_list) == 3:
                        df = df[(df['Primus/MAC'] >= 0.3) & (df['BIS/BIS'] > 0) & (df['Solar8000/ART_MBP'] > 0)]
                    elif len(input_list) == 5:
                        df = df[(df['Primus/MAC'] >= 0.3) & (df['BIS/BIS'] > 0) & (df['Solar8000/ART_MBP'] > 0)
                                & (df[input_list[3]] > 0) & (df[input_list[4]] > 0)]
                    elif len(input_list) == 6:
                        df = df[(df['Primus/MAC'] >= 0.3) & (df['BIS/BIS'] > 0) & (df['Solar8000/ART_MBP'] > 0)
                                & (df[input_list[3]] > 0) & (df[input_list[4]] > 0) & (df[input_list[5]] > 0)]
                    elif len(input_list) == 7:
                        df = df[(df['Primus/MAC'] >= 0.3) & (df['BIS/BIS'] > 0) & (df['Solar8000/ART_MBP'] > 0)
                                & (df[input_list[3]] > 0) & (df[input_list[4]] > 0) & (df[input_list[5]] > 0) & (df[input_list[6]] > 0)]
                    else:
                        print("input_list!!!!!")
                    len_dataset += 1
                    df = df.reset_index(drop=True)

                    # 按照data_jiange间隔提取训练数据和目标数据
                    for i in range(df.shape[0] - data_long - t, data_long, -data_jiange): 
                        # 提取时间序列数据作为输入
                        train_x.append(df.iloc[i - data_long:i].values.tolist())
                        # 提取目标数据（t分钟后的目标）
                        train_y.append(df.iloc[i + t - 1:i + t, :len(input_list)].values.tolist()[0])

        print('the num of data:', len_dataset)
        print('total data len:', len(train_x))
        mm = MinMaxScaler(feature_range=(0, 1))
        train_x = np.array(train_x, dtype='float32')  # (369,100,3) 3/5/6
        i, j, k = train_x.shape
        train_y = np.array(train_y, dtype='float32')
        train_x = train_x.reshape(-1, k)
        data = np.append(train_x, train_y, axis=0)
        data = mm.fit_transform(data)
        train_x = data[:i*j]
        train_y = data[i*j:]
        train_x = train_x.reshape(i, j, k)
        train_x = torch.from_numpy(train_x)
        train_y = train_y.reshape(i, k)
        train_y = torch.from_numpy(train_y)

        return train_x, train_y, mm
    elif type == 'classification':
        train_x = []
        train_y = []
        data_long = data_long
        data_jiange = data_jiange  #
        len_dataset = 0
        for path, file_dir, files in os.walk(data_dir):
            for file_name in files:
                df = pd.read_csv(os.path.join(path, file_name))
                if input_list[-1].endswith('CI'):
                    for s in df.columns:
                        if re.search('.*CI$', s):
                            input_list[-1] = s
                try:
                    df = df[input_list]
                except:
                    print(file_name)

                if df.shape[0] >= (data_long + t):  # 数据不够长的不要

                    if len(input_list) == 3:
                        df = df[(df['Primus/MAC'] >= 0.3) & (df['BIS/BIS'] > 0) & (df['Solar8000/ART_MBP'] > 0)]
                    elif len(input_list) == 5:
                        df = df[(df['Primus/MAC'] >= 0.3) & (df['BIS/BIS'] > 0) & (df['Solar8000/ART_MBP'] > 0)
                                & (df[input_list[3]] > 0) & (df[input_list[4]] > 0)]
                    elif len(input_list) == 6:
                        df = df[(df['Primus/MAC'] >= 0.3) & (df['BIS/BIS'] > 0) & (df['Solar8000/ART_MBP'] > 0)
                                & (df[input_list[3]] > 0) & (df[input_list[4]] > 0) & (df[input_list[5]] > 0)]
                    else:
                        print("input_list!!!!!")

                    len_dataset += 1
                    df = df.reset_index(drop=True)

                    # 按照data_jiange间隔提取训练数据和目标数据
                    for i in range(df.shape[0] - data_long - t, data_long, -data_jiange): 
                        # 提取时间序列数据作为输入
                        train_x.append(df.iloc[i - data_long:i].values.tolist())
                        # 提取目标数据（t分钟后的目标）
                        train_y.append(df.iloc[i + t - 1:i + t, :len(input_list)].values.tolist()[0])

        # 处理train_y
        print('the num of data:', len_dataset)
        print('total data len:', len(train_x))

        train_y = fc.getClass(train_y)  # 将训练数据标签转化为类别
        if state == 'train':
            train_x, train_y = balance_data(train_x, train_y, target_count=len(train_x)//5)
        train_y_onehot = np.eye(5)[train_y]
        train_y_onehot = train_y_onehot.astype('float32')
        train_y_onehot = torch.from_numpy(train_y_onehot)

        # 处理train_x
        train_x = np.array(train_x, dtype='float32')
        train_x = torch.from_numpy(train_x)

        # # 对train_x添加归一化
        # mm = StandardScaler()
        # # mm = MinMaxScaler()
        # train_x = np.array(train_x, dtype='float32')
        # i, j, k = train_x.shape
        # train_x = train_x.reshape(-1, k)
        # train_x = mm.fit_transform(train_x)
        # train_x = train_x.reshape(i, j, k)

        return train_x, train_y_onehot
