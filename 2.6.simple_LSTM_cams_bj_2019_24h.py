import warnings
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from numba import njit
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm
warnings.filterwarnings('ignore')


################
# loading data #
################
def load_data(start, end, period):
    # load caqra data
    df = cams_data
    df.index = pd.to_datetime([str(i) for i in df['date'].values], format="%Y-%m-%d")
    df = df.iloc[:, 2:]
    # variables + obs
    interval = input_size + 1
    df = df.iloc[:, point_num * interval:point_num * interval + interval]
    # add exist sequence before start, similar to warmup period!
    df = df[start - pd.DateOffset(days=int(sequence_length / 24)):end]
    # extract variables[0:input_size] and obs[-1]
    x = df.iloc[:, 0:input_size].values
    x = x.reshape(x.shape[0], input_size)
    y = df.iloc[:, -1].values
    y = y.reshape(y.shape[0], 1)
    return x, y


####################
# get mean and std #
####################
def get_mean_std(start, end):
    # get mean and std in training period for normalize and rescale
    df = caqra_data
    df.index = pd.to_datetime([str(i) for i in df['date'].values], format="%Y-%m-%d")
    df = df.iloc[:, 2:]
    # variables + obs
    interval = input_size + 1
    df = df.iloc[:, point_num * interval:point_num * interval + interval]
    df_train = df[start: end]
    means = df_train.mean()
    stds = df_train.std()
    return means, stds


###################
# normalize data  #
###################
def normalize_data(input_data, varible: str):
    # mean and std only train period
    if varible == 'x':
        x_means = np.array(mean[:input_size])
        x_stds = np.array(std[:input_size])
        x_nor = (input_data - x_means) / x_stds
        return x_nor
    if varible == 'y':
        y_means = np.array(mean[-1])
        y_stds = np.array(std[-1])
        y_nor = (input_data - y_means) / y_stds
        return y_nor


##################################
# reshape data for LSTM training #
##################################
@njit  # decorator JIT-compiles for speeding up !!!
def reshape_data(x, y):
    s = sequence_length  # sequence length
    n = x.shape[0]  # samples
    m = x.shape[1]  # features
    x_new = np.zeros((n - s, s, m))
    y_new = np.zeros((n - s, 1))
    # !!! remove last sequence to avoid "0" !!!
    for i in range(x_new.shape[0]):
        x_new[i, :, :] = x[i: i + s, :]
        y_new[i, :] = y[i + s]
    return x_new, y_new


#################################
# delete data where Nan or -999 #
#################################
def check_data(x, y):
    # delete NaN or -999, because check from y, so first x then y
    x = np.delete(x, np.argwhere(np.isnan(y)), axis=0)
    y = np.delete(y, np.argwhere(np.isnan(y)), axis=0)
    x = np.delete(x, np.argwhere(y < 0)[:, 0], axis=0)
    y = np.delete(y, np.argwhere(y < 0)[:, 0], axis=0)
    return x, y


################
# tensor data  #
################
def torch_input(start, end, period):
    # load x and y
    x, y = load_data(start, end, period)
    # normalize x without missing
    x_normalize = normalize_data(x, 'x')
    # reshape x and y
    x_reshape, y_reshape = reshape_data(x_normalize, y)
    x_check, y_check = check_data(x_reshape, y_reshape)
    x_torch = torch.from_numpy(x_check.astype(np.float32))
    y_torch = torch.from_numpy(y_check.astype(np.float32))
    return x_torch, y_torch


####################
# PyTorch data set #
####################
class DATA(Dataset):
    # inherit from the Dataset class
    def __init__(self, start, end, period):
        self.x, self.y = torch_input(start, end, period)
        self.num_samples = self.x.shape[0]

    def __len__(self):
        return self.num_samples

    def __getitem__(self, item):
        return self.x[item], self.y[item]


####################
# Build LSTM model #
####################
class Model(nn.Module):
    # inherit from nn.Module
    def __init__(self, input_input_size, input_hidden_size, input_dropout_rate):
        super(Model, self).__init__()
        self.input_size = input_input_size
        self.hidden_size = input_hidden_size
        self.dropout_rate = input_dropout_rate
        self.lstm = nn.LSTM(input_size=self.input_size, hidden_size=self.hidden_size, num_layers=1, bias=True, batch_first=True)
        self.dropout = nn.Dropout(p=self.dropout_rate)
        self.fc = nn.Linear(in_features=self.hidden_size, out_features=1)

    def forward(self, x):
        output, (h_n, c_n) = self.lstm(x)  # hidden state and cell state
        predict = self.fc(self.dropout(h_n[-1, :, :]))
        return predict


##################
# training model #
##################
def train_model(model, optimizer, loader, loss_func):
    model.train()  # set model to train mode
    for x, y in loader:
        optimizer.zero_grad()  # delete previously stored gradients
        x, y = x.to(DEVICE), y.to(DEVICE)
        y_hat = model(x)
        loss = torch.sqrt(loss_func(y_hat, y))
        loss.backward()  # back-propagation
        optimizer.step()  # update the weights


####################
# evaluating model #
####################
def eval_model(model, loader):
    model.eval()  # set model to eval mode
    obs, preds = [], []
    with torch.no_grad():  # No backpropagation
        for x, y in loader:
            x, y = x.to(DEVICE), y.to(DEVICE)
            y_hat = model(x)
            obs.append(y)
            preds.append(y_hat)
    return torch.cat(obs), torch.cat(preds)


###################################
# Prepare everything for training #
###################################
def simple_lstm():
    # data set up
    test_data = DATA(test_start, test_end, 'test')
    test = DataLoader(test_data, batch_size=batch_size, shuffle=False)
    # model set up
    model = Model(input_input_size=input_size, input_hidden_size=hidden_size, input_dropout_rate=dropout_rate).to(DEVICE)
    model.load_state_dict(torch.load(para_path[num]))
    loss_func = nn.MSELoss()
    # testing set up
    obs, pred = eval_model(model, test)
    pred = pred * std[-1] + mean[-1]  # rescale prediction
    rmse = torch.sqrt(loss_func(obs, pred))
    return rmse.cpu().numpy(), pred.cpu().numpy()


if __name__ == "__main__":
    #########################
    # global hyperparameter #
    #########################
    DEVICE = torch.device("cuda:0")
    sequence_length = 24 * 30
    batch_size = 512
    hidden_size = 256
    dropout_rate = 0.5
    learning_rate = 0.001  # lr * 0.5 per 10 epoches
    epoch = 100
    point_nums = 34
    input_size = 11
    ##################
    # global dataset #
    ##################
    train_start_months = pd.to_datetime(["2014-01-01 00:00:00"] * 12, format="%Y-%m-%d %H:%M:%S")
    train_end_months = pd.to_datetime(["2018-06-30 23:00:00", "2018-07-31 23:00:00", "2018-08-31 23:00:00",
                                       "2018-09-30 23:00:00", "2018-10-31 23:00:00", "2018-11-30 23:00:00",
                                       "2018-12-31 23:00:00", "2019-01-31 23:00:00", "2019-02-28 23:00:00",
                                       "2019-03-31 23:00:00", "2019-04-30 23:00:00", "2019-05-31 23:00:00"],
                                      format="%Y-%m-%d %H:%M:%S")
    validate_start_months = pd.to_datetime(["2018-07-01 00:00:00", "2018-08-01 00:00:00", "2018-09-01 00:00:00",
                                            "2018-10-01 00:00:00", "2018-11-01 00:00:00", "2018-12-01 00:00:00",
                                            "2019-01-01 00:00:00", "2019-02-01 00:00:00", "2019-03-01 00:00:00",
                                            "2019-04-01 00:00:00", "2019-05-01 00:00:00", "2019-06-01 00:00:00"],
                                           format="%Y-%m-%d %H:%M:%S")
    validate_end_months = pd.to_datetime(["2018-12-31 23:00:00", "2019-01-31 23:00:00", "2019-02-28 23:00:00",
                                          "2019-03-31 23:00:00", "2019-04-30 23:00:00", "2019-05-31 23:00:00",
                                          "2019-06-30 23:00:00", "2019-07-31 23:00:00", "2019-08-31 23:00:00",
                                          "2019-09-30 23:00:00", "2019-10-31 23:00:00", "2019-11-30 23:00:00"],
                                         format="%Y-%m-%d %H:%M:%S")
    test_start_months = pd.to_datetime(["2019-01-01 00:00:00", "2019-02-01 00:00:00", "2019-03-01 00:00:00",
                                        "2019-04-01 00:00:00", "2019-05-01 00:00:00", "2019-06-01 00:00:00",
                                        "2019-07-01 00:00:00", "2019-08-01 00:00:00", "2019-09-01 00:00:00",
                                        "2019-10-01 00:00:00", "2019-11-01 00:00:00", "2019-12-01 00:00:00"],
                                       format="%Y-%m-%d %H:%M:%S")
    test_end_months = pd.to_datetime(["2019-01-31 23:00:00", "2019-02-28 23:00:00", "2019-03-31 23:00:00",
                                      "2019-04-30 23:00:00", "2019-05-31 23:00:00", "2019-06-30 23:00:00",
                                      "2019-07-31 23:00:00", "2019-08-31 23:00:00", "2019-09-30 23:00:00",
                                      "2019-10-31 23:00:00", "2019-11-30 23:00:00", "2019-12-31 23:00:00"],
                                     format="%Y-%m-%d %H:%M:%S")
    #############
    # file path #
    #############
    rmse_path = r'D:\2023.8.19-LSTM-BJ-Correction\beijing\rmse\rmse_simple_11_nan_avg_0_cams_bj_2019_24h.csv'
    pred_path = [r'D:\2023.8.19-LSTM-BJ-Correction\beijing\pred\pred_simple_11_nan_avg_0_cams_bj_2019_24h_'
                 + str(i+1) + '.csv' for i in range(12)]
    ###################
    # start go go go! #
    ###################
    caqra_data = pd.read_csv(r'D:\2023.8.19-LSTM-BJ-Correction\beijing\6.input\caqra_11_nan_avg_0.csv')
    cams_data = pd.read_csv(r'D:\2023.8.19-LSTM-BJ-Correction\beijing\6.input\cams_11_bj_2019_24h.csv')
    points = ['东四', '天坛', '官园', '万寿西宫', '奥体中心', '农展馆', '万柳', '北部新区', '丰台花园', '云岗',
              '古城', '房山', '大兴', '亦庄', '通州', '顺义', '昌平', '门头沟', '平谷', '怀柔', '密云', '延庆', '定陵',
              '八达岭', '密云水库', '东高村', '永乐店', '榆垡', '琉璃河', '前门', '永定门内', '西直门北', '南三环',
              '东四环']
    rmse_array = np.zeros([12, point_nums])
    for month in range(12):
        print('start ' + 'Month ' + str(month+1))
        train_start = train_start_months[month]
        train_end = train_end_months[month]
        test_start = test_start_months[month]
        test_end = test_end_months[month]
        para_path = [r'D:\2023.8.19-LSTM-BJ-Correction\beijing\para\para_simple_11_nan_avg_0_v6m_t1m_es_month_'
                     + str(month + 1) + '\\11_' + str(num) + '.pth' for num in range(point_nums)]
        pred_array = np.zeros([pd.date_range(test_start, test_end).shape[0] * 24, point_nums])
        with tqdm(total=point_nums) as pbar:
            for num in range(34):
                point_num = num
                mean, std = get_mean_std(train_start, train_end)
                rmse_simple, pred_simple = simple_lstm()
                rmse_array[month, num] = rmse_simple
                pred_array[:, num] = pred_simple.reshape(len(pred_simple))
                pbar.update(1)
        pd.DataFrame(pred_array).to_csv(pred_path[month], header=points)
    pd.DataFrame(rmse_array).to_csv(rmse_path[month], header=points)
