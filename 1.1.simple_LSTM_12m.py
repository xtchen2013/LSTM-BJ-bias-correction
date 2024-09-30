import warnings
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from numba import njit
from torch.utils.data import DataLoader, Dataset
import os
from tqdm import tqdm
from sklearn import metrics

warnings.filterwarnings('ignore')


# find basin file path
def find_file(disc, file):
    try:
        for root, dirs, files in os.walk(disc, topdown=True):
            if file in files:
                print(f'!!!!! Find {file} !!!!!')
                return root + '\\' + file
            else:
                pass
        print(f'!!!!! Sorry! Not Find {file} !!!!!')
    except PermissionError:
        pass


################
# loading data #
################
def load_data(start, end, period):
    # load caqra data
    df = caqra_data
    df.index = pd.to_datetime([str(i) for i in df['date'].values], format="%Y-%m-%d")
    df = df.iloc[:, 2:]
    # variables + obs
    interval = input_size + 1
    df = df.iloc[:, point_num * interval:point_num * interval + interval]
    # train period
    if period == 'train':
        df = df[start: end]
    # validate period and test period
    else:
        # add exist sequence before start, similar to warmup period!
        df = df[start - pd.DateOffset(days=int(sequence_length / 24)):end]
    # extract variables[0:input_size] and obs[-1]
    x = df.iloc[:, 0:input_size].values
    x = x.reshape(x.shape[0], input_size)
    y = df.iloc[:, -1].values
    y = y.reshape(y.shape[0], 1)
    return df, x, y


####################
# get mean and std #
####################
def get_mean_std():
    # get mean and std in training period for normalize and rescale
    df_train = load_data(train_start, train_end, 'train')[0]
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
    x, y = load_data(start, end, period)[1:]
    # normalize x without missing
    x_normalize = normalize_data(x, 'x')
    # reshape x and y
    x_reshape, y_reshape = reshape_data(x_normalize, y)
    if period == 'train':
        # normalize y in train period
        x_check, y_check = check_data(x_reshape, y_reshape)
        y_normalize = normalize_data(y_check, 'y')  # for calculating loss function
        x_torch = torch.from_numpy(x_check.astype(np.float32))
        y_torch = torch.from_numpy(y_normalize.astype(np.float32))
    else:
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
        self.lstm = nn.LSTM(input_size=self.input_size, hidden_size=self.hidden_size, num_layers=1, bias=True,
                            batch_first=True)
        self.dropout = nn.Dropout(p=self.dropout_rate)
        self.fc = nn.Linear(in_features=self.hidden_size, out_features=1)

    def forward(self, x):
        # hidden state and cell state
        output, (h_n, c_n) = self.lstm(x)
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
        loss = loss_func(y_hat, y)
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
            x = x.to(DEVICE)
            y_hat = model(x)
            obs.append(y)
            preds.append(y_hat)
    return torch.cat(obs), torch.cat(preds)


##################
# rescale output #
##################
def rescale_output(pred):
    # return real value from normalization
    y_means = np.array(mean[-1])
    y_stds = np.array(std[-1])
    pred_rescale = pred * y_stds + y_means
    return pred_rescale


###################################
# Prepare everything for training #
###################################
def get_dataloader():
    # use by DataLoader class for generating mini-batches
    # training data
    train_data = DATA(train_start, train_end, 'train')
    train_loader = DataLoader(train_data, batch_size=batch_size, shuffle=True)
    # validation data
    validate_data = DATA(validate_start, validate_end, 'validate')
    validate_loader = DataLoader(validate_data, batch_size=batch_size, shuffle=False)
    # test data
    test_data = DATA(test_start, test_end, 'test')
    test_loader = DataLoader(test_data, batch_size=batch_size, shuffle=False)
    return train_loader, validate_loader, test_loader


def simple_lstm():
    # data set up
    rmse = []
    train, validate, test = get_dataloader()
    # model set up
    model = Model(input_input_size=input_size, input_hidden_size=hidden_size, input_dropout_rate=dropout_rate).to(DEVICE)
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=10, gamma=0.5)
    loss_func = nn.MSELoss(reduction='mean')  # rmse
    for i in range(epoch):
        train_model(model, optimizer, train, loss_func)  # training set up
        obs, pred = eval_model(model, validate)  # validating set up
        obs = obs.cpu().numpy()
        pred = pred.cpu().numpy()
        pred = rescale_output(pred)
        rmse.append(metrics.mean_squared_error(obs, pred) ** 0.5)
        scheduler.step()
        # append parameter in last epoch
        if i == (epoch - 1):
            para = model.state_dict()
    # testing set up
    obs, pred = eval_model(model, test)
    obs = obs.cpu().numpy()
    pred = pred.cpu().numpy()
    pred = rescale_output(pred)
    rmse.append(metrics.mean_squared_error(obs, pred) ** 0.5)
    return rmse, pred, obs, para


if __name__ == "__main__":
    #########################
    # global hyperparameter #
    #########################
    DEVICE = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    sequence_length = 24 * 30
    batch_size = 512
    hidden_size = 256
    dropout_rate = 0.5
    learning_rate = 0.001  # lr * 0.5 per 10 epoches
    epoch = 30
    ##################
    # global dataset #
    ##################
    train_start = pd.to_datetime("2014-01-01 00:00:00", format="%Y-%m-%d %H:%M:%S")
    train_end = pd.to_datetime("2017-12-31 23:00:00", format="%Y-%m-%d %H:%M:%S")
    validate_start = pd.to_datetime("2018-01-01 00:00:00", format="%Y-%m-%d %H:%M:%S")
    validate_end = pd.to_datetime("2018-12-31 23:00:00", format="%Y-%m-%d %H:%M:%S")
    test_start = pd.to_datetime("2019-01-01 00:00:00", format="%Y-%m-%d %H:%M:%S")
    test_end = pd.to_datetime("2019-12-31 23:00:00", format="%Y-%m-%d %H:%M:%S")
    #############
    # file path #
    #############
    caqra_path = ['caqra_4_nan_avg_0.csv', 'caqra_6_nan_avg_0.csv', 'caqra_11_nan_avg_0.csv']
    rmse_path = [r'D:\beijing\rmse\rmse_simple_4_nan_avg_0_12m.csv',
                 r'D:\beijing\rmse\rmse_simple_6_nan_avg_0_12m.csv',
                 r'D:\beijing\rmse\rmse_simple_11_nan_avg_0_12m.csv']
    pred_path = [r'D:\beijing\pred\pred_simple_4_nan_avg_0_12m.csv',
                 r'D:\beijing\pred\pred_simple_6_nan_avg_0_12m.csv',
                 r'D:\beijing\pred\pred_simple_11_nan_avg_0_12m.csv']
    obs_path = [r'D:\beijing\obs\obs_simple_4_nan_avg_0_12m.csv',
                r'D:\beijing\obs\obs_simple_6_nan_avg_0_12m.csv',
                r'D:\beijing\obs\obs_simple_11_nan_avg_0_12m.csv']
    ###################
    # start go go go! #
    ###################
    for exp in range(3):
        point_nums = 34
        if exp == 0:
            input_size = 4
            para_path = [r'D:\beijing\para\para_simple_4_nan_avg_0_12m\4_' + str(i) + '.pth' for i in range(point_nums)]
        if exp == 1:
            input_size = 6
            para_path = [r'D:\beijing\para\para_simple_6_nan_avg_0_12m\6_' + str(i) + '.pth' for i in range(point_nums)]
        if exp == 2:
            input_size = 11
            para_path = [r'D:\beijing\para\para_simple_11_nan_avg_0_12m\6_' + str(i) + '.pth' for i in range(point_nums)]
        caqra_data = pd.read_csv(find_file('D:\\', caqra_path[exp]))
        # get 34 points
        points = ['东四', '天坛', '官园', '万寿西宫', '奥体中心', '农展馆', '万柳', '北部新区', '丰台花园', '云岗',
                  '古城', '房山', '大兴', '亦庄', '通州', '顺义', '昌平', '门头沟', '平谷', '怀柔', '密云', '延庆', '定陵',
                  '八达岭', '密云水库', '东高村', '永乐店', '榆垡', '琉璃河', '前门', '永定门内', '西直门北', '南三环', '东四环']
        rmse_array = np.zeros([epoch + 1, point_nums])
        pred_array = np.zeros([pd.date_range(test_start, test_end).shape[0] * 24, point_nums])
        obs_array = np.zeros([pd.date_range(test_start, test_end).shape[0] * 24, point_nums])
        with tqdm(total=point_nums) as pbar:
            for num in range(34):
                point_num = num
                mean, std = get_mean_std()
                rmse_simple, pred_simple, obs_simple, para_simple = simple_lstm()
                rmse_array[:, num] = rmse_simple
                pred_array[:, num] = pred_simple.reshape(len(pred_simple))
                obs_array[:, num] = obs_simple.reshape(len(obs_simple))
                torch.save(para_simple, para_path[num])
                pbar.update(1)
        pd.DataFrame(rmse_array).to_csv(rmse_path[exp], header=points)
        pd.DataFrame(pred_array).to_csv(pred_path[exp], header=points)
        pd.DataFrame(obs_array).to_csv(obs_path[exp], header=points)
