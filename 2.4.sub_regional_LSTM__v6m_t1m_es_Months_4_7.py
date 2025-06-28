import warnings
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from numba import njit
from torch.utils.data import DataLoader, Dataset
import os
from tqdm import tqdm

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
    df = caqra_input
    # df.index = pd.to_datetime([str(i) for i in df['date'].values], format="%Y-%m-%d")
    # df = df.iloc[:, 2:]
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
        self.lstm = nn.LSTM(input_size=self.input_size, hidden_size=self.hidden_size, num_layers=2, bias=True, batch_first=True)
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
def get_dataloader():
    # use by DataLoader class for generating mini-batches
    # training data
    train_data = DATA(train_start, train_end, 'train')
    train_loader = DataLoader(train_data, batch_size=batch_size, shuffle=False)
    # validation data
    validate_data = DATA(validate_start, validate_end, 'validate')
    validate_loader = DataLoader(validate_data, batch_size=batch_size, shuffle=False)
    # test data
    test_data = DATA(test_start, test_end, 'test')
    test_loader = DataLoader(test_data, batch_size=batch_size, shuffle=False)
    return train_loader, validate_loader, test_loader


##################
# Early stopping #
##################
class EarlyStopping:
    def __init__(self, patience=20, verbose=False, delta=0):
        self.patience = patience
        self.verbose = verbose
        self.counter = 0
        self.best_score = None
        self.early_stop = False
        self.val_loss_min = np.Inf
        self.delta = delta

    def __call__(self, val_loss, model):
        score = -val_loss
        if self.best_score is None:
            self.best_score = score
            self.save_checkpoint(val_loss, model)
        elif score < self.best_score + self.delta:
            self.counter += 1
            # print(f'EarlyStopping counter: {self.counter} out of {self.patience}')
            if self.counter >= self.patience:
                self.early_stop = True
        else:
            self.best_score = score
            self.save_checkpoint(val_loss, model)
            self.counter = 0

    def save_checkpoint(self, val_loss, model):
        # if self.verbose:
        #     print(f'Validation loss decreased ({self.val_loss_min:.6f} --> {val_loss:.6f}).  Saving model ...')
        # torch.save(model.state_dict(), 'checkpoint.pt')
        self.val_loss_min = val_loss


def regional_lstm():
    # data set up
    train, validate, test = get_dataloader()
    # model set up
    model = Model(input_input_size=input_size, input_hidden_size=hidden_size, input_dropout_rate=dropout_rate).to(DEVICE)
    if num not in [0, 16, 21, 27, 30]:
        print("Loading parameters from: " + str(points_subregions[num]))
        model.load_state_dict(para_last_point)
    else:
        print("Random parameters from: " + str(points_subregions[num]))
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=10, gamma=0.5)
    loss_func = nn.MSELoss()  # mse
    early_stopping = EarlyStopping(patience=20, verbose=False)
    rmse = []
    for i in range(epoch):
        train_model(model, optimizer, train, loss_func)  # training set up
        obs, pred = eval_model(model, validate)  # validating set up
        pred = pred * std[-1] + mean[-1]  # rescale prediction
        rmse.append(torch.sqrt(loss_func(obs, pred)).cpu().numpy())
        scheduler.step()
        early_stopping(torch.sqrt(loss_func(obs, pred)), model)
        if early_stopping.early_stop:
            print("Early stopping in " + str(i+1) + " epoch")
            para = model.state_dict()
            break
        # append parameter in last epoch
        if i == (epoch - 1):
            para = model.state_dict()
    # testing set up
    obs, pred = eval_model(model, test)
    pred = pred * std[-1] + mean[-1]  # rescale prediction
    rmse.append(torch.sqrt(loss_func(obs, pred)).cpu().numpy())
    return rmse, pred.cpu().numpy(), para


if __name__ == "__main__":
    #########################
    # global hyperparameter #
    #########################
    DEVICE = torch.device("cuda:6")
    sequence_length = 24 * 30
    batch_size = 512
    hidden_size = 256
    dropout_rate = 0.5
    learning_rate = 0.001  # lr * 0.5 per 10 epoches
    epoch = 100
    ##################
    # global dataset #
    ##################
    # train_start = pd.to_datetime("2014-01-01 00:00:00", format="%Y-%m-%d %H:%M:%S")
    # train_end = pd.to_datetime("2019-05-31 23:00:00", format="%Y-%m-%d %H:%M:%S")
    # validate_start = pd.to_datetime("2019-06-01 00:00:00", format="%Y-%m-%d %H:%M:%S")
    # validate_end = pd.to_datetime("2019-11-30 23:00:00", format="%Y-%m-%d %H:%M:%S")
    # test_start = pd.to_datetime("2019-12-01 00:00:00", format="%Y-%m-%d %H:%M:%S")
    # test_end = pd.to_datetime("2019-12-31 23:00:00", format="%Y-%m-%d %H:%M:%S")
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
    # caqra_path = ['caqra_4_nan_avg_0.csv', 'caqra_6_nan_avg_0.csv', 'caqra_11_nan_avg_0.csv']
    # rmse_path = [r'H:\2023.8.19-LSTM-BJ-Correction\1.Beijing\rmse\rmse_sub_regional_4_nan_avg_0_v6m_t1m_es_Dec.csv',
    #              r'H:\2023.8.19-LSTM-BJ-Correction\1.Beijing\rmse\rmse_sub_regional_6_nan_avg_0_v6m_t1m_es_Dec.csv',
    #              r'H:\2023.8.19-LSTM-BJ-Correction\1.Beijing\rmse\rmse_sub_regional_11_nan_avg_v6m_t1m_es_Dec.csv']
    # pred_path = [r'H:\2023.8.19-LSTM-BJ-Correction\1.Beijing\pred\pred_sub_regional_4_nan_avg_0_v6m_t1m_es_Dec.csv',
    #              r'H:\2023.8.19-LSTM-BJ-Correction\1.Beijing\pred\pred_sub_regional_6_nan_avg_0_v6m_t1m_es_Dec.csv',
    #              r'H:\2023.8.19-LSTM-BJ-Correction\1.Beijing\pred\pred_sub_regional_11_nan_avg_0_v6m_t1m_es_Dec.csv']
    # obs_path = [r'H:\2023.8.19-LSTM-BJ-Correction\1.Beijing\obs\obs_sub_regional_4_nan_avg_0_v6m_t1m_es_Dec.csv',
    #             r'H:\2023.8.19-LSTM-BJ-Correction\1.Beijing\obs\obs_sub_regional_6_nan_avg_0_v6m_t1m_es_Dec.csv',
    #             r'H:\2023.8.19-LSTM-BJ-Correction\1.Beijing\obs\obs_sub_regional_11_nan_avg_0_v6m_t1m_es_Dec.csv']
    # para_path = [r'H:\2023.8.19-LSTM-BJ-Correction\1.Beijing\para_sub_regional\4.pth',
    #              r'H:\2023.8.19-LSTM-BJ-Correction\1.Beijing\para_sub_regional\6.pth',
    #              r'H:\2023.8.19-LSTM-BJ-Correction\1.Beijing\para_sub_regional\11.pth']
    # rmse_path = ['./rmse/rmse_sub_regional_4_nan_avg_0_v6m_t1m_es_Dec.csv',
    #              './rmse/rmse_sub_regional_6_nan_avg_0_v6m_t1m_es_Dec.csv',
    #              './rmse/rmse_sub_regional_11_nan_avg_v6m_t1m_es_Dec.csv']
    # pred_path = ['./pred/pred_sub_regional_4_nan_avg_0_v6m_t1m_es_Dec.csv',
    #              './pred/pred_sub_regional_6_nan_avg_0_v6m_t1m_es_Dec.csv',
    #              './pred/pred_sub_regional_11_nan_avg_0_v6m_t1m_es_Dec.csv']
    # para_path = [['./para_sub_regional/4_sub_0.pth', './para_sub_regional/4_sub_1.pth', './para_sub_regional/4_sub_2.pth',
    #               './para_sub_regional/4_sub_3.pth', './para_sub_regional/4_sub_4.pth'],
    #              ['./para_sub_regional/6_sub_0.pth', './para_sub_regional/6_sub_1.pth', './para_sub_regional/6_sub_2.pth',
    #               './para_sub_regional/6_sub_3.pth', './para_sub_regional/6_sub_4.pth'],
    #              ['./para_sub_regional/11_sub_0.pth', './para_sub_regional/11_sub_1.pth', './para_sub_regional/11_sub_2.pth',
    #               './para_sub_regional/11_sub_3.pth', './para_sub_regional/11_sub_4.pth']]
    rmse_path = ['./rmse/rmse_sub_regional_11_nan_avg_v6m_t1m_es_Month' + str(i+1) + '.csv' for i in range(12)]
    pred_path = ['./pred/pred_sub_regional_11_nan_avg_0_v6m_t1m_es_Month' + str(i+1) + '.csv' for i in range(12)]
    ######################
    # re-index 34 points #
    ######################
    points = ['东四', '天坛', '官园', '万寿西宫', '奥体中心', '农展馆', '万柳', '北部新区', '丰台花园', '云岗',
              '古城', '房山', '大兴', '亦庄', '通州', '顺义', '昌平', '门头沟', '平谷', '怀柔', '密云', '延庆', '定陵',
              '八达岭', '密云水库', '东高村', '永乐店', '榆垡', '琉璃河', '前门', '永定门内', '西直门北', '南三环',
              '东四环']
    points_subregions = ['东四', '前门', '天坛', '永定门内', '南三环', '万寿西宫', '官园', '西直门北', '奥体中心',
                         '农展馆','东四环', '丰台花园', '万柳', '云岗', '古城', '北部新区',      # 16 center (1 ~ 16)
                         '通州', '亦庄', '大兴', '永乐店', '榆垡',                            # 5 south-east (17 ~ 21)
                         '顺义', '怀柔', '密云', '密云水库', '平谷', '东高村',                 # 6 north-east (22 ~ 27)
                         '门头沟', '房山', '琉璃河',                                        # 3 south-west (28 ~ 30)
                         '昌平', '定陵', '延庆', '八达岭', ]                                # 4 north-west (31 ~ 34)
    points_idx = []
    for i in range(34):
        points_idx.append(np.where(np.array(points) == points_subregions[i])[0][0])
    ###################
    # start go go go! #
    ###################
    for month in range(4, 8):
        print('start ' + 'Month ' + str(month+1))
        train_start = train_start_months[month]
        train_end = train_end_months[month]
        validate_start = validate_start_months[month]
        validate_end = validate_end_months[month]
        test_start = test_start_months[month]
        test_end = test_end_months[month]
        # load caqra
        # caqra_data = pd.read_csv(find_file('H:\\', caqra_path[exp]))
        caqra_data = pd.read_csv('caqra_11_nan_avg_0.csv')
        caqra_data.index = pd.to_datetime([str(i) for i in caqra_data['date'].values], format="%Y-%m-%d %H:%M:%S")
        caqra_data = caqra_data.iloc[:, 2:]
        point_nums = 34
        # if exp == 0:
        #     print("Starting Scenario 1: so2, no2, co, o3")
        #     input_size = 4
        #     vars_idx = []
        #     for idx in points_idx:
        #         vars_idx.extend([i for i in range((input_size+1) * idx, (input_size+1) * (idx + 1))])
        #     caqra_data_re = caqra_data.iloc[:, vars_idx]
        #     para_path = ['./para_sub_regional/4_' + str(i) + '.pth' for i in range(point_nums)]
        # if exp == 1:
        #     print("Starting Scenario 2: so2, no2, co, o3, pm25, pm10")
        #     input_size = 6
        #     vars_idx = []
        #     for idx in points_idx:
        #         vars_idx.extend([i for i in range((input_size+1) * idx, (input_size+1) * (idx + 1))])
        #     caqra_data_re = caqra_data.iloc[:, vars_idx]
        #     para_path = ['./para_sub_regional/6_' + str(i) + '.pth' for i in range(point_nums)]
        # if exp == 2:
        print("Starting Scenario 3: so2, no2, co, o3, pm25, pm10, temp, rh, psfc, u, v")
        input_size = 11
        vars_idx = []
        for idx in points_idx:
            vars_idx.extend([i for i in range((input_size+1) * idx, (input_size+1) * (idx + 1))])
        caqra_data_re = caqra_data.iloc[:, vars_idx]
        para_path = ['./para_sub_regional/month'+str(month)+'_11_' + str(i) + '.pth' for i in range(point_nums)]
        # set rmse and pred
        rmse_array = np.zeros([epoch + 1, point_nums])
        pred_array = np.zeros([pd.date_range(test_start, test_end).shape[0] * 24, point_nums])
        # obs_array = np.zeros([pd.date_range(test_start, test_end).shape[0] * 24, point_nums])
        with tqdm(total=point_nums) as pbar:
            # 16 center (1 ~ 16)
            caqra_input = caqra_data_re.iloc[:, (input_size+1)*0:(input_size+1)*16]
            center = caqra_input.keys().values[input_size::input_size+1]
            print("Starting Center sites: "+str(center))
            for num in range(0, 16):
                print("Calculating Center site: "+str(center[num]))
                point_num = num
                mean, std = get_mean_std()
                rmse_regional, pred_regional, para_last_point = regional_lstm()
                rmse_array[:len(rmse_regional), num] = rmse_regional
                pred_array[:, num] = pred_regional.reshape(len(pred_regional))
                # obs_array[:, num] = obs_regional.reshape(len(obs_regional))
                torch.save(para_last_point, para_path[num])
                pbar.update(1)
            # 5 south-east (17 ~ 21)
            caqra_input = caqra_data_re.iloc[:, (input_size+1)*16:(input_size+1)*21]
            south_east = caqra_input.keys().values[input_size::input_size+1]
            print("Starting South-East sites: " + str(south_east))
            for num in range(16, 21):
                print("Calculating South-East site: " + str(south_east[num-16]))
                point_num = num-16
                mean, std = get_mean_std()
                rmse_regional, pred_regional, para_last_point = regional_lstm()
                rmse_array[:len(rmse_regional), num] = rmse_regional
                pred_array[:, num] = pred_regional.reshape(len(pred_regional))
                # obs_array[:, num] = obs_regional.reshape(len(obs_regional))
                torch.save(para_last_point, para_path[num])
                pbar.update(1)
            # 6 north-east (22 ~ 27)
            caqra_input = caqra_data_re.iloc[:, (input_size+1)*21:(input_size+1)*27]
            north_east = caqra_input.keys().values[input_size::input_size+1]
            print("Starting North-East sites: " + str(north_east))
            for num in range(21, 27):
                print("Calculating North-East site: " + str(north_east[num - 21]))
                point_num = num-21
                mean, std = get_mean_std()
                rmse_regional, pred_regional, para_last_point = regional_lstm()
                rmse_array[:len(rmse_regional), num] = rmse_regional
                pred_array[:, num] = pred_regional.reshape(len(pred_regional))
                # obs_array[:, num] = obs_regional.reshape(len(obs_regional))
                torch.save(para_last_point, para_path[num])
                pbar.update(1)
            # 3 south-west (28 ~ 30)
            caqra_input = caqra_data_re.iloc[:, (input_size+1)*27:(input_size+1)*30]
            south_west = caqra_input.keys().values[input_size::input_size+1]
            print("Starting South-West sites: " + str(south_west))
            for num in range(27, 30):
                print("Calculating South-West site: " + str(south_west[num - 27]))
                point_num = num-27
                mean, std = get_mean_std()
                rmse_regional, pred_regional, para_last_point = regional_lstm()
                rmse_array[:len(rmse_regional), num] = rmse_regional
                pred_array[:, num] = pred_regional.reshape(len(pred_regional))
                # obs_array[:, num] = obs_regional.reshape(len(obs_regional))
                torch.save(para_last_point, para_path[num])
                pbar.update(1)
            # 4 north-west (31 ~ 34)
            caqra_input = caqra_data_re.iloc[:, (input_size+1)*30:(input_size+1)*34]
            north_west = caqra_input.keys().values[input_size::input_size+1]
            print("Starting North-West sites: " + str(north_west))
            for num in range(30, 34):
                print("Calculating North-West site: " + str(north_west[num - 30]))
                point_num = num-30
                mean, std = get_mean_std()
                rmse_regional, pred_regional, para_last_point = regional_lstm()
                rmse_array[:len(rmse_regional), num] = rmse_regional
                pred_array[:, num] = pred_regional.reshape(len(pred_regional))
                # obs_array[:, num] = obs_regional.reshape(len(obs_regional))
                torch.save(para_last_point, para_path[num])
                pbar.update(1)
        # save rmse and pred
        pd.DataFrame(rmse_array).to_csv(rmse_path[month], header=points_subregions)
        pd.DataFrame(pred_array).to_csv(pred_path[month], header=points_subregions)
        # pd.DataFrame(obs_array).to_csv(obs_path[exp], header=points)
