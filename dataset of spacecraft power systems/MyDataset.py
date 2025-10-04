import os
import pickle
import numpy as np
import pandas as pd
import random
import torch
from scienceplots import new_data_path
from torch.utils.data import Dataset, DataLoader
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from utils.process import *
import warnings
from utils.decompose import *
from utils.decompose import *

warnings.filterwarnings('ignore')



class XJTU_SPS_for_Modeling_and_PINN_Dataset(Dataset):
    def __init__(self, args, root_path='/home/data/DiYi/DATA', flag='train', lag=None,
                 features='M', data_path='XJTU/XJTU_SPS_for_Modeling_and_PINN', data_name='XJTU_SPS_for_Modeling_and_PINN_1Hz',
                 missing_rate=0, missvalue=np.nan, target='OT', scale=False, scaler=None, timestamp_scaler=None, work_condition_scaler=None):
        # size [seq_len, label_len pred_len]
        # info
        self.args = args
        self.lag = lag
        # init
        assert flag in ['train', 'test', 'val']
        type_map = {'train': 0, 'val': 1, 'test': 2}
        self.set_type = type_map[flag]
        self.flag = flag

        self.features = features
        self.target = target
        self.scale = scale
        self.scaler = scaler
        self.timestamp_scaler = timestamp_scaler
        self.work_condition_scaler = work_condition_scaler

        self.root_path = root_path
        self.data_path = data_path
        self.data_name = data_name

        self.__read_data__()

    def get_data_dim(self, dataset):
        return self.args.sensor_num

    def __read_data__(self):
        """
        get data from pkl files

        return shape: (([train_size, x_dim], [train_size] or None), ([test_size, x_dim], [test_size]))
        """

        """导入数据"""
        try:
            # f = open(os.path.join(self.root_path, self.data_path, '{}.pkl'.format(self.data_name)), "rb")
            # data = pickle.load(f).values.reshape((-1, x_dim))
            # f.close()
            try:
                data_df = pd.read_csv(os.path.join(self.root_path, self.data_path, '{}.csv'.format(self.data_name)),
                                   sep=',', index_col=False)
            except:
                new_data_path = self.data_path[self.data_path.find('/')+1:]
                data_df = pd.read_csv(os.path.join(self.root_path, new_data_path, '{}.csv'.format(self.data_name)),
                                   sep=',', index_col=False)
            if self.features == 'S':
                data = data_df[[self.target]].values
            else:
                data = data_df.drop(['Time'], axis=1).values

            # 读取Work_Condition数据
            Work_Condition_data_name = self.data_name + '_Work_Condition'
            try:
                work_condition_df = pd.read_csv(os.path.join(self.root_path, self.data_path, '{}.csv'.format(Work_Condition_data_name)),
                                                sep=',', index_col=False)
            except:
                new_data_path = self.data_path[self.data_path.find('/') + 1:]
                work_condition_df = pd.read_csv(os.path.join(self.root_path, new_data_path, '{}.csv'.format(Work_Condition_data_name)),
                                                sep=',', index_col=False)
            work_condition = work_condition_df.drop(['Time'], axis=1).values

            # # 如果是在SPS_Model_PINN时，可以直接使用之前纯仿真结果作为物理信息输入特征。PI_info_batch is pure physical simluation data, which can be got by running [args.model_selection='SPS_Model_Phy', args.if_save_simulate_result=True, args.BasedOn='reconstruct'] first
            if self.args.model_selection == 'SPS_Model_PINN':
                if not self.args.SPS_Model_PINN_if_has_wrong_Phy:
                    if self.args.SPS_Model_PINN_if_has_Phy_of_BCR:
                        file_name = 'physical_simulate_result/' + str(self.args.data_name) + '_reconstruct' + '_physical_simulate_result.csv'
                        PI_data_df = pd.read_csv(os.path.join(self.root_path, self.data_path, file_name),
                                                    sep=',', index_col=False)
                    else:
                        if self.args.replace_BCR_with_MLP_or_0 == 'MLP':
                            suffix = '_physical_simulate_result_withoutBCR_MLPreplace.csv'
                        elif self.args.replace_BCR_with_MLP_or_0 == '0':
                            suffix = '_physical_simulate_result_withoutBCR_1replace.csv'
                        file_name = 'physical_simulate_result/' + str(self.args.data_name) + '_reconstruct' + suffix
                        PI_data_df = pd.read_csv(os.path.join(self.root_path, self.data_path, file_name),
                                                sep=',', index_col=False)
                else:
                    file_name = 'physical_simulate_result/' + str(self.args.data_name) + '_reconstruct' + '_' + str(self.args.wrong_Phy_error) + '_' + str(self.args.random_seed)  + '_physical_simulate_withWrongPhy.csv'
                    PI_data_df = pd.read_csv(os.path.join(self.root_path, self.data_path, file_name),
                                            sep=',', index_col=False)
                PI_data = PI_data_df.values
        except (KeyError, FileNotFoundError):
            data = None

        """检查数据维度是否正确"""
        x_dim = self.get_data_dim(self.data_name)
        if self.features != 'S':
            if data.shape[1] != x_dim:
                raise ValueError('Data shape error, please check it')

        """df_stamp是时间标签信息"""
        df_stamp = data_df[['Time']]
        df_stamp['Time'] = pd.to_datetime(df_stamp.Time)

        df_stamp['day'] = df_stamp.Time.apply(lambda row: row.day, 1)
        df_stamp['hour'] = df_stamp.Time.apply(lambda row: row.hour, 1)
        df_stamp['minute'] = df_stamp.Time.apply(lambda row: row.minute, 1)
        df_stamp['second'] = df_stamp.Time.apply(lambda row: row.second, 1)
        data_stamp = df_stamp.drop(['Time'], axis=1).values
        # data_stamp再补一列timestamp，从0到len(data_stamp)
        data_stamp = np.concatenate([data_stamp, np.arange(len(data_stamp)).reshape(-1, 1)], axis=1)

        """如果是在SPS_Model_PINN时，可以直接使用之前纯仿真结果作为物理信息输入特征。PI_info_batch is pure physical simluation data, which can be got by running [args.model_selection='SPS_Model_Phy', args.if_save_simulate_result=True, args.BasedOn='reconstruct'] first"""
        if self.args.model_selection == 'SPS_Model_PINN':
            # len0 = len(df_stamp) if self.args.BaseOn == 'reconstruct' else len(df_stamp) - self.args.lag
            self.PI_data = PI_data
            # 检查和df_stamp的维度是否一致
            if len(self.PI_data) != len(df_stamp):
                raise ValueError('PI Simulation Data shape do not match df_stamp shape, please check it')

        """***数据划分***"""
        train_len = int(self.args.dataset_split_ratio * len(data))
        border1s = [0, train_len-(train_len//8), train_len]
        border2s = [train_len-(train_len//self.args.dataset_tra_d_val), train_len, len(data)]
        border1 = border1s[self.set_type]
        border2 = border2s[self.set_type]
        # 如果是在SPS_Model_Phy时，参数定型后，将所有数据仿真一遍，保存仿真结果，用于PINN加速训练不再需要batch_size设置为1
        if self.args.if_save_simulate_result and self.args.model_selection == 'SPS_Model_Phy' and self.flag == 'test':
            border1 = 0
            border2 = len(data)+1
        data = data[border1:border2]
        data_stamp = data_stamp[border1:border2]
        work_condition = work_condition[border1:border2]
        PI_data = PI_data[border1:border2] if self.args.model_selection == 'SPS_Model_PINN' else None

        """不使用全部数据，只能使用截取部分数据，根据self.args.only_use_data_ratio"""
        if self.args.only_use_data_ratio < 1 and self.flag == 'train':
            lim_len = int(int(len(data)*self.args.only_use_data_ratio) // (95*60*self.args.exp_frequency) * (95*60*self.args.exp_frequency))
            data = data[-lim_len:]
            data_stamp = data_stamp[-lim_len:]
            work_condition = work_condition[-lim_len:]
            PI_data = PI_data[-lim_len:] if self.args.model_selection == 'SPS_Model_PINN' else None

        """***preprocessing***"""

        """数据标准化归一化"""
        if self.scale:
            norm_data, data, scale_list, mean_list = self.normalize(data, self.flag, self.scaler)
            norm_data_stamp, _, _, _ = self.normalize(data_stamp, self.flag, self.timestamp_scaler)
            norm_work_condition, work_condition, scale_list_work_condition, mean_list_work_condition = self.normalize(work_condition, self.flag,  self.work_condition_scaler)
        else:
            norm_data = data
            scale_list = [1.0] * data.shape[1]
            mean_list = [0.0] * data.shape[1]
            norm_data_stamp = data_stamp
            norm_work_condition = work_condition
            scale_list_work_condition = [1.0] * work_condition.shape[1]
            mean_list_work_condition = [0.0] * work_condition.shape[1]

        """在进行数据缺失或者加噪声之前，保留原始数据用于评估MSE"""
        orig_data = data.copy()
        orig_norm_data = norm_data.copy()

        """进行数据缺失"""
        miss_data, miss_norm_data = make_missing_data(data, self.args.missing_rate, self.args.missvalue, norm_data) \
            if self.args.missing_rate > 0 else (None, None)

        """nan填充:用前一个或者后一个时间步进行nan填充"""
        if np.isnan(data).any():
            data = nan_filling(data)
        if np.isnan(norm_data).any():
            norm_data = nan_filling(norm_data)
        if miss_data is not None and np.isnan(miss_data).any():
            miss_data = nan_filling(miss_data) if self.args.missing_rate > 0 else None
            miss_norm_data = nan_filling(miss_norm_data) if self.args.missing_rate > 0 else None

        """加入噪声"""
        if self.args.add_noise_SNR > 0:
            signal_power = np.mean(data ** 2)
            noise_power = signal_power / (10 ** (self.args.add_noise_SNR / 10))
            noise = np.random.normal(0, np.sqrt(noise_power), data.shape)
            data = data + noise
            miss_data = miss_data + noise if self.args.missing_rate > 0 else None
            signal_power = np.mean(norm_data ** 2)
            noise_power = signal_power / (10 ** (self.args.add_noise_SNR / 10))
            noise = np.random.normal(0, np.sqrt(noise_power), norm_data.shape)
            norm_data = norm_data + noise
            miss_norm_data = miss_norm_data + noise if self.args.missing_rate > 0 else None

        """含噪数据滑动平均预处理"""
        # 先去野点、异常点
        if self.args.remove_outliers:
            data = remove_outliers(data, factor=1)
            norm_data = remove_outliers(norm_data, factor=1)
            miss_data = remove_outliers(miss_data, factor=1) if self.args.missing_rate > 0 else None
            miss_norm_data = remove_outliers(miss_norm_data, factor=1) if self.args.missing_rate > 0 else None
        # 再滑动平均
        if self.args.preMA:
            data = preMA(data, self.args.preMA_win)
            norm_data = preMA(norm_data, self.args.preMA_win)
            miss_data = preMA(miss_data, self.args.preMA_win) if self.args.missing_rate > 0 else None
            miss_norm_data = preMA(miss_norm_data, self.args.preMA_win) if self.args.missing_rate > 0 else None

        """数据定型"""
        self.data = data
        self.norm_data = norm_data
        self.orig_data = orig_data
        self.orig_norm_data = orig_norm_data
        self.miss_data = miss_data
        self.norm_miss_data = miss_norm_data
        self.data_stamp = norm_data_stamp
        self.work_condition = work_condition
        self.norm_work_condition = norm_work_condition
        self.scaler_info = {'scale_list': np.array(scale_list).astype(np.float32),
                            'mean_list': np.array(mean_list).astype(np.float32),
                            'scale_list_work_condition': np.array(scale_list_work_condition).astype(np.float32),
                            'mean_list_work_condition': np.array(mean_list_work_condition).astype(np.float32)}
        self.PI_data = PI_data if self.args.model_selection == 'SPS_Model_PINN' else None


    def __getitem__(self, index):
        ##### init
        init_index = index % self.args.batch_size
        init_r_begin = init_index * self.args.lag_step
        init_r_end = init_index * self.args.lag_step + self.lag
        init_sample = self.data[init_r_begin:init_r_end]
        init_sample_norm = self.norm_data[init_r_begin:init_r_end]
        init_sample = self.miss_data[init_r_begin:init_r_end] if self.args.missing_rate > 0 else init_sample
        init_sample_norm = self.norm_miss_data[init_r_begin:init_r_end] if self.args.missing_rate > 0 else init_sample_norm

        ##### x
        s_begin = index * self.args.lag_step
        s_end = s_begin + self.lag
        x_batch = self.data[s_begin:s_end]
        x_batch_norm = self.norm_data[s_begin:s_end]

        ##### y
        if self.args.BaseOn == 'reconstruct':
            r_begin = s_begin
            r_end = s_end
        elif self.args.BaseOn == 'forecast':
            r_begin = s_end - self.args.label_len
            r_end = s_end + self.args.pred_len
        else:
            raise ValueError('BaseOn must be "reconstruct" or "forecast"')
        y_batch = self.data[r_begin:r_end]
        y_batch_norm = self.norm_data[r_begin:r_end]
        y_batch = self.miss_data[r_begin:r_end] if self.args.missing_rate > 0 else y_batch
        y_batch_norm = self.norm_miss_data[r_begin:r_end] if self.args.missing_rate > 0 else y_batch_norm
        y_primitive_batch = self.orig_data[r_begin:r_end]
        y_primitive_norm_batch = self.orig_norm_data[r_begin:r_end]

        ##### work condition
        WC_batch = self.work_condition[r_end-self.lag:r_end]
        WC_batch_norm = self.norm_work_condition[r_end-self.lag:r_end]
        "work condition use r_begin:r_end, not s_begin:s_end, taht is not information leakage," \
        "because it is the available known condition, and the physical model also uses this, " \
        "not belonging to future data. "
        # """如果是在SPS_Model_PINN时，可以直接使用物理信息的仿真结果"""
        if self.args.model_selection == 'SPS_Model_PINN':
            PI_info_batch = self.PI_data[r_end-self.lag:r_end]
            "if use PINN, the physical simulation result can be used as the physical feature input of the model, " \
            "and the physical simulation result is not belonging to future data, so it can be used."

        ##### t
        datetime_batch = self.data_stamp[r_end-self.lag:r_end]

        """如果是在SPS_Model_PINN时，可以直接使用物理信息的仿真结果"""
        if self.args.model_selection != 'SPS_Model_PINN':
            return (x_batch.astype(np.float32),
                    x_batch_norm.astype(np.float32),
                    WC_batch.astype(np.float32),
                    WC_batch_norm.astype(np.float32),
                    y_batch.astype(np.float32),
                    y_batch_norm.astype(np.float32),
                    y_primitive_batch.astype(np.float32),
                    y_primitive_norm_batch.astype(np.float32),
                    datetime_batch.astype(np.float32),
                    init_sample.astype(np.float32),
                    init_sample_norm.astype(np.float32),
                    self.scaler_info)
        else:
            return (x_batch.astype(np.float32),
                    x_batch_norm.astype(np.float32),
                    WC_batch.astype(np.float32),
                    WC_batch_norm.astype(np.float32),
                    y_batch.astype(np.float32),
                    y_batch_norm.astype(np.float32),
                    y_primitive_batch.astype(np.float32),
                    y_primitive_norm_batch.astype(np.float32),
                    datetime_batch.astype(np.float32),
                    init_sample.astype(np.float32),
                    init_sample_norm.astype(np.float32),
                    self.scaler_info,
                    PI_info_batch.astype(np.float32))

    def __len__(self):
        if self.args.BaseOn == 'reconstruct':
            return (len(self.data) - self.lag) // self.args.lag_step + 1
        elif self.args.BaseOn == 'forecast':
            return (len(self.data) - self.lag - self.args.pred_len) // self.args.lag_step + 1

    def normalize(self, data, flag, scaler):
        """
        returns normalized and standardized data.

        data : array-like
        flag: 'train' or 'val' or 'test'
        scaler: StandardScaler or MinMaxScaler
        """
        if flag == 'train':
            scaler.fit(data)
        elif flag in ['val', 'test']:
            if not hasattr(scaler, 'scale_'):
                scaler.fit(data)
        else:
            pass
        norm_data = scaler.transform(data)
        scale_list = scaler.scale_.tolist()
        mean_list = scaler.mean_.tolist() if hasattr(scaler, 'mean_') else scaler.min_.tolist()

        return norm_data, data, scale_list, mean_list

    def my_inverse_transform(self, data, scaler_str=None):
        """

        data: tensor or numpy, shape: (batch_size, node_num, len)
        scaler_str: str, 调用哪个归一化器    'data' or 'timestamp' or 'work_condition'
        """
        if type(data) == torch.Tensor:
            output = data.numpy()
        else:
            output = data
        if data.shape[0] < data.shape[1]:
            output = output.T
        if scaler_str == 'data' or scaler_str == None:
            output = self.scaler.inverse_transform(output)
        elif scaler_str == 'timestamp':
            output = self.timestamp_scaler.inverse_transform(output)
        elif scaler_str == 'work_condition':
            output = self.work_condition_scaler.inverse_transform(output)
        else:
            raise ValueError('scaler_str must be "data" or "timestamp" or "work_condition"')
        if data.shape[0] < data.shape[1]:
            output = output.T
        if type(data) == torch.Tensor:
            output = torch.from_numpy(output).float()
        return output




