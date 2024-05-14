import numpy as np
import os
import wget
from OEBench import pipeline, experiments, outliers
from torch.utils.data import Dataset
import torch
import pandas as pd
from utils import shingle
import scipy.io

class Dataloader(Dataset):
    '''
    For datasets in OEBench,
    dataset_name in ['dataset_experiment_info/allstate_claims_severity', 'dataset_experiment_info/bike_sharing_demand', 'dataset_experiment_info/rssi', 
                        'dataset_experiment_info/noaa', 'dataset_experiment_info/KDDCUP99', 'dataset_experiment_info/electricity_prices', 'dataset_experiment_info/tetouan', 
                        'dataset_experiment_info/beijing_multisite/wanliu', 'dataset_experiment_info/beijing_multisite/wanshouxingong', 
                        'dataset_experiment_info/beijing_multisite/gucheng', 'dataset_experiment_info/beijing_multisite/huairou', 'dataset_experiment_info/beijing_multisite/nongzhanguan', 
                        'dataset_experiment_info/beijing_multisite/changping', 'dataset_experiment_info/beijing_multisite/dingling', 'dataset_experiment_info/beijing_multisite/aotizhongxin', 
                        'dataset_experiment_info/beijing_multisite/dongsi', 'dataset_experiment_info/beijing_multisite/shunyi', 'dataset_experiment_info/beijing_multisite/guanyuan', 
                        'dataset_experiment_info/beijing_multisite/tiantan', 'dataset_experiment_info/weather_indian_cities/bangalore', 
                        'dataset_experiment_info/weather_indian_cities/lucknow', 'dataset_experiment_info/weather_indian_cities/mumbai', 'dataset_experiment_info/weather_indian_cities/Rajasthan', 
                        'dataset_experiment_info/weather_indian_cities/Bhubhneshwar', 'dataset_experiment_info/weather_indian_cities/delhi', 'dataset_experiment_info/weather_indian_cities/chennai', 
                        'dataset_experiment_info/insects/abrupt_imbalanced', 'dataset_experiment_info/insects/out-of-control', 
                        'dataset_experiment_info/insects/incremental_imbalanced', 'dataset_experiment_info/insects/incremental_reoccurring_balanced', 
                        'dataset_experiment_info/insects/incremental_balanced', 'dataset_experiment_info/insects/incremental_abrupt_balanced', 'dataset_experiment_info/insects/gradual_imbalanced', 
                        'dataset_experiment_info/insects/abrupt_balanced', 'dataset_experiment_info/insects/incremental_abrupt_imbalanced', 
                        'dataset_experiment_info/insects/incremental_reoccurring_imbalanced', 'dataset_experiment_info/insects/gradual_balanced', 'dataset_experiment_info/italian_city_airquality', 
                        'dataset_experiment_info/taxi_ride_duration', 'dataset_experiment_info/room_occupancy', 'dataset_experiment_info/bitcoin', 
                        'dataset_experiment_info/airlines', 'dataset_experiment_info/traffic_volumn', 'dataset_experiment_info/news_popularity', 
                        'dataset_experiment_info/beijingPM2.5', 'dataset_experiment_info/energy_prediction', 'dataset_experiment_info/household', 'dataset_experiment_info/election', 
                        'dataset_experiment_info/covtype', 'dataset_experiment_info/safe_driver', 'dataset_experiment_info/5cities/shenyang', 
                        'dataset_experiment_info/5cities/guangzhou', 'dataset_experiment_info/5cities/beijing', 'dataset_experiment_info/5cities/shanghai', 'dataset_experiment_info/5cities/chengdu']
    
    for datasets in METER (outlier detection task with provided ground-truth),
    dataset_name in ['OD_datasets/NSL', 'OD_datasets/AT', 'OD_datasets/CPU', 'OD_datasets/MT', 'OD_datasets/NYC', 
                        'OD_datasets/INSECTS_Abr', 'OD_datasets/INSECTS_Incr', 'OD_datasets/INSECTS_IncrGrd', 'OD_datasets/INSECTS_IncrRecr', 
                        'OD_datasets/ionosphere', 'OD_datasets/mammography', 'OD_datasets/pima', 'OD_datasets/satellite']
    '''
    def __init__(self, dataset_name, data_dir="./data/", window_size=0):
        self.dataset_name = dataset_name
        self.data_dir = data_dir
        self.window_size = window_size
        if not os.path.exists(data_dir):
            os.makedirs(data_dir)
        if not os.path.exists(data_dir+"dataset") or not os.path.exists(data_dir+"OD_datasets") or not os.path.exists(data_dir+"dataset_experiment_info"):
            self.download_dataset()
        # try:
        if "OD_datasets" in self.dataset_name:
            self.load_od_dataset()
        else:
            self.load_dataset()
        # except:
        #     print("failed")
        #     os.system("rm -rf {}".format(data_dir))
        #     os.makedirs(data_dir)
        #     self.download_dataset()
        #     self.load_dataset()

        self.current_index = 0


    def set_window_size(self, window_size):
        self.window_size = window_size
        

    def __len__(self):
        return self.num_samples

    def __getitem__(self, idx, return_outlier_label=False):
        if return_outlier_label == False:
            return (self.data[idx], self.target[idx])
        else:
            return (self.data[idx], self.target[idx], self.outlier_label[idx])


    def reach_end(self):
        return (self.current_index>=self.num_samples)

    def get_next_sample(self, return_outlier_label=False):
        if return_outlier_label == False:
            value = (self.data[self.current_index:self.current_index+self.window_size], self.target[self.current_index:self.current_index+self.window_size])
        else:
            value = (self.data[self.current_index:self.current_index+self.window_size], self.target[self.current_index:self.current_index+self.window_size], self.outlier_label[self.current_index:self.current_index+self.window_size])
        self.current_index = self.current_index + self.window_size
        return value

    
    def set_local_outlier_label(self, outlier_label):
        self.outlier_label[self.current_index:self.current_index+self.window_size] = outlier_label

    def set_global_outlier_label(self, outlier_label):
        self.outlier_label = outlier_label
        self.outlier_ratio = np.sum(self.outlier_label) / self.num_samples

    def get_outlier_ratio(self):
        return self.outlier_ratio
    
    def get_missing_value_ratio(self):
        return self.missing_value_ratio
    
    def get_data_drift_ratio(self):
        return self.data_drift_ratio
    
    def get_concept_drift_ratio(self):
        return self.concept_drift_ratio
    
    def get_drift_ratio(self):
        return (self.data_drift_ratio + self.concept_drift_ratio)/2
        
    def get_window_size(self):
        return self.window_size
    
    def get_num_samples(self):
        return self.num_samples
    
    def get_num_columns(self):
        return self.num_columns
    
    def get_output_dim(self):
        return self.output_dim

    def download_dataset(self):
        print("Downloading datasets")
        try:
            wget.download("http://137.132.83.144/yiqun/dataset.zip",out=self.data_dir)
            wget.download("http://137.132.83.144/yiqun/dataset_experiment_info.zip",out=self.data_dir)
            wget.download("http://137.132.83.144/yiqun/OD_datasets.zip",out=self.data_dir)
            print("Downloading successful!")
            os.system("unzip {}dataset.zip -d {}".format(self.data_dir,self.data_dir))
            os.system("unzip {}dataset_experiment_info.zip -d {}".format(self.data_dir,self.data_dir))
            os.system("unzip {}OD_datasets.zip -d {}".format(self.data_dir,self.data_dir))
        except:
            print("Downloading failed!")
        
    def load_dataset(self):
        try:
            overall_stats, target_data_nonnull, data_before_onehot, data_onehot_nonnull, window_size, output_dim, data_one_hot, task = pipeline.run_pipeline(dataset_prefix_list=[self.dataset_name], return_info=True, prefix=self.data_dir)
        except:
            raise ValueError("Dataset not supported.")

        self.outlier_label = outliers.outlier_detector_marker(data_onehot_nonnull)
        self.data = data_one_hot
        self.target = target_data_nonnull
        self.task = task

        self.num_samples = overall_stats.loc[self.dataset_name]["size"]
        self.num_columns = overall_stats.loc[self.dataset_name]["#columns"]
        if self.window_size == 0:
            self.window_size = window_size
        self.output_dim = output_dim
        self.outlier_ratio = np.sum(self.outlier_label) / self.num_samples
        self.data_drift_ratio = overall_stats.loc[self.dataset_name]["ave_drift_percentage"]
        self.concept_drift_ratio = overall_stats.loc[self.dataset_name]["concept_drift_ratio"]
        self.missing_value_ratio = overall_stats.loc[self.dataset_name]["overall missing value ratio"]


    def load_od_dataset(self):
        if self.dataset_name == 'OD_datasets/NSL':
            nfile = self.data_dir+'OD_datasets/nsl.txt'
            lfile = self.data_dir+'OD_datasets/nsllabel.txt'
            numeric = torch.FloatTensor(np.loadtxt(nfile, delimiter = ','))
            labels = np.loadtxt(lfile, delimiter=',')
        elif self.dataset_name == 'OD_datasets/AT':
            data = pd.read_csv(self.data_dir+'OD_datasets/ambient_temperature_system_failure.csv')
            numeric = data["value"].values#.reshape(-1, 1)
            labels = data["label"].values
            X = shingle(numeric, 10) # shape (windowsize, len-win+1)
            numeric = torch.FloatTensor(np.transpose(X))
            t1, _ = np.shape(numeric) 
            labels=labels[:t1]
        elif self.dataset_name == 'OD_datasets/CPU':
            data = pd.read_csv(self.data_dir+'OD_datasets/cpu_utilization_asg_misconfiguration.csv')
            numeric = data["value"].values
            labels = data["label"].values
            X = shingle(numeric, 10)
            numeric = torch.FloatTensor(np.transpose(X))
            t1, _ = np.shape(numeric) 
            labels=labels[:t1]
        elif self.dataset_name == 'OD_datasets/MT':
            data = pd.read_csv(self.data_dir+'OD_datasets/machine_temperature_system_failure.csv')
            numeric = data["value"].values
            labels = data["label"].values
            X = shingle(numeric, 10) 
            numeric = torch.FloatTensor(np.transpose(X))
            t1, _ = np.shape(numeric) 
            labels=labels[:t1]
        elif self.dataset_name == 'OD_datasets/NYC':
            data = pd.read_csv(self.data_dir+'OD_datasets/nyc_taxi.csv')
            numeric = data["value"].values
            labels = data["label"].values
            X = shingle(numeric, 10) 
            numeric = torch.FloatTensor(np.transpose(X))
            t1, _ = np.shape(numeric) 
            labels=labels[:t1]
        elif self.dataset_name in ["OD_datasets/INSECTS_Abr", "OD_datasets/INSECTS_Incr",
                            "OD_datasets/INSECTS_IncrGrd", "OD_datasets/INSECTS_IncrRecr"]:
            data = pd.read_csv(self.data_dir+self.dataset_name+".csv", dtype=np.float64, header=None)
            data_label = data.pop(data.columns[-1])
            numeric = torch.FloatTensor(data.values)
            labels = data_label.values.reshape(-1)   
        elif self.dataset_name in ['OD_datasets/ionosphere', 'OD_datasets/mammography', 'OD_datasets/pima', 'OD_datasets/satellite']:
            df = scipy.io.loadmat(self.data_dir+self.dataset_name+".mat")
            numeric = torch.FloatTensor(df['X'])
            labels = (df['y']).astype(float).reshape(-1)
        else:
            raise ValueError("Dataset not supported.")


        self.data = numeric
        self.target = labels
        self.outlier_label = labels
        self.task = "outlier detection"
        self.num_samples = self.data.shape[0]
        self.num_columns = self.data.shape[1]
        if self.window_size == 0:
            self.window_size = 1
        self.output_dim = 1

    @staticmethod
    def get_oebench_datasets():
        return ['dataset_experiment_info/allstate_claims_severity', 'dataset_experiment_info/bike_sharing_demand', 'dataset_experiment_info/rssi', 
                    'dataset_experiment_info/noaa', 'dataset_experiment_info/KDDCUP99', 'dataset_experiment_info/electricity_prices', 'dataset_experiment_info/tetouan', 
                    'dataset_experiment_info/beijing_multisite/wanliu', 'dataset_experiment_info/beijing_multisite/wanshouxingong', 
                    'dataset_experiment_info/beijing_multisite/gucheng', 'dataset_experiment_info/beijing_multisite/huairou', 'dataset_experiment_info/beijing_multisite/nongzhanguan', 
                    'dataset_experiment_info/beijing_multisite/changping', 'dataset_experiment_info/beijing_multisite/dingling', 'dataset_experiment_info/beijing_multisite/aotizhongxin', 
                    'dataset_experiment_info/beijing_multisite/dongsi', 'dataset_experiment_info/beijing_multisite/shunyi', 'dataset_experiment_info/beijing_multisite/guanyuan', 
                    'dataset_experiment_info/beijing_multisite/tiantan', 'dataset_experiment_info/weather_indian_cities/bangalore', 
                    'dataset_experiment_info/weather_indian_cities/lucknow', 'dataset_experiment_info/weather_indian_cities/mumbai', 'dataset_experiment_info/weather_indian_cities/Rajasthan', 
                    'dataset_experiment_info/weather_indian_cities/Bhubhneshwar', 'dataset_experiment_info/weather_indian_cities/delhi', 'dataset_experiment_info/weather_indian_cities/chennai', 
                    'dataset_experiment_info/insects/abrupt_imbalanced', 'dataset_experiment_info/insects/out-of-control', 
                    'dataset_experiment_info/insects/incremental_imbalanced', 'dataset_experiment_info/insects/incremental_reoccurring_balanced', 
                    'dataset_experiment_info/insects/incremental_balanced', 'dataset_experiment_info/insects/incremental_abrupt_balanced', 'dataset_experiment_info/insects/gradual_imbalanced', 
                    'dataset_experiment_info/insects/abrupt_balanced', 'dataset_experiment_info/insects/incremental_abrupt_imbalanced', 
                    'dataset_experiment_info/insects/incremental_reoccurring_imbalanced', 'dataset_experiment_info/insects/gradual_balanced', 'dataset_experiment_info/italian_city_airquality', 
                    'dataset_experiment_info/taxi_ride_duration', 'dataset_experiment_info/room_occupancy', 'dataset_experiment_info/bitcoin', 
                    'dataset_experiment_info/airlines', 'dataset_experiment_info/traffic_volumn', 'dataset_experiment_info/news_popularity', 
                    'dataset_experiment_info/beijingPM2.5', 'dataset_experiment_info/energy_prediction', 'dataset_experiment_info/household', 'dataset_experiment_info/election', 
                    'dataset_experiment_info/covtype', 'dataset_experiment_info/safe_driver', 'dataset_experiment_info/5cities/shenyang', 
                    'dataset_experiment_info/5cities/guangzhou', 'dataset_experiment_info/5cities/beijing', 'dataset_experiment_info/5cities/shanghai', 'dataset_experiment_info/5cities/chengdu']

    @staticmethod
    def get_oebench_representative_dataset():
        return ['dataset_experiment_info/electricity_prices', 'dataset_experiment_info/tetouan', 'dataset_experiment_info/beijing_multisite/shunyi', 
                    'dataset_experiment_info/insects/incremental_reoccurring_balanced', 'dataset_experiment_info/room_occupancy']

    @staticmethod
    def get_meter_dataset():
        return ['OD_datasets/NSL', 'OD_datasets/AT', 'OD_datasets/CPU', 'OD_datasets/MT', 'OD_datasets/NYC', 
                    'OD_datasets/INSECTS_Abr', 'OD_datasets/INSECTS_Incr', 'OD_datasets/INSECTS_IncrGrd', 'OD_datasets/INSECTS_IncrRecr', 
                    'OD_datasets/ionosphere', 'OD_datasets/mammography', 'OD_datasets/pima', 'OD_datasets/satellite']

    