import os
import wget
import logging
import scipy.io
import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset
from PyOE.utils import shingle
from PyOE.OEBench import pipeline, outliers


class Dataloader(Dataset):
    """
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
    """

    def __init__(
        self, dataset_name: str, data_dir: str = "./data/", window_size: int = 0
    ):
        self.dataset_name: str = dataset_name
        self.data_dir: str = data_dir
        self.window_size: int = window_size
        self.current_index: int = 0

        # prepare for the dataset
        self.__load_dataset_with_error_checking()

    def __load_dataset_with_error_checking(self) -> None:
        # prepare some varibles here
        try_time = 0
        max_try_time = 1

        # try to load the dataset, re-download if failed
        while try_time <= max_try_time:
            try:
                # prepare dataset
                self.__prepare_dataset()
                # using two different loading method
                if "OD_datasets" in self.dataset_name:
                    self.__load_od_dataset()
                else:
                    self.__load_dataset()
                # loading succeeded
                break
            except Exception as e:
                if try_time >= max_try_time:
                    logging.error(
                        f"Failed to load the dataset after re-trying for {max_try_time} times"
                    )
                    raise e
                else:
                    # error occurred when loading data
                    try_time += 1
                    logging.error(
                        "Failed to load the dataset, now try to re-download it"
                    )
                    # os.system(f"rm -rf {self.data_dir}")

    def __prepare_dataset(self) -> None:
        if not os.path.exists(self.data_dir):
            os.makedirs(self.data_dir)
        if (
            not os.path.exists(f"{self.data_dir}dataset")
            or not os.path.exists(f"{self.data_dir}OD_datasets")
            or not os.path.exists(f"{self.data_dir}dataset_experiment_info")
        ):
            self.__download_dataset()

    def __len__(self):
        return self.num_samples

    def __getitem__(self, idx, return_outlier_label=False):
        return (
            torch.tensor(self.data.iloc[idx]),
            torch.tensor(self.target.iloc[idx]),
            (
                torch.tensor(self.outlier_label[idx])
                if return_outlier_label
                else torch.tensor([])
            ),
        )

    def reach_end(self) -> bool:
        return self.current_index >= self.num_samples

    def get_next_sample(self, return_outlier_label: bool = False):
        if return_outlier_label == False:
            value = (
                torch.tensor(
                    self.data[
                        self.current_index : self.current_index + self.window_size
                    ].values
                ),
                torch.tensor(
                    self.target[
                        self.current_index : self.current_index + self.window_size
                    ].values
                ),
            )
        else:
            value = (
                torch.tensor(
                    self.data[
                        self.current_index : self.current_index + self.window_size
                    ].values
                ),
                torch.tensor(
                    self.target[
                        self.current_index : self.current_index + self.window_size
                    ].values
                ),
                torch.tensor(
                    self.outlier_label[
                        self.current_index : self.current_index + self.window_size
                    ]
                ),
            )
        self.current_index = self.current_index + self.window_size
        return value

    def set_local_outlier_label(self, outlier_label) -> None:
        self.outlier_label[
            self.current_index : self.current_index + self.window_size
        ] = outlier_label

    def set_global_outlier_label(self, outlier_label) -> None:
        self.outlier_label = outlier_label
        self.outlier_ratio = np.sum(self.outlier_label) / self.num_samples

    def set_window_size(self, window_size: int) -> None:
        self.window_size = window_size

    def get_outlier_ratio(self) -> float:
        return self.outlier_ratio

    def get_missing_value_ratio(self) -> float:
        return self.missing_value_ratio

    def get_data_drift_ratio(self) -> float:
        return self.data_drift_ratio

    def get_concept_drift_ratio(self) -> float:
        return self.concept_drift_ratio

    def get_drift_ratio(self) -> float:
        return (self.data_drift_ratio + self.concept_drift_ratio) / 2

    def get_window_size(self) -> int:
        return self.window_size

    def get_num_samples(self) -> int:
        return self.num_samples

    def get_num_columns(self) -> int:
        return self.num_columns

    def get_output_dim(self) -> int:
        return self.output_dim

    def get_task(self) -> str:
        return self.task

    def __download_dataset(self) -> None:
        logging.info(
            "Start to download datasets from http://137.132.83.144/yiqun/dataset.zip"
        )
        try:
            # download from the website
            wget.download("http://137.132.83.144/yiqun/dataset.zip", out=self.data_dir)
            wget.download(
                "http://137.132.83.144/yiqun/dataset_experiment_info.zip",
                out=self.data_dir,
            )
            wget.download(
                "http://137.132.83.144/yiqun/OD_datasets.zip", out=self.data_dir
            )
            logging.info("Downloading finished!")
            # unzip those downloaded files
            logging.info("Start to unzip the downloaded archive...")
            os.system(f"unzip {self.data_dir}dataset.zip -d {self.data_dir}")
            os.system(
                f"unzip {self.data_dir}dataset_experiment_info.zip -d {self.data_dir}"
            )
            os.system(f"unzip {self.data_dir}OD_datasets.zip -d {self.data_dir}")
            logging.info("Unzipping finished!")
        except Exception as e:
            # error occurred while downloading
            logging.error("Obtaining datasets failed!")
            raise e

    def __load_dataset(self) -> None:
        try:
            (
                overall_stats,
                target_data_nonnull,
                data_before_onehot,
                data_onehot_nonnull,
                window_size,
                output_dim,
                data_one_hot,
                task,
            ) = pipeline.run_pipeline(
                dataset_prefix_list=[self.dataset_name],
                return_info=True,
                prefix=self.data_dir,
            )
        except Exception as e:
            raise e

        self.outlier_label = outliers.outlier_detector_marker(data_onehot_nonnull)
        self.data = data_one_hot
        self.target = target_data_nonnull
        self.task = task

        self.num_samples = overall_stats.loc[self.dataset_name]["size"]
        self.num_columns = data_one_hot.shape[1]
        self.output_dim = target_data_nonnull.shape[1]
        self.window_size = window_size if self.window_size == 0 else self.window_size
        self.outlier_ratio = np.sum(self.outlier_label) / self.num_samples
        self.data_drift_ratio, self.concept_drift_ratio, self.missing_value_ratio = (
            overall_stats.loc[self.dataset_name][key]
            for key in [
                "ave_drift_percentage",
                "concept_drift_ratio",
                "overall missing value ratio",
            ]
        )

    def __load_od_dataset(self) -> None:
        if self.dataset_name == "OD_datasets/NSL":
            nfile = self.data_dir + "OD_datasets/nsl.txt"
            lfile = self.data_dir + "OD_datasets/nsllabel.txt"
            numeric = torch.FloatTensor(np.loadtxt(nfile, delimiter=","))
            labels = np.loadtxt(lfile, delimiter=",")
        elif self.dataset_name == "OD_datasets/AT":
            data = pd.read_csv(
                self.data_dir + "OD_datasets/ambient_temperature_system_failure.csv"
            )
            numeric = data["value"].values  # .reshape(-1, 1)
            labels = data["label"].values
            X = shingle(numeric, 10)  # shape (windowsize, len-win+1)
            numeric = torch.FloatTensor(np.transpose(X))
            t1, _ = np.shape(numeric)
            labels = labels[:t1]
        elif self.dataset_name == "OD_datasets/CPU":
            data = pd.read_csv(
                self.data_dir + "OD_datasets/cpu_utilization_asg_misconfiguration.csv"
            )
            numeric = data["value"].values
            labels = data["label"].values
            X = shingle(numeric, 10)
            numeric = torch.FloatTensor(np.transpose(X))
            t1, _ = np.shape(numeric)
            labels = labels[:t1]
        elif self.dataset_name == "OD_datasets/MT":
            data = pd.read_csv(
                self.data_dir + "OD_datasets/machine_temperature_system_failure.csv"
            )
            numeric = data["value"].values
            labels = data["label"].values
            X = shingle(numeric, 10)
            numeric = torch.FloatTensor(np.transpose(X))
            t1, _ = np.shape(numeric)
            labels = labels[:t1]
        elif self.dataset_name == "OD_datasets/NYC":
            data = pd.read_csv(self.data_dir + "OD_datasets/nyc_taxi.csv")
            numeric = data["value"].values
            labels = data["label"].values
            X = shingle(numeric, 10)
            numeric = torch.FloatTensor(np.transpose(X))
            t1, _ = np.shape(numeric)
            labels = labels[:t1]
        elif self.dataset_name in [
            "OD_datasets/INSECTS_Abr",
            "OD_datasets/INSECTS_Incr",
            "OD_datasets/INSECTS_IncrGrd",
            "OD_datasets/INSECTS_IncrRecr",
        ]:
            data = pd.read_csv(
                self.data_dir + self.dataset_name + ".csv",
                dtype=np.float64,
                header=None,
            )
            data_label = data.pop(data.columns[-1])
            numeric = torch.FloatTensor(data.values)
            labels = data_label.values.reshape(-1)
        elif self.dataset_name in [
            "OD_datasets/ionosphere",
            "OD_datasets/mammography",
            "OD_datasets/pima",
            "OD_datasets/satellite",
        ]:
            df = scipy.io.loadmat(self.data_dir + self.dataset_name + ".mat")
            numeric = torch.FloatTensor(df["X"])
            labels = (df["y"]).astype(float).reshape(-1)
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
    def get_oebench_datasets() -> list[str]:
        return [
            "dataset_experiment_info/allstate_claims_severity",
            "dataset_experiment_info/bike_sharing_demand",
            "dataset_experiment_info/rssi",
            "dataset_experiment_info/noaa",
            "dataset_experiment_info/KDDCUP99",
            "dataset_experiment_info/electricity_prices",
            "dataset_experiment_info/tetouan",
            "dataset_experiment_info/beijing_multisite/wanliu",
            "dataset_experiment_info/beijing_multisite/wanshouxingong",
            "dataset_experiment_info/beijing_multisite/gucheng",
            "dataset_experiment_info/beijing_multisite/huairou",
            "dataset_experiment_info/beijing_multisite/nongzhanguan",
            "dataset_experiment_info/beijing_multisite/changping",
            "dataset_experiment_info/beijing_multisite/dingling",
            "dataset_experiment_info/beijing_multisite/aotizhongxin",
            "dataset_experiment_info/beijing_multisite/dongsi",
            "dataset_experiment_info/beijing_multisite/shunyi",
            "dataset_experiment_info/beijing_multisite/guanyuan",
            "dataset_experiment_info/beijing_multisite/tiantan",
            "dataset_experiment_info/weather_indian_cities/bangalore",
            "dataset_experiment_info/weather_indian_cities/lucknow",
            "dataset_experiment_info/weather_indian_cities/mumbai",
            "dataset_experiment_info/weather_indian_cities/Rajasthan",
            "dataset_experiment_info/weather_indian_cities/Bhubhneshwar",
            "dataset_experiment_info/weather_indian_cities/delhi",
            "dataset_experiment_info/weather_indian_cities/chennai",
            "dataset_experiment_info/insects/abrupt_imbalanced",
            "dataset_experiment_info/insects/out-of-control",
            "dataset_experiment_info/insects/incremental_imbalanced",
            "dataset_experiment_info/insects/incremental_reoccurring_balanced",
            "dataset_experiment_info/insects/incremental_balanced",
            "dataset_experiment_info/insects/incremental_abrupt_balanced",
            "dataset_experiment_info/insects/gradual_imbalanced",
            "dataset_experiment_info/insects/abrupt_balanced",
            "dataset_experiment_info/insects/incremental_abrupt_imbalanced",
            "dataset_experiment_info/insects/incremental_reoccurring_imbalanced",
            "dataset_experiment_info/insects/gradual_balanced",
            "dataset_experiment_info/italian_city_airquality",
            "dataset_experiment_info/taxi_ride_duration",
            "dataset_experiment_info/room_occupancy",
            "dataset_experiment_info/bitcoin",
            "dataset_experiment_info/airlines",
            "dataset_experiment_info/traffic_volumn",
            "dataset_experiment_info/news_popularity",
            "dataset_experiment_info/beijingPM2.5",
            "dataset_experiment_info/energy_prediction",
            "dataset_experiment_info/household",
            "dataset_experiment_info/election",
            "dataset_experiment_info/covtype",
            "dataset_experiment_info/safe_driver",
            "dataset_experiment_info/5cities/shenyang",
            "dataset_experiment_info/5cities/guangzhou",
            "dataset_experiment_info/5cities/beijing",
            "dataset_experiment_info/5cities/shanghai",
            "dataset_experiment_info/5cities/chengdu",
        ]

    @staticmethod
    def get_oebench_representative_dataset() -> list[str]:
        return [
            "dataset_experiment_info/electricity_prices",
            "dataset_experiment_info/tetouan",
            "dataset_experiment_info/beijing_multisite/shunyi",
            "dataset_experiment_info/insects/incremental_reoccurring_balanced",
            "dataset_experiment_info/room_occupancy",
        ]

    @staticmethod
    def get_meter_dataset() -> list[str]:
        return [
            "OD_datasets/NSL",
            "OD_datasets/AT",
            "OD_datasets/CPU",
            "OD_datasets/MT",
            "OD_datasets/NYC",
            "OD_datasets/INSECTS_Abr",
            "OD_datasets/INSECTS_Incr",
            "OD_datasets/INSECTS_IncrGrd",
            "OD_datasets/INSECTS_IncrRecr",
            "OD_datasets/ionosphere",
            "OD_datasets/mammography",
            "OD_datasets/pima",
            "OD_datasets/satellite",
        ]


class DataloaderWrapper(Dataset):
    """
    This class is a wrapper for the dataset. It will call the dataset to get the data and target.
    """

    def __init__(self, dataset: Dataloader, return_outlier_label=False):
        self.dataset = dataset
        self.return_outlier_label = return_outlier_label

    def __getitem__(self, idx: int):
        return self.dataset.__getitem__(idx, self.return_outlier_label)

    def __len__(self):
        return self.dataset.__len__()
