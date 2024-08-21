import os
import wget
import logging
import scipy.io
import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset
from ..utils import shingle
from .pipeline import load_data


class Dataloader(Dataset):
    """
    For datasets in OEBench,
    dataset_name in ``['dataset_experiment_info/allstate_claims_severity',
                     'dataset_experiment_info/bike_sharing_demand',
                     'dataset_experiment_info/rssi',
                     'dataset_experiment_info/noaa',
                     'dataset_experiment_info/KDDCUP99',
                     'dataset_experiment_info/electricity_prices',
                     'dataset_experiment_info/tetouan',
                     'dataset_experiment_info/beijing_multisite/wanliu',
                     'dataset_experiment_info/beijing_multisite/wanshouxingong',
                     'dataset_experiment_info/beijing_multisite/gucheng',
                     'dataset_experiment_info/beijing_multisite/huairou',
                     'dataset_experiment_info/beijing_multisite/nongzhanguan',
                     'dataset_experiment_info/beijing_multisite/changping',
                     'dataset_experiment_info/beijing_multisite/dingling',
                     'dataset_experiment_info/beijing_multisite/aotizhongxin',
                     'dataset_experiment_info/beijing_multisite/dongsi',
                     'dataset_experiment_info/beijing_multisite/shunyi',
                     'dataset_experiment_info/beijing_multisite/guanyuan',
                     'dataset_experiment_info/beijing_multisite/tiantan',
                     'dataset_experiment_info/weather_indian_cities/bangalore',
                     'dataset_experiment_info/weather_indian_cities/lucknow',
                     'dataset_experiment_info/weather_indian_cities/mumbai',
                     'dataset_experiment_info/weather_indian_cities/Rajasthan',
                     'dataset_experiment_info/weather_indian_cities/Bhubhneshwar',
                     'dataset_experiment_info/weather_indian_cities/delhi',
                     'dataset_experiment_info/weather_indian_cities/chennai',
                     'dataset_experiment_info/insects/abrupt_imbalanced',
                     'dataset_experiment_info/insects/out-of-control',
                     'dataset_experiment_info/insects/incremental_imbalanced',
                     'dataset_experiment_info/insects/incremental_reoccurring_balanced',
                     'dataset_experiment_info/insects/incremental_balanced',
                     'dataset_experiment_info/insects/incremental_abrupt_balanced',
                     'dataset_experiment_info/insects/gradual_imbalanced',
                     'dataset_experiment_info/insects/abrupt_balanced',
                     'dataset_experiment_info/insects/incremental_abrupt_imbalanced',
                     'dataset_experiment_info/insects/incremental_reoccurring_imbalanced',
                     'dataset_experiment_info/insects/gradual_balanced',
                     'dataset_experiment_info/italian_city_airquality',
                     'dataset_experiment_info/taxi_ride_duration',
                     'dataset_experiment_info/room_occupancy',
                     'dataset_experiment_info/bitcoin',
                     'dataset_experiment_info/airlines',
                     'dataset_experiment_info/traffic_volumn',
                     'dataset_experiment_info/news_popularity',
                     'dataset_experiment_info/beijingPM2.5',
                     'dataset_experiment_info/energy_prediction',
                     'dataset_experiment_info/household',
                     'dataset_experiment_info/election',
                     'dataset_experiment_info/covtype',
                     'dataset_experiment_info/safe_driver',
                     'dataset_experiment_info/5cities/shenyang',
                     'dataset_experiment_info/5cities/guangzhou',
                     'dataset_experiment_info/5cities/beijing',
                     'dataset_experiment_info/5cities/shanghai',
                     'dataset_experiment_info/5cities/chengdu']``

    for datasets in METER (outlier detection task with provided ground-truth),
    dataset_name in ``['OD_datasets/NSL',
                     'OD_datasets/AT',
                     'OD_datasets/CPU',
                     'OD_datasets/MT',
                     'OD_datasets/NYC',
                     'OD_datasets/INSECTS_Abr',
                     'OD_datasets/INSECTS_Incr',
                     'OD_datasets/INSECTS_IncrGrd',
                     'OD_datasets/INSECTS_IncrRecr',
                     'OD_datasets/ionosphere',
                     'OD_datasets/mammography',
                     'OD_datasets/pima',
                     'OD_datasets/satellite']``
    """

    def __init__(
        self, dataset_name: str, data_dir: str = "./data/", window_size: int = 0
    ):
        """
        Args:
            dataset_name (str): the name of the dataset.
            data_dir (str): the directory to store the dataset.
            window_size (int): the window size of the dataset.
        """
        self.dataset_name: str = dataset_name
        self.data_dir: str = data_dir
        self.window_size: int = window_size
        self.current_index: int = 0

        # prepare for the dataset
        self.__load_dataset_with_error_checking()

    def __load_dataset_with_error_checking(self) -> None:
        """
        This is a wrapper function to load the dataset with error checking.
        If the dataset is not loaded successfully, it will try to reload the dataset
        for certain times (defaultly set to `1`). Actually we want to re-download the
        dataset if data loading fails but that may cause trouble.
        """
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
        """
        Prepare the dataset by downloading it if it does not exist.
        """
        if not os.path.exists(self.data_dir):
            os.makedirs(self.data_dir)
        if (
            not os.path.exists(f"{self.data_dir}dataset")
            or not os.path.exists(f"{self.data_dir}OD_datasets")
            or not os.path.exists(f"{self.data_dir}dataset_experiment_info")
        ):
            self.__download_dataset()

    def __len__(self):
        """
        Return the number of samples in the dataset. This function is required by PyTorch.

        Returns:
            int: the number of samples in the dataset.
        """
        return self.num_samples

    def __getitem__(self, idx, return_outlier_label=False):
        """
        Return the data and target at the given index. This function is required by
        PyTorch. Note that ``return_outlier_label`` is passed by using a wrapper
        class ``DataloaderWrapper``.

        Args:
            idx (int): the index of the sample.
            return_outlier_label (bool): whether to return the outlier label.

        Returns:
            value (tuple): a tuple of data, target and outlier (if ``return_outlier_label`` is ``True``).
        """
        return (
            torch.tensor(self.data[idx]),
            torch.tensor(self.target[idx]),
            (
                torch.tensor(self.outlier_label[idx])
                if return_outlier_label
                else torch.tensor([])
            ),
        )

    def reach_end(self) -> bool:
        """
        Check if the dataloader reaches the end of the dataset. Now it is not used.

        Returns:
            bool: whether the dataloader reaches the end of the dataset.
        """
        return self.current_index >= self.num_samples

    def get_next_sample(self, return_outlier_label: bool = False):
        """
        Return the next sample in the dataset. Now it is not used.

        Args:
            return_outlier_label (bool): whether to return the outlier label.

        Returns:
            value (tuple): a tuple of data, target and outlier (if ``return_outlier_label`` is ``True``).
        """
        if return_outlier_label == False:
            value = (
                self.data[self.current_index : self.current_index + self.window_size],
                self.target[self.current_index : self.current_index + self.window_size],
            )
        else:
            value = (
                self.data[self.current_index : self.current_index + self.window_size],
                self.target[self.current_index : self.current_index + self.window_size],
                torch.tensor(
                    self.outlier_label[
                        self.current_index : self.current_index + self.window_size
                    ]
                ),
            )
        self.current_index = self.current_index + self.window_size
        return value

    def set_local_outlier_label(self, outlier_label: np.array) -> None:
        """
        Set the local outlier label for the current window.

        Args:
            outlier_label (np.array): the outlier label for the current window.
        """
        self.outlier_label[
            self.current_index : self.current_index + self.window_size
        ] = outlier_label

    def set_global_outlier_label(self, outlier_label: np.array) -> None:
        """
        Set the global outlier label for the dataset.

        Args:
            outlier_label (np.array): the outlier label for the dataset.
        """
        self.outlier_label = outlier_label
        self.outlier_ratio = np.sum(self.outlier_label) / self.num_samples

    def set_window_size(self, window_size: int) -> None:
        """
        Set the window size for the dataset.

        Args:
            window_size (int): the window size for the dataset.
        """
        self.window_size = window_size

    def get_outlier_ratio(self) -> float:
        """
        Return the outlier ratio for the dataset.

        Returns:
            float: the outlier ratio for the dataset.
        """
        return self.outlier_ratio

    def get_window_size(self) -> int:
        """
        Return the window size for the dataset.

        Returns:
            int: the window size for the dataset.
        """
        return self.window_size

    def get_num_samples(self) -> int:
        """
        Return the number of samples in the dataset.

        Returns:
            int: the number of samples in the dataset.
        """
        return self.num_samples

    def get_num_columns(self) -> int:
        """
        Return the number of columns in the dataset.

        Returns:
            int: the number of columns in the dataset.
        """
        return self.num_columns

    def get_output_dim(self) -> int:
        """
        Return the output dimension of the dataset.

        Returns:
            int: the output dimension of the dataset.
        """
        return self.output_dim

    def get_task(self) -> str:
        """
        Return the task of the dataset.

        Returns:
            str: the task of the dataset.
        """
        return self.task

    def __download_dataset(self) -> None:
        """
        Download the dataset from the website.
        """
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
        """
        Load the dataset from local files.
        """
        try:
            (
                target_data_nonnull,
                data_before_onehot,
                data_onehot_nonnull,
                window_size,
                output_dim,
                data_one_hot,
                task,
            ) = load_data(
                dataset_path=self.dataset_name,
                prefix=self.data_dir,
            )
        except Exception as e:
            raise e

        # import at Runtime to avoid circular import
        from ..models import OutlierDetectorNet

        self.outlier_label = OutlierDetectorNet.outlier_detector_marker(
            data_onehot_nonnull.astype(float)
        )
        self.data = torch.tensor(data_one_hot.astype(float).values)
        self.target = torch.tensor(target_data_nonnull.astype(float).values)
        self.task = task

        self.num_samples = data_one_hot.shape[0]
        self.num_columns = data_one_hot.shape[1]
        self.output_dim = output_dim
        self.window_size = window_size if self.window_size == 0 else self.window_size
        self.outlier_ratio = np.sum(self.outlier_label) / self.num_samples

    def __load_od_dataset(self) -> None:
        """
        Load OD dataset from local files.
        """
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
        """
        Args:
            dataset (Dataloader): the dataset to wrap.
            return_outlier_label (bool): whether to return the outlier label.
        """
        self.dataset = dataset
        self.return_outlier_label = return_outlier_label

    def __getitem__(self, idx: int):
        """
        The wrapper function to get the data and target from the dataset.
        This function is required by PyTorch.

        Args:
            idx (int): the index of the sample.

        Returns:
            value (tuple): a tuple of data, target and outlier (if ``return_outlier_label`` is ``True``).
        """
        return self.dataset.__getitem__(idx, self.return_outlier_label)

    def __len__(self) -> int:
        """
        The wrapper function to get the length of the dataset.
        This function is required by PyTorch.

        Returns:
            int: the number of samples in the dataset.
        """
        return self.dataset.__len__()
