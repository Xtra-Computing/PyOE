import os
import wget
import logging
import scipy.io
import numpy as np
import pandas as pd
import torch
from abc import abstractmethod
from torch.utils.data import Dataset
from ..utils import shingle
from .pipeline import load_data


class BaseDataloader(Dataset):
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
        self,
        dataset_name: str,
        data_dir: str = "./data/",
        reload: bool = False,
    ):
        """
        Args:
            dataset_name (str): the name of the dataset.
            data_dir (str): the directory to store the dataset.
            reload (bool):
                whether to reload the dataset or load from cache files if exists.
        """
        self.dataset_name: str = dataset_name
        self.data_dir: str = data_dir
        self.reload: bool = reload
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
                if self.dataset_name.startswith("dataset_experiment_info"):
                    self._load_common_dataset()
                elif self.dataset_name.startswith("OD_datasets"):
                    self._load_od_dataset()
                elif self.dataset_name.startswith("financial_datasets"):
                    self._load_financial_dataset()
                else:
                    # dataset not supported
                    raise ValueError("Dataset not supported.")
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
                        f'Failed to load the dataset due to "{e}", now try to re-download it'
                    )
                    # os.system(f"rm -rf {self.data_dir}")

    def __prepare_dataset(self) -> None:
        """
        Prepare the dataset by downloading it if it does not exist.
        """
        if not os.path.exists(self.data_dir):
            os.makedirs(self.data_dir)
        self.__download_dataset()

    def __len__(self):
        """
        Return the number of samples in the dataset. This function is required by PyTorch.

        Returns:
            int: the number of samples in the dataset.
        """
        return self.num_samples

    @abstractmethod
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
        pass

    def get_data(self) -> torch.Tensor | pd.DataFrame:
        """
        Return the data in the dataset.
        pd.DataFrame for time series data, torch.Tensor for others.

        Returns:
            out (torch.Tensor | pd.DataFrame): the data in the dataset.
        """
        return self.data

    def get_target(self) -> torch.Tensor | pd.DataFrame:
        """
        Return the target in the dataset.
        pd.DataFrame for time series data, torch.Tensor for others.

        Returns:
            out (torch.Tensor | pd.DataFrame): the target in the dataset.
        """
        return self.target

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
        if self.dataset_name.startswith("dataset_experiment_info"):
            # download the common dataset if not exists
            if not os.path.exists(f"{self.data_dir}dataset") or not os.path.exists(
                f"{self.data_dir}dataset_experiment_info"
            ):
                self.__download_common_dataset()
        elif self.dataset_name.startswith("OD_datasets"):
            # download the OD dataset if not exists
            if not os.path.exists(f"{self.data_dir}OD_datasets"):
                self.__download_od_dataset()
        elif self.dataset_name.startswith("financial_datasets"):
            # download the financial dataset if not exists
            if not os.path.exists(f"{self.data_dir}financial_datasets"):
                self.__download_financial_dataset()
        else:
            # dataset not supported
            raise ValueError("Dataset not supported.")

    def __download_common_dataset(self) -> None:
        """
        Download the dataset from the website.
        """
        try:
            # download from the website
            logging.info(
                "Start to download datasets from http://137.132.83.144/yiqun/dataset.zip"
            )
            wget.download("http://137.132.83.144/yiqun/dataset.zip", out=self.data_dir)
            wget.download(
                "http://137.132.83.144/yiqun/dataset_experiment_info.zip",
                out=self.data_dir,
            )
            logging.info("Downloading finished!")
            # unzip those downloaded files
            logging.info("Start to unzip the downloaded archive...")
            os.system(f"unzip {self.data_dir}dataset.zip -d {self.data_dir}")
            os.system(
                f"unzip {self.data_dir}dataset_experiment_info.zip -d {self.data_dir}"
            )
            logging.info("Unzipping finished!")
        except Exception as e:
            # error occurred while downloading
            logging.error("Obtaining datasets failed!")
            raise e

    def __download_od_dataset(self) -> None:
        """
        Download the dataset from the website.
        """
        try:
            # download from the website
            logging.info(
                "Start to download datasets from http://137.132.83.144/yiqun/OD_datasets.zip"
            )
            # download from the website
            wget.download(
                "http://137.132.83.144/yiqun/OD_datasets.zip", out=self.data_dir
            )
            logging.info("Downloading finished!")
            # unzip those downloaded files
            logging.info("Start to unzip the downloaded archive...")
            os.system(f"unzip {self.data_dir}OD_datasets.zip -d {self.data_dir}")
            logging.info("Unzipping finished!")
        except Exception as e:
            # error occurred while downloading
            logging.error("Obtaining datasets failed!")
            raise e

    def __download_financial_dataset(self) -> None:
        """
        Download the dataset from the website.
        """
        try:
            # download from the website
            logging.info(
                "Start to download datasets from http://137.132.83.144/yiqun/financial_datasets.zip"
            )
            # download from the website
            wget.download(
                "http://137.132.83.144/yiqun/financial_datasets.zip", out=self.data_dir
            )
            logging.info("Downloading finished!")
            # unzip those downloaded files
            logging.info("Start to unzip the downloaded archive...")
            os.system(f"unzip {self.data_dir}financial_datasets.zip -d {self.data_dir}")
            logging.info("Unzipping finished!")
        except Exception as e:
            # error occurred while downloading
            logging.error("Obtaining datasets failed!")
            raise e

    @abstractmethod
    def _load_common_dataset(self) -> None:
        """
        Load the dataset from local files.
        """
        raise NotImplementedError("This function should be implemented in subclass.")

    @abstractmethod
    def _load_od_dataset(self) -> None:
        """
        Load the OD dataset from local files.
        """
        raise NotImplementedError("This function should be implemented in subclass.")

    @abstractmethod
    def _load_financial_dataset(self) -> None:
        """
        Load the financial dataset from local files.
        """
        raise NotImplementedError("This function should be implemented in subclass.")

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


class Dataloader(BaseDataloader):
    """
    This class is used to load the dataset from local files.
    For non-time-series data only, the data is stored in a torch tensor.
    """

    def __init__(
        self,
        dataset_name: str,
        data_dir: str = "./data/",
        reload: bool = False,
    ):
        """
        Args:
            dataset_name (str): the name of the dataset.
            data_dir (str): the directory to store the dataset.
            reload (bool):
                whether to reload the dataset or load from cache files if exists
        """
        super().__init__(
            dataset_name=dataset_name,
            data_dir=data_dir,
            reload=reload,
        )

    def __getitem__(self, idx: int, return_outlier_label=False):
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

    def _load_common_dataset(self) -> None:
        """
        Load the dataset from local files.
        """
        try:
            (
                target_data_nonnull,
                data_before_onehot,
                data_one_hot,
                data_onehot_nonnull,
                window_size,
                output_dim,
                task,
            ) = load_data(
                dataset_path=self.dataset_name,
                prefix=self.data_dir,
                reload=self.reload,
            )
        except Exception as e:
            raise e

        # import at Runtime to avoid circular import
        from ..models import OutlierDetectorNet

        self.data = torch.tensor(data_one_hot.astype(float).values)
        self.target = torch.tensor(target_data_nonnull.astype(float).values)
        self.task = task
        self.outlier_label = OutlierDetectorNet.outlier_detector_marker(
            data_onehot_nonnull.astype(float)
        )

        self.num_samples = data_one_hot.shape[0]
        self.num_columns = data_one_hot.shape[1]
        self.output_dim = output_dim
        self.outlier_ratio = np.sum(self.outlier_label) / self.num_samples

    def _load_od_dataset(self) -> None:
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
        self.output_dim = 1

    def _load_financial_dataset(self) -> None:
        """
        Load the financial dataset from local files.
        Currently only supported with ``TimeSeriesDataloader``.
        """
        raise ValueError(
            "Currently financial dataset is not supported for non-time-series data."
        )

    """
    Below are some helper functions with regard to outlier detection.
    """

    def get_outlier_ratio(self) -> float:
        """
        Return the outlier ratio for the dataset.

        Returns:
            float: the outlier ratio for the dataset.
        """
        return self.outlier_ratio


class TimeSeriesDataloader(BaseDataloader):
    """
    This class is used to load the time series dataset from local files.
    For time-series data only, the data is stored in a pandas dataframe.
    """

    def __init__(
        self,
        dataset_name: str,
        data_dir: str = "./data/",
        reload: bool = False,
    ):
        """
        Args:
            dataset_name (str): the name of the dataset.
            data_dir (str): the directory to store the dataset.
            reload (bool):
                whether to reload the dataset or load from cache files if exists.
        """
        super().__init__(
            dataset_name=dataset_name,
            data_dir=data_dir,
            reload=reload,
        )

    def __getitem__(self, idx: int, return_outlier_label=False):
        """
        Return the data and target at the given index. This function is required by
        PyTorch.

        Args:
            idx (int): the index of the sample.
            return_outlier_label (bool): whether to return the outlier label
                (not used for time series data).

        Returns:
            value (tuple): a tuple of data, target and an empty tensor.
        """
        return (self.data.iloc[idx], self.target.iloc[idx], torch.tensor([]))

    def _load_common_dataset(self) -> None:
        """
        Load the dataset from local files.
        """
        try:
            (
                target_data_nonnull,
                data_before_onehot,
                data_one_hot,
                data_onehot_nonnull,
                window_size,
                output_dim,
                task,
            ) = load_data(
                dataset_path=self.dataset_name,
                prefix=self.data_dir,
                reload=self.reload,
            )
        except Exception as e:
            raise e

        # set the data and target
        self.data = data_onehot_nonnull
        self.target = target_data_nonnull
        self.task = task

        # time series data needs item_id and timestamp (for target)
        self.data["item_id"] = 0
        self.target["item_id"] = 0

        self.num_samples = data_one_hot.shape[0]
        self.num_columns = data_one_hot.shape[1]
        self.output_dim = output_dim

    def _load_od_dataset(self) -> None:
        """
        Load OD dataset from local files.
        Currently not supported for time series data.
        """
        raise ValueError("Currently OD dataset is not supported for time series data.")

    def _load_financial_dataset(self) -> None:
        """
        Load the financial dataset from local files.
        """
        # change the extension name of self.dataset_name to "csv"
        base_name, _ = os.path.splitext(self.dataset_name)
        self.dataset_name = f"{base_name}.csv"

        # if the dataset not exists, raise an error
        if not os.path.exists(f"{self.data_dir}/{self.dataset_name}"):
            raise ValueError("Dataset not supported.")

        # load the dataset
        target_label = "1. open"
        whole_data = pd.read_csv(f"{self.data_dir}/{self.dataset_name}")
        whole_data["date"] = pd.to_datetime(whole_data["date"])

        # infer and then set frequency
        freq = "D"  # note: datasets in ``financial_datasets`` are daily
        whole_data.set_index("date", inplace=True)
        whole_data = whole_data.asfreq(freq, method="ffill")
        whole_data.reset_index(inplace=True)

        target = whole_data[["date", target_label]]
        target["date"] = pd.to_datetime(target["date"])
        data = whole_data.drop(columns=["date", target_label])

        # rename the columns
        rename_dict = {target_label: "target", "date": "timestamp"}
        target.rename(columns=rename_dict, inplace=True)

        # add some necessary columns
        target["item_id"] = 0
        data["item_id"] = 0

        self.data = data
        self.target = target
        self.task = "forecasting"
        self.num_samples = self.data.shape[0]
        self.num_columns = self.data.shape[1]
        self.output_dim = 1


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
