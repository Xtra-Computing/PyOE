import torch
import operator
import functools
import numpy as np
import pandas as pd
from torch import nn
from typing import Literal
from river import cluster, stream
from autogluon.timeseries import TimeSeriesDataFrame, TimeSeriesPredictor
from ..OEBench.ADBench.baseline.PyOD import PYOD
from streamad.model import (
    xStreamDetector,
    RShashDetector,
    HSTreeDetector,
    LodaDetector,
    RrcfDetector,
)


class ClusterNet(nn.Module):
    """
    Cluster network template for clustering with CluStream
    """

    def __init__(self, stream) -> None:
        """
        Args:
            stream: the stream object that you want to use for clustering.
        """
        super().__init__()
        self.stream = stream

    def fit(self, X: torch.Tensor) -> None:
        """
        Learn from the data one by one.

        Args:
            X (torch.Tensor): the input data.
        """
        for x, _ in stream.iter_array(X.numpy().tolist()):
            self.stream.learn_one(x)

    def forward(self, X: torch.Tensor) -> torch.Tensor:
        """
        Predict cluster assignments for the input data.

        Args:
            X (torch.Tensor): the input data.

        Returns:
            out (torch.Tensor): the predicted cluster assignments.
        """
        return torch.tensor(
            [
                self.stream.predict_one(x)
                for x, _ in stream.iter_array(X.numpy().tolist())
            ]
        )


class CluStreamNet(ClusterNet):
    """
    CluStream network for clustering
    """

    def __init__(self) -> None:
        """
        Initialize the CluStream network.
        """
        super().__init__(cluster.CluStream())


class DbStreamNet(ClusterNet):
    """
    DBStream network for clustering
    """

    def __init__(self) -> None:
        """
        Initialize the DBStream network.
        """
        super().__init__(cluster.DBSTREAM())


class DenStreamNet(ClusterNet):
    """
    DenStream network for clustering
    """

    def __init__(self) -> None:
        """
        Initialize the DenStream network.
        """
        super().__init__(cluster.DenStream())


class StreamKMeansNet(ClusterNet):
    """
    StreamKMeans network for clustering
    """

    def __init__(self) -> None:
        """
        Initialize the StreamKMeans network.
        """
        super().__init__(cluster.STREAMKMeans())


class OutlierDetectorNet(nn.Module):

    def __init__(self, model) -> None:
        """
        Args:
            model: the model that you want to use for outlier detection.
        """
        super().__init__()
        self.model = model

    @staticmethod
    def outlier_detector_marker(data):
        """
        This function is used to detect outliers in the input data.

        Args:
            data (np.ndarray): the input data.

        Returns:
            out (np.ndarray): the detected outliers.
        """

        # assign PYOD seed and model
        seed = 0
        model_dict = {
            "ECOD": PYOD,
            "IForest": PYOD,
        }

        anomaly_list = []
        for name, clf in model_dict.items():
            # fit the model
            clf = clf(seed=seed, model_name=name).fit(data, [])
            # output predicted anomaly score on testing set
            score = clf.predict_score(data)
            # calculate the threshold
            t = score.mean() + 2 * score.std()
            # add the anomaly index to the list
            anomaly_list.append(np.where(score > t, 1, 0))

        # multiply the anomaly index and return
        return functools.reduce(operator.mul, anomaly_list)

    def forward(self, X: torch.Tensor) -> torch.Tensor:
        """
        Predict cluster assignments for the input data.

        Args:
            X (torch.Tensor): the input data.

        Returns:
            out (torch.Tensor): the predicted cluster assignments.
        """
        return self.outlier_detector_marker(X)

    def get_model_score(self, X: torch.Tensor) -> torch.Tensor:
        """
        Using a stream model to get the anomaly score, and with this function
        you could compare the result with the ground truth to evaluate the model.

        Args:
            X (torch.Tensor): the input data.

        Returns:
            out (torch.Tensor): the anomaly score of the input data.
        """
        scores = [self.model.fit(x).score(x) for x in X.cpu().detach().numpy()]
        return torch.tensor(scores, dtype=torch.float)


class XStreamDetectorNet(OutlierDetectorNet):
    """
    xStreamDetector model for outlier detection
    """

    def __init__(self) -> None:
        """
        Initialize the xStreamDetector network.
        """
        super().__init__(xStreamDetector(depth=10))


class RShashDetectorNet(OutlierDetectorNet):
    """
    RShashDetector model for outlier detection
    """

    def __init__(self) -> None:
        """
        Initialize the RShashDetector network.
        """
        super().__init__(RShashDetector(components_num=10))


class HSTreeDetectorNet(OutlierDetectorNet):
    """
    HSTreeDetector model for outlier detection
    """

    def __init__(self) -> None:
        """
        Initialize the HSTreeDetector network.
        """
        super().__init__(HSTreeDetector())


class LodaDetectorNet(OutlierDetectorNet):
    """
    LodaDetector model for outlier detection
    """

    def __init__(self) -> None:
        """
        Initialize the LodaDetector network.
        """
        super().__init__(LodaDetector())


class RrcfDetectorNet(OutlierDetectorNet):
    """
    RrcfDetector model for outlier detection
    """

    def __init__(self) -> None:
        """
        Initialize the RrcfDetector network.
        """
        super().__init__(RrcfDetector())


class ChronosPredictorNet(nn.Module):
    """
    Chronos model for time series prediction.
    See https://github.com/amazon-science/chronos-forecasting.
    """

    def __init__(
        self,
        prediction_length: int,
        model_path: Literal["tiny", "mini", "small", "base", "large"] = "tiny",
        device: Literal["cpu", "cuda"] = "cpu",
    ) -> None:
        """
        Args:
            prediction_length (int): the length of the prediction.
            model_path (Literal["tiny", "mini", "small", "base", "large"]): the type of the model.
            device (Literal["cpu", "cuda"]): the device to run the model.
        """
        super().__init__()
        self.model = TimeSeriesPredictor(prediction_length=prediction_length)
        self.model_parameters = {
            "Chronos": {
                "model_path": model_path,
                "batch_size": 64,
                "device": device,
            }
        }

    def fit(self, X: pd.DataFrame, y: pd.DataFrame) -> None:
        """
        Learn from the time series data.

        Args:
            X (pd.DataFrame): the input data.
            y (pd.DataFrame): the target data.
        """
        time_series = TimeSeriesDataFrame(y, X)
        self.model = self.model.fit(
            time_series,
            hyperparameters=self.model_parameters,
            skip_model_selection=True,
            verbosity=0,
        )

    def forward(self, X: pd.DataFrame, y: pd.DataFrame) -> TimeSeriesDataFrame:
        """
        Predict the future time series data.

        Args:
            X (pd.DataFrame): the input data.
            y (pd.DataFrame): the target data.

        Returns:
            out (TimeSeriesDataFrame): the predicted time series data.
        """
        time_series = TimeSeriesDataFrame(y, X)
        return self.model.predict(time_series)
