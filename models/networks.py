import torch
from torch import nn
from river import cluster, stream


class ClusterNet(nn.Module):
    """
    Cluster network template for clustering with CluStream
    """

    def __init__(self, stream) -> None:
        super().__init__()
        self.stream = stream

    def fit(self, X: torch.Tensor) -> None:
        """
        Learn from the data one by one.
        """
        for x, _ in stream.iter_array(X.numpy().tolist()):
            self.stream.learn_one(x)

    def forward(self, X: torch.Tensor) -> torch.Tensor:
        """
        Predict cluster assignments for the input data.
        """
        assigned = []
        for x, _ in stream.iter_array(X.numpy().tolist()):
            assigned.append(self.stream.predict_one(x))
        return torch.tensor(assigned)


class CluStreamNet(ClusterNet):
    """
    CluStream network for clustering
    """

    def __init__(self) -> None:
        super().__init__(cluster.CluStream())


class DbStreamNet(ClusterNet):
    """
    DBStream network for clustering
    """

    def __init__(self) -> None:
        super().__init__(cluster.DBSTREAM())


class DenStreamNet(ClusterNet):
    """
    DenStream network for clustering
    """

    def __init__(self) -> None:
        super().__init__(cluster.DenStream())


class StreamKMeansNet(ClusterNet):
    """
    StreamKMeans network for clustering
    """

    def __init__(self) -> None:
        super().__init__(cluster.STREAMKMeans())
