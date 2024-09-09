import pyoe
import torch
from river import cluster
from typing import Literal
from torch.utils.data import DataLoader as TorchDataLoader


class KMeansNet(pyoe.ClusterNet):
    """
    KMeans network for clustering
    """

    def __init__(self) -> None:
        """
        Initialize the ODAC network.
        """
        super().__init__(cluster.KMeans())


class KMeansModel(pyoe.ModelTemplate):
    """
    KMeans model for clustering
    """

    def __init__(
        self,
        dataloader: pyoe.Dataloader,
        ensemble: int = 1,
        device: Literal["cpu"] = "cpu",
    ):
        super().__init__(dataloader, ensemble, device)
        self.model_type = "odac"
        self.net = KMeansNet()

    def process_model(self, **kwargs):
        pass

    def train_cluster(self, X: torch.Tensor):
        self.net.fit(X)

    def predict_cluster(self, X: torch.Tensor) -> torch.Tensor:
        return self.net(X)


# prepare dataloader, model, preprocessor and trainer, and then train the model
dataloader = pyoe.Dataloader(dataset_name="OD_datasets/AT")
model = KMeansModel(dataloader=dataloader)
preprocessor = pyoe.Preprocessor(missing_fill="knn2")
trainer = pyoe.ClusterTrainer(dataloader=dataloader, model=model, preprocessor=preprocessor, epochs=16)
trainer.train()

# predict which cluster these data points belong to
torch_dataloader = TorchDataLoader(dataloader, batch_size=32, shuffle=True)
for X, y, _ in torch_dataloader:
    print(X, model.predict_cluster(X))
