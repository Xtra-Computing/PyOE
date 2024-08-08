import torch
from abc import abstractmethod
from torch.utils.data import DataLoader
from menelaus.concept_drift import DDM
from ..models import ModelTemplate
from ..dataloaders import Dataloader


class MetricTemplate:
    """
    This is a template for metric function. We implemented some common metric
    functions and you can also implement your own metric function using this template.
    """

    def __init__(self, model: ModelTemplate, dataloader: Dataloader, **kwargs):
        self.model = model
        self.dataloader = dataloader

    @abstractmethod
    def measure(self, **kwargs) -> float:
        pass


class EffectivenessMetric(MetricTemplate):

    def __init__(self, model: ModelTemplate, dataloader: Dataloader, **kwargs):
        super().__init__(model, dataloader, **kwargs)

    def measure(self, **kwargs) -> float:
        loss = 0.0
        torch_dataloader = DataLoader(self.dataloader, batch_size=64, shuffle=True)
        for X, y, _ in torch_dataloader:
            X, y = X.to(self.model.device).float(), y.to(self.model.device).float()
            loss += self.model.calculate_loss(X, y)

        return loss / len(torch_dataloader)


class DriftDelayMetric(MetricTemplate):
    """
    This metric measures the drift delay of the model. The drift delay is defined as the
    average number of samples between a detected drift point and the nearest true drift
    point to it.
    """

    def __init__(
        self,
        model: ModelTemplate,
        dataloader: Dataloader,
        n_threshold: int = 5,
        warning_scale: int = 2,
        drift_scale: int = 3,
        **kwargs,
    ):
        super().__init__(model, dataloader, **kwargs)
        self.ddm = DDM(
            n_threshold=n_threshold,
            warning_scale=warning_scale,
            drift_scale=drift_scale,
        )

    def measure(self, ground_truth: list[int] | None = None) -> float | list[int]:
        # load data from dataloader
        torch_dataloader = DataLoader(self.dataloader, batch_size=1, shuffle=True)

        drift_index = []
        drift_length = 0.0
        for idx, (X, y, _) in enumerate(torch_dataloader):
            # calculate the predicted value and update DDM model
            X, y = X.to(self.model.device).float(), y.to(self.model.device).float()
            # TODO: the code below is not compatible with some model interfaces
            predict_y = self.model.net(X)
            true_y = y.cpu().detach().numpy()

            # ensure predict_y is a numpy array
            if isinstance(predict_y, torch.Tensor):
                predict_y = predict_y.cpu().detach().numpy()

            # check if the model is in drift state
            self.ddm.update(true_y, predict_y)
            if self.ddm.drift_state == "drift":
                drift_index.append(idx)
                if ground_truth is not None:
                    drift_length += min(ground_truth, key=lambda x: abs(x - idx))

        return drift_index if ground_truth is None else drift_length / len(drift_index)
