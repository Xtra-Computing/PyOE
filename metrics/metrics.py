from abc import abstractmethod
from torch.utils.data import DataLoader
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
    def measure(self) -> float:
        pass


class EffectivenessMetric(MetricTemplate):

    def __init__(self, model: ModelTemplate, dataloader: Dataloader, **kwargs):
        super().__init__(model, dataloader, **kwargs)

    def measure(self) -> float:
        loss = 0.0
        torch_dataloader = DataLoader(self.dataloader, batch_size=64, shuffle=True)
        for X, y, _ in torch_dataloader:
            X, y = X.to(self.model.device).float(), y.to(self.model.device).float()
            loss += self.model.calculate_loss(X, y)

        return loss / len(torch_dataloader)
