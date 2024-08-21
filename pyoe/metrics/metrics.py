import torch
from abc import abstractmethod
from sklearn.linear_model import LinearRegression
from torch.utils.data import DataLoader
from menelaus.concept_drift import DDM
from ..models import ModelTemplate
from ..dataloaders import Dataloader


class MetricTemplate:
    """
    This is a template for metric function. We implemented some common metric
    functions and you can also implement your own metric function using this template.
    """

    def __init__(self, dataloader: Dataloader, model: ModelTemplate, **kwargs):
        """
        Args:
            dataloader (Dataloader): the dataloader object that contains the dataset.
            model (ModelTemplate): the model object that you want to evaluate.
            **kwargs: other arguments that you want to pass.
        """
        self.dataloader = dataloader
        self.model = model

    @abstractmethod
    def measure(self, **kwargs):
        """
        This function should be implemented in the subclass. By calling this function,
        you can get the evaluation result of the model.

        Args:
            **kwargs: any arguments that you want to pass.
        """
        pass


class EffectivenessMetric(MetricTemplate):

    def __init__(self, dataloader: Dataloader, model: ModelTemplate, **kwargs):
        """
        Args:
            dataloader (Dataloader): the dataloader object that contains the dataset.
            model (ModelTemplate): the model object that you want to evaluate.
            **kwargs: other arguments that you want to pass.
        """
        super().__init__(dataloader, model, **kwargs)

    def measure(self, **kwargs) -> float:
        """
        This function calculates the effectiveness of the model. We do this by calculating
        the average loss of the model on the whole dataset.

        Args:
            **kwargs: any arguments that you want to pass.

        Returns:
            out (float): the average loss of the model on the whole dataset.
        """
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

    The metric uses the DDM algorithm to detect drift points. The DDM algorithm is a
    statistical method that detects drift points by comparing the predicted value with
    the true value. When the difference between the predicted value and the true value
    exceeds a certain extent on continuous ``n_threshold`` points, the model is
    considered to have drifted.
    """

    def __init__(
        self,
        dataloader: Dataloader,
        model=LinearRegression(),
        n_threshold: int = 30,
        warning_scale: int = 2,
        drift_scale: int = 3,
        **kwargs,
    ):
        """
        Args:
            dataloader (Dataloader): the dataloader object that contains the dataset.
            model (ModelTemplate): the model object that you want to evaluate.
            n_threshold (int, optional): the minimum number of samples required to test
                whether drift has occurred. Defaults to 30.
            warning_scale (int, optional): defines the threshold over which to enter the
                warning state. Defaults to 2.
            drift_scale (int, optional): defines the threshold over which to enter the
                drift state. Defaults to 3.
            **kwargs: other arguments that you want to pass
        """
        super().__init__(dataloader, model, **kwargs)
        self.ddm = DDM(
            n_threshold=n_threshold,
            warning_scale=warning_scale,
            drift_scale=drift_scale,
        )

        self.__init_model()

    def __init_model(self):
        """
        This function initializes the model using the first batch of samples.
        It is called in the constructor.
        """
        X, y = self.dataloader.get_next_sample()
        self.model.fit(X, y)

    def measure(self, ground_truth: list[int] | None = None) -> float | list[int]:
        """
        This function calculates the drift delay of the model. If the ground_truth is
        not provided, it returns the index of the detected drift points. Otherwise, it
        returns the average drift delay of the model.

        Args:
            ground_truth (list[int], optional): the ground truth drift points. If it is
                provided, the function will return the average drift delay of the model.
                Defaults to ``None``.

        Returns:
            out (float | list[int]): the average drift delay of the model or the index
                of the detected drift.
        """
        # load data from dataloader
        torch_dataloader = DataLoader(self.dataloader, batch_size=1, shuffle=False)

        drift_index = []
        drift_length = 0.0
        for idx, (X, y, _) in enumerate(torch_dataloader):
            # calculate the predicted value and update DDM model
            predict_y = self.model.predict(X)
            true_y = y.cpu().detach().numpy().item()

            # ensure predict_y is a numpy array
            if isinstance(predict_y, torch.Tensor):
                predict_y = predict_y.cpu().detach().numpy()
            predict_y = predict_y.item()

            # update and continue
            self.ddm.update(true_y, predict_y)
            self.model.fit(X, y)

            # check if the model is in drift state
            if self.ddm.drift_state == "drift":
                drift_index.append(idx)
                if ground_truth is not None:
                    drift_length += min(ground_truth, key=lambda x: abs(x - idx))

        return drift_index if ground_truth is None else drift_length / len(drift_index)
