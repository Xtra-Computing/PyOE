import logging
from typing import Literal
from pytorch_tabnet.tab_model import TabNetClassifier, TabNetRegressor
from sklearn.tree import DecisionTreeClassifier, DecisionTreeRegressor
from sklearn.ensemble import GradientBoostingClassifier, GradientBoostingRegressor
from OEBench.model import *
from OEBench.ewc import *
from OEBench.arf import *
from OEBench.armnet import *
from dataloaders import Dataloader


class ModelTemplate:
    """
    A template class for all models. It contains the common attributes and methods for all models.
    ATTENTION: If you need to use your own model, you should inherit this class and assign self.net in
    the __init__ method. The self.net should be a PyTorch model.
    """

    def __init__(
        self,
        dataloader: Dataloader,
        ensemble: int = 1,
        device: Literal["cpu", "cuda"] = "cuda",
    ):
        # fetch metadata of the dataset from dataloader
        self.column_count = dataloader.get_num_columns()
        self.output_dim = dataloader.get_output_dim()
        self.window_size = dataloader.get_window_size()
        self.task = dataloader.get_task()
        self.device = device
        # some default values
        self.net_ensemble = None
        self.ensemble_num = ensemble

    def get_net(self):
        return self.net

    def get_net_ensemble(self):
        return self.net_ensemble

    def get_model_type(self) -> str:
        return self.model_type

    def get_ensemble_number(self) -> int:
        return self.ensemble_num

    def get_device(self) -> Literal["cpu", "cuda"]:
        return self.device


class MlpModel(ModelTemplate):
    """
    A simple MLP model for classification and regression tasks.
    """

    def __init__(
        self,
        dataloader: Dataloader,
        ensemble: int = 1,
        device: Literal["cpu", "cuda"] = "cuda",
    ):
        super().__init__(dataloader, ensemble, device)
        self.model_type = "mlp"
        # initialization for MLP model
        hidden_layers = [32, 16, 8]
        self.net = FcNet(self.column_count, hidden_layers, self.output_dim).to(device)
        self.net_ensemble = [
            FcNet(self.column_count, hidden_layers, self.output_dim)
            for i in range(ensemble)
        ]


class TreeModel(ModelTemplate):
    """
    A simple Tree model for classification and regression tasks.
    """

    def __init__(
        self,
        dataloader: Dataloader,
        ensemble: int = 1,
        device: Literal["cpu"] = "cpu",
    ):
        super().__init__(dataloader, ensemble, device)
        self.model_type = "tree"
        # initialization for Tree model
        if self.task == "classification":
            self.net = DecisionTreeClassifier()
            self.net_ensemble = [DecisionTreeClassifier() for i in range(ensemble)]
        elif self.task == "regression":
            self.net = DecisionTreeRegressor()
            self.net_ensemble = [DecisionTreeRegressor() for i in range(ensemble)]
        else:
            logging.error(f"Task {self.task} not supported.")
            raise ValueError("Task not supported.")


class GbdtModel(ModelTemplate):
    """
    A simple GBDT model for classification and regression tasks.
    """

    def __init__(
        self,
        dataloader: Dataloader,
        ensemble: int = 1,
        device: Literal["cpu"] = "cpu",
    ):
        super().__init__(dataloader, ensemble, device)
        self.model_type = "gbdt"
        # initialization for GBDT model
        if self.task == "classification":
            self.net = GradientBoostingClassifier()
            self.net_ensemble = [GradientBoostingClassifier() for i in range(ensemble)]
        elif self.task == "regression":
            self.net = GradientBoostingRegressor()
            self.net_ensemble = [GradientBoostingRegressor() for i in range(ensemble)]
        else:
            logging.error(f"Task {self.task} not supported.")
            raise ValueError("Task not supported.")


class TabnetModel(ModelTemplate):
    """
    A simple TabNet model for classification and regression tasks.
    """

    def __init__(
        self,
        dataloader: Dataloader,
        ensemble: int = 1,
        device: Literal["cpu"] = "cpu",
    ):
        super().__init__(dataloader, ensemble, device)
        self.model_type = "tabnet"
        # initialization for TabNet model
        if self.task == "classification":
            self.net = TabNetClassifier(seed=0)
        elif self.task == "regression":
            self.net = TabNetRegressor(seed=0)
        else:
            logging.error(f"Task {self.task} not supported.")
            raise ValueError("Task not supported.")


class ArmnetModel(ModelTemplate):
    """
    A simple ARMNet model for classification and regression tasks.
    """

    def __init__(
        self,
        dataloader: Dataloader,
        ensemble: int = 1,
        device: Literal["cpu", "cuda"] = "cuda",
    ):
        super().__init__(dataloader, ensemble, device)
        self.model_type = "armnet"
        # initialization for ARMNet model
        self.net = ARMNetModel(
            self.column_count,
            self.column_count,
            self.column_count,
            2,
            1.7,
            32,
            3,
            16,
            0,
            False,
            3,
            16,
            noutput=self.output_dim,
        ).to(device)
