import torchvision.models as models
from pytorch_tabnet.tab_model import TabNetClassifier, TabNetRegressor
from sklearn.tree import DecisionTreeClassifier, DecisionTreeRegressor
from sklearn.ensemble import GradientBoostingClassifier, GradientBoostingRegressor
from OEBench.model import *
from OEBench.ewc import *
from OEBench.arf import *
from OEBench.armnet import *
from dataloaders import Dataloader
from typing import Literal


class BasicModel:
    # Input dataset info from dataloader, and chosen model type, to automatically return machine learning model
    def __init__(
        self,
        dataloader: Dataloader,
        model_type: Literal["mlp", "tree", "gbdt", "tabnet", "armnet"] = "mlp",
        ensemble: int = 1,
    ):
        column_count = dataloader.get_num_columns()
        output_dim = dataloader.get_output_dim()
        window_size = dataloader.get_window_size()
        task = dataloader.get_task()

        self.model_type = model_type
        self.net_ensemble = None
        self.ensemble_num = ensemble

        # choose the correct model for training
        if model_type == "mlp":
            hidden_layers = [32, 16, 8]
            self.net = FcNet(column_count, hidden_layers, output_dim)
            self.net_ensemble = [
                FcNet(column_count, hidden_layers, output_dim) for i in range(ensemble)
            ]
        # elif model_type == "arf": # do not support due to bad performance
        #     self.net = AdaptiveRandomForest(nb_features=output_dim, nb_trees=ensemble, pretrain_size=window_size)
        elif model_type == "tree":
            if task == "classification":
                self.net = DecisionTreeClassifier()
                self.net_ensemble = [DecisionTreeClassifier() for i in range(ensemble)]
            else:
                self.net = DecisionTreeRegressor()
                self.net_ensemble = [DecisionTreeRegressor() for i in range(ensemble)]
        elif model_type == "gbdt":
            if task == "classification":
                self.net = GradientBoostingClassifier()
                self.net_ensemble = [
                    GradientBoostingClassifier() for i in range(ensemble)
                ]
            else:
                self.net = GradientBoostingRegressor()
                self.net_ensemble = [
                    GradientBoostingRegressor() for i in range(ensemble)
                ]
        elif model_type == "tabnet":
            if task == "classification":
                self.net = TabNetClassifier(seed=0)
            else:
                self.net = TabNetRegressor(seed=0)
        elif model_type == "armnet":
            self.net = ARMNetModel(
                column_count,
                column_count,
                column_count,
                2,
                1.7,
                32,
                3,
                16,
                0,
                False,
                3,
                16,
                noutput=output_dim,
            )
        else:
            raise ValueError("Model type not supported.")

    def get_net(self):
        return self.net

    def get_net_ensemble(self):
        return self.net_ensemble

    def get_model_type(self) -> str:
        return self.model_type

    def get_ensemble_number(self) -> int:
        return self.ensemble_num
