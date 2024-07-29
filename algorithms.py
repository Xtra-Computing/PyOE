import copy
from OEBench.ewc import *
from OEBench.arf import *
import torch
from sklearn.impute import IterativeImputer, KNNImputer
from sklearn.tree import DecisionTreeClassifier, DecisionTreeRegressor
from sklearn.ensemble import GradientBoostingClassifier, GradientBoostingRegressor
import numpy
import logging
import argparse
import time
import random
import torch.nn.functional as F
from pytorch_tabnet.tab_model import TabNetClassifier, TabNetRegressor
from OEBench.armnet import *
from datasets import Dataloader
from models import BasicModel
from typing import Literal


def compute_loss(net, window_x, window_y, task, tree=False, y_outlier=None):
    if task == "classification":
        if tree:
            pred_label = torch.LongTensor(net.predict(window_x))
        else:
            out = net(window_x)
            _, pred_label = torch.max(out.data, 1)
        if y_outlier is None:
            acc = (pred_label == window_y).sum().item() / window_y.shape[0]
        else:
            acc = (
                (pred_label == window_y)
                * (1 - y_outlier).sum().item()
                / (window_y.shape[0] - y_outlier.sum().item())
            )
        return 1 - acc
    else:
        if tree:
            out = torch.Tensor(net.predict(window_x))
        else:
            out = net(window_x).reshape(-1).detach()
        if y_outlier is None:
            loss = torch.mean(torch.square(out - window_y))
        else:
            loss = torch.sum(torch.square(out - window_y) * (1 - y_outlier)) / (
                window_y.shape[0] - y_outlier.sum().item()
            )
        return loss.item()


class BasicTrainer:
    def __init__(
        self,
        dataloader: Dataloader,
        basic_model: BasicModel,
        algorithm: Literal["naive", "icarl"] = "naive",
        device: Literal["cpu", "cuda"] = "cpu",
        lr: float = 0.01,
        epochs: int = 1,
        batch_size: int = 64,
        buffer_size: int = 100,
    ):
        self.dataloader = dataloader
        self.basic_model = basic_model
        self.device = device
        self.algorithm = algorithm
        self.epochs = epochs
        self.batch_size = batch_size
        self.buffer_size = buffer_size

        if self.algorithm == "naive":
            if basic_model.get_model_type() not in ("tree", "tabnet", "gbdt"):
                self.optimizer = torch.optim.SGD(
                    filter(
                        lambda p: p.requires_grad, basic_model.get_net().parameters()
                    ),
                    lr=lr,
                )
            if dataloader.task == "classification":
                self.criterion = nn.CrossEntropyLoss().to(device)
            elif dataloader.task == "regression":
                self.criterion = nn.MSELoss().to(device)
            else:
                raise ValueError(
                    "Task not supported. BasicTrainer only supports classification and regression."
                )
        elif self.algorithm == "icarl":
            if basic_model.get_model_type() in ("tree", "tabnet", "gbdt"):
                raise ValueError("ICaRL does not support tree model.")
            self.optimizer = torch.optim.SGD(
                filter(lambda p: p.requires_grad, basic_model.get_net().parameters()),
                lr=lr,
            )
            if dataloader.task == "classification":
                self.criterion = nn.CrossEntropyLoss().to(device)
            elif dataloader.task == "regression":
                self.criterion = nn.MSELoss().to(device)
            else:
                raise ValueError(
                    "Task not supported. BasicTrainer only supports classification and regression."
                )
            self.x_example = None
            self.y_example = None

    def train(
        self, X: torch.Tensor, y: torch.Tensor, y_outlier: torch.Tensor, need_test=False
    ):
        start_time = time.time()
        X, y, y_outlier = (
            torch.Tensor(X).to(self.device),
            torch.Tensor(y).to(self.device),
            torch.Tensor(y_outlier).to(self.device),
        )
        task = self.dataloader.get_task()
        model = self.basic_model.get_model_type()
        net = self.basic_model.get_net().to(self.device)

        if task == "classification":
            y = y.long()
        if need_test:
            if model == "armnet":
                x_tmp = dict({})
                x_tmp["value"] = X
                x_tmp["id"] = id.repeat((X.shape[0], 1))
                accuracy_loss = compute_loss(
                    net,
                    x_tmp,
                    y,
                    task,
                    tree=(model in ("tree", "tabnet", "gbdt")),
                    y_outlier=y_outlier,
                )
            else:
                accuracy_loss = compute_loss(
                    net,
                    X,
                    y,
                    task,
                    tree=(model in ("tree", "tabnet", "gbdt")),
                    y_outlier=y_outlier,
                )

        if self.algorithm == "naive":
            length = y.shape[0]
            for epoch in range(self.epochs):
                if model in ("mlp", "armnet"):
                    for batch_ind in range(0, length, self.batch_size):
                        # get a batch of x and y
                        batch_x = torch.tensor(
                            X[batch_ind : batch_ind + self.batch_size],
                            dtype=torch.float64,
                        ).to(self.device)
                        batch_y = torch.tensor(
                            y[batch_ind : batch_ind + self.batch_size],
                            dtype=torch.float64,
                        ).to(self.device)

                        # if the chosen model is "armnet", we should do something more
                        if model == "armnet":
                            x_tmp = dict({})
                            x_tmp["value"] = batch_x
                            x_tmp["id"] = id.repeat((batch_x.shape[0], 1)).to(
                                self.device
                            )
                            batch_x = x_tmp

                        # using gradient descent to optimize the parameters
                        self.optimizer.zero_grad()
                        # the loss function require 0D or 1D tensors for input
                        out: torch.Tensor = net(batch_x).reshape(-1)
                        ref: torch.Tensor = batch_y.reshape(-1)
                        loss = self.criterion(out, ref)
                        loss.backward()
                        self.optimizer.step()
                elif model == "tabnet":
                    try:
                        window_x = X.numpy()
                        window_y = y.numpy()
                    except:
                        pass
                    if task == "regression":
                        window_y = window_y.reshape(-1, 1)
                    net.fit(
                        window_x,
                        window_y,
                        batch_size=self.batch_size,
                        virtual_batch_size=self.batch_size,
                        max_epochs=self.epochs,
                    )
                    break
                else:
                    net.fit(X, y)
                    break  # tree model does not need epoch

        elif self.algorithm == "icarl":
            length = y.shape[0]
            self.examples = [[] for i in range(self.dataloader.get_output_dim())]
            buffer_class = int(self.buffer_size / self.dataloader.get_output_dim())
            for epoch in range(self.epochs):
                if model in ("mlp"):
                    for batch_ind in range(0, length, self.batch_size):
                        batch_x = X[batch_ind : batch_ind + self.batch_size].to(
                            self.device
                        )
                        batch_y = y[batch_ind : batch_ind + self.batch_size].to(
                            self.device
                        )
                        self.optimizer.zero_grad()
                        out = net(batch_x)
                        if task == "regression":
                            out = out.reshape(-1)
                        loss = self.criterion(out, batch_y)
                        if self.x_example is not None:
                            out_e = net(self.x_example)
                            loss += self.criterion(out_e, self.y_example)

                        loss.backward()
                        self.optimizer.step()
                else:
                    raise ValueError("ICaRL only supports NN model.")

            if task == "regression":
                if self.x_example is None:
                    features = net.feature_extractor(X).detach()
                    avg = torch.mean(features, dim=0).unsqueeze(0)
                    distance = torch.norm(features - avg, dim=1)
                    _, indices = torch.topk(distance, k=self.buffer_size, largest=False)
                    self.x_example = X[indices].to(self.device)
                    self.y_example = y[indices].to(self.device)
            else:
                update = False
                for i in range(self.dataloader.get_output_dim()):
                    if len(self.examples[i]) < buffer_class and i in y:
                        data_class = X[y == i]
                        features = net.feature_extractor(data_class).detach()
                        avg = torch.mean(features, dim=0).unsqueeze(0)
                        distance = torch.norm(features - avg, dim=1)
                        _, indices = torch.topk(
                            distance,
                            k=min(buffer_class - len(self.examples[i]), len(distance)),
                            largest=False,
                        )
                        if len(self.examples[i]) == 0:
                            self.examples[i] = data_class[indices]
                        else:
                            self.examples[i] = torch.cat(
                                (self.examples[i], data_class[indices]), dim=0
                            )
                        update = True
                if update:
                    self.x_example = []
                    for i in range(self.dataloader.get_output_dim()):
                        if len(self.examples[i]) == 0:
                            continue
                        if len(self.x_example) == 0:
                            self.x_example = self.examples[i]
                            self.y_example = torch.ones(self.examples[i].shape[0]) * i
                        else:
                            self.x_example = torch.cat(
                                (self.x_example, self.examples[i]), dim=0
                            )
                            self.y_example = torch.cat(
                                (
                                    self.y_example,
                                    torch.ones(self.examples[i].shape[0]) * i,
                                ),
                                dim=0,
                            )
                    if task == "classification":
                        self.y_example = self.y_example.long()
                    self.x_example, self.y_example = self.x_example.to(
                        self.device
                    ), self.y_example.to(self.device)

        elif self.algorithm == "sea":
            # TODO
            pass
