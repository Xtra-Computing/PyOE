import torch
import logging
import numpy as np
from torch import nn
from abc import abstractmethod
from typing import Literal, Optional
from .loss import *
from ..dataloaders.base import Dataloader
from ..models import BasicModel


class TrainerTemplate:
    """
    This is a template for training function. We implemented some common training
    functions and you can also implement your own training function using this template.
    """

    def __init__(
        self,
        dataloader: Dataloader,
        basic_model: BasicModel,
        device: Literal["cpu", "cuda"] = "cpu",
        loss: Optional[LossTemplate] = None,
        lr: float = 0.01,
        epochs: int = 1,
        batch_size: int = 64,
        buffer_size: int = 100,
        **kargws,
    ) -> None:
        # basic assignments
        self.dataloader = dataloader
        self.basic_model = basic_model
        self.device = device
        self.lr = lr
        self.epochs = epochs
        self.batch_size = batch_size
        self.buffer_size = buffer_size
        # task, model and net
        self.task = self.dataloader.get_task()
        self.model = self.basic_model.get_model_type()
        self.net = self.basic_model.get_net().to(self.device)
        # choose the loss function
        self.loss = loss if loss is not None else self.choose_loss_function()

    def choose_loss_function(self) -> LossTemplate:
        """
        Choose the loss function based on the task and model
        """
        if self.task == "classification":
            if self.model in ("tree", "tabnet", "gbdt"):
                return classification_loss_tree(self.net)
            else:
                return classification_loss(self.net)
        elif self.task == "regression":
            if self.model in ("tree", "tabnet", "gbdt"):
                return regression_loss_tree(self.net)
            else:
                return regression_loss(self.net)
        else:
            logging.error(f"Task not supported: {self.task}")
            raise ValueError(
                "Task not supported. TrainTemplate only supports classification and regression now."
            )

    @abstractmethod
    def process_model(self) -> None:
        """
        This is the function to process the model. You can implement your own model processing function here.
        """
        pass

    @abstractmethod
    def train(
        self,
        X: torch.Tensor,
        y: torch.Tensor,
        y_outlier: torch.Tensor,
        need_test: bool = False,
        **kargws,
    ) -> None:
        """
        This is the training function. You can implement your own training function here.
        """
        pass


class NaiveTrainer(TrainerTemplate):

    def __init__(
        self,
        dataloader: Dataloader,
        basic_model: BasicModel,
        device: Literal["cpu", "cuda"] = "cpu",
        loss: Optional[LossTemplate] = None,
        lr: float = 0.01,
        epochs: int = 1,
        batch_size: int = 64,
        buffer_size: int = 100,
        **kargws,
    ) -> None:
        super().__init__(
            dataloader,
            basic_model,
            device,
            loss,
            lr,
            epochs,
            batch_size,
            buffer_size,
            **kargws,
        )
        # choose the loss function
        self.loss = loss if loss is not None else self.choose_loss_function()
        # preprocess the model
        self.process_model()

    def process_model(self):
        # choose optimizer for tree models
        if self.model not in ("tree", "tabnet", "gbdt"):
            self.optimizer = torch.optim.SGD(
                filter(lambda p: p.requires_grad, self.net.parameters()),
                lr=self.lr,
            )

        # choose criterion for different tasks
        if self.task == "classification":
            self.criterion = nn.CrossEntropyLoss().to(self.device)
        elif self.task == "regression":
            self.criterion = nn.MSELoss().to(self.device)
        else:
            logging.error(f"Task not supported: {self.task}")
            raise ValueError(
                "Task not supported. BasicTrainer only supports classification and regression now."
            )

    def train(
        self,
        X: torch.Tensor,
        y: torch.Tensor,
        y_outlier: torch.Tensor,
        need_test: bool = False,
        **kargws,
    ) -> None:
        # use the correct device for training
        X, y, y_outlier = (
            torch.Tensor(X).to(self.device),
            torch.Tensor(y).to(self.device),
            torch.Tensor(y_outlier).to(self.device),
        )

        # train preparation
        if self.task == "classification":
            y = y.long()

        # test preparation
        if need_test:
            x_tmp = dict({})
            x_tmp["value"] = X
            x_tmp["id"] = np.repeat(np.arange(X.shape[0]), X.shape[0], axis=0).reshape(
                X.shape[0], -1
            )
            accuracy_loss = self.loss.loss(
                x_tmp if self.model == "armnet" else X,
                y,
                y_outlier,
            )

        # train the model
        length = y.shape[0]
        if self.model == "tabnet":
            x_window = X.numpy()
            y_window = y.numpy()
            if self.task == "regression":
                y_window = y_window.reshape(-1, 1)
            self.net.fit(
                x_window,
                y_window,
                batch_size=self.batch_size,
                virtual_batch_size=self.batch_size,
                max_epochs=self.epochs,
            )
        elif self.model in ("tree", "gbdt"):
            # tree model does not need epoch
            self.net.fit(X, y)
        elif self.model in ("mlp", "armnet"):
            # training for mlp and armnet with epochs times
            for epoch in range(self.epochs):
                logging.info(f"Starting epoch {epoch + 1}/{self.epochs}")
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
                    if self.model == "armnet":
                        x_tmp = dict({})
                        x_tmp["value"] = batch_x
                        # TODO：the line below is buggy
                        x_tmp["id"] = torch.tensor(np.arange(batch_x.shape[0])).to(
                            self.device
                        )
                        batch_x = x_tmp

                    # using gradient descent to optimize the parameters
                    self.optimizer.zero_grad()
                    # the loss function require 0D or 1D tensors for input
                    out: torch.Tensor = self.net(batch_x).reshape(-1)
                    ref: torch.Tensor = batch_y.reshape(-1)
                    loss = self.criterion(out, ref)
                    loss.backward()
                    self.optimizer.step()
        else:
            logging.error(f"Model not supported: {self.model}")
            raise ValueError("Model not supported.")


class IcarlTrainer(TrainerTemplate):

    def __init__(
        self,
        dataloader: Dataloader,
        basic_model: BasicModel,
        device: Literal["cpu", "cuda"] = "cpu",
        loss: Optional[LossTemplate] = None,
        lr: float = 0.01,
        epochs: int = 1,
        batch_size: int = 64,
        buffer_size: int = 100,
        **kargws,
    ) -> None:
        super().__init__(
            dataloader,
            basic_model,
            device,
            loss,
            lr,
            epochs,
            batch_size,
            buffer_size,
            **kargws,
        )
        self.process_model()
        self.x_example = None
        self.y_example = None

    def process_model(self):
        # tree model is not supported for this algorithm
        if self.model in ("tree", "tabnet", "gbdt"):
            logging.error(f"Model not supported: {self.model}")
            raise ValueError("ICaRL does not support tree model.")

        # choose optimizer for the algorithm
        self.optimizer = torch.optim.SGD(
            filter(lambda p: p.requires_grad, self.net.parameters()),
            lr=self.lr,
        )

        # choose criterion for different tasks
        if self.task == "classification":
            self.criterion = nn.CrossEntropyLoss().to(self.device)
        elif self.task == "regression":
            self.criterion = nn.MSELoss().to(self.device)
        else:
            logging.error(f"Task not supported: {self.task}")
            raise ValueError(
                "Task not supported. BasicTrainer only supports classification and regression."
            )

    def train(
        self,
        X: torch.Tensor,
        y: torch.Tensor,
        y_outlier: torch.Tensor,
        need_test: bool = False,
        **kargws,
    ) -> None:
        # ICaRL requires the model to be trained with NN model
        if self.model not in ("mlp"):
            logging.error(f"Model not supported: {self.model}")
            raise ValueError("ICaRL only supports NN model.")

        length = y.shape[0]
        self.examples = [[] for i in range(self.dataloader.get_output_dim())]
        buffer_class = int(self.buffer_size / self.dataloader.get_output_dim())

        # train epochs times
        for epoch in range(self.epochs):
            logging.info(f"Starting epoch {epoch + 1}/{self.epochs}")
            # train the model with batch_size each time
            for batch_ind in range(0, length, self.batch_size):
                batch_x = X[batch_ind : batch_ind + self.batch_size].to(self.device)
                batch_y = y[batch_ind : batch_ind + self.batch_size].to(self.device)
                # zero the gradient
                self.optimizer.zero_grad()
                out = self.net(batch_x)
                if self.task == "regression":
                    out = out.reshape(-1)
                # calculate the loss
                loss = self.criterion(out, batch_y)
                if self.x_example is not None:
                    out_e = self.net(self.x_example)
                    loss += self.criterion(out_e, self.y_example)
                # backward and optimize
                loss.backward()
                self.optimizer.step()

        # post processing
        if self.task == "regression":
            if self.x_example is None:
                features = self.net.feature_extractor(X).detach()
                avg = torch.mean(features, dim=0).unsqueeze(0)
                distance = torch.norm(features - avg, dim=1)
                _, indices = torch.topk(distance, k=self.buffer_size, largest=False)
                self.x_example = X[indices].to(self.device)
                self.y_example = y[indices].to(self.device)
        elif self.task == "classification":
            update = False
            for i in range(self.dataloader.get_output_dim()):
                if len(self.examples[i]) < buffer_class and i in y:
                    data_class = X[y == i]
                    features = self.net.feature_extractor(data_class).detach()
                    avg = torch.mean(features, dim=0).unsqueeze(0)
                    distance = torch.norm(features - avg, dim=1)
                    _, indices = torch.topk(
                        distance,
                        k=min(buffer_class - len(self.examples[i]), len(distance)),
                        largest=False,
                    )
                    self.examples[i] = (
                        data_class[indices]
                        if len(self.examples[i]) == 0
                        else torch.cat((self.examples[i], data_class[indices]), dim=0)
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

                self.y_example = self.y_example.long()
                self.x_example, self.y_example = self.x_example.to(
                    self.device
                ), self.y_example.to(self.device)
