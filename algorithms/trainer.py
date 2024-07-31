import torch
import logging
from abc import abstractmethod
from torch import nn
from .loss import *
from ..models import ModelTemplate
from ..dataloaders import Dataloader


class TrainerTemplate:
    """
    This is a template for training function. We implemented some common training
    functions and you can also implement your own training function using this template.
    """

    def __init__(
        self,
        dataloader: Dataloader,
        model: ModelTemplate,
        lr: float = 0.01,
        epochs: int = 1,
        batch_size: int = 64,
        buffer_size: int = 100,
        **kargws,
    ) -> None:
        # basic assignments
        self.dataloader = dataloader
        self.model = model
        self.lr = lr
        self.epochs = epochs
        self.batch_size = batch_size
        self.buffer_size = buffer_size
        # device, task, model_type and net
        self.device = self.model.get_device()
        self.task = self.dataloader.get_task()
        self.model_type = self.model.get_model_type()
        self.net = self.model.get_net()

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
    """
    This class is a training wrapper for the model. It will call the
    model's preprocessing and training function to train the model.
    """

    def __init__(
        self,
        dataloader: Dataloader,
        model: ModelTemplate,
        lr: float = 0.01,
        epochs: int = 1,
        batch_size: int = 64,
        buffer_size: int = 100,
        **kargws,
    ) -> None:
        super().__init__(
            dataloader,
            model,
            lr,
            epochs,
            batch_size,
            buffer_size,
            **kargws,
        )
        # preprocess the model
        self.model.process_model(lr=self.lr)

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
            torch.tensor(X, dtype=torch.float).to(self.device),
            torch.tensor(y, dtype=torch.float).to(self.device),
            torch.tensor(y_outlier, dtype=torch.float).to(self.device),
        )

        # train the model
        self.model.train_naive(X, y, y_outlier, self.batch_size, self.epochs, need_test)


class IcarlTrainer(TrainerTemplate):
    """
    This class is a training wrapper for the model. It will call the
    model's preprocessing and training function to train the model.
    """

    def __init__(
        self,
        dataloader: Dataloader,
        model: ModelTemplate,
        lr: float = 0.01,
        epochs: int = 1,
        batch_size: int = 64,
        buffer_size: int = 100,
        **kargws,
    ) -> None:
        super().__init__(
            dataloader,
            model,
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
        if self.model_type in ("tree", "tabnet", "gbdt"):
            logging.error(f"Model not supported: {self.model_type}")
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
        if self.model_type not in ("mlp"):
            logging.error(f"Model not supported: {self.model_type}")
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
