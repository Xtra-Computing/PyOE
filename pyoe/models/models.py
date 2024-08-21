import logging
from abc import abstractmethod
from typing import Literal
from torch.utils.data import DataLoader as TorchDataLoader
from pytorch_tabnet.tab_model import TabNetClassifier, TabNetRegressor
from sklearn.tree import DecisionTreeClassifier, DecisionTreeRegressor
from sklearn.ensemble import GradientBoostingClassifier, GradientBoostingRegressor
from ..OEBench.model import *
from ..OEBench.ewc import *
from ..OEBench.arf import *
from ..OEBench.armnet import *
from .networks import *
from ..algorithms.loss import *
from ..dataloaders import Dataloader


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
        """
        Args:
            dataloader (Dataloader): the dataloader object that contains the dataset.
            ensemble (int): the number of models in the ensemble.
            device (Literal["cpu", "cuda"]): the device that you want to use for training.
        """
        # fetch metadata of the dataset from dataloader
        self.column_count = dataloader.get_num_columns()
        self.output_dim = dataloader.get_output_dim()
        self.window_size = dataloader.get_window_size()
        self.task = dataloader.get_task()
        self.device = device
        # some default values
        self.net_ensemble = None
        self.ensemble_num = ensemble

    @abstractmethod
    def process_model(self, **kwargs):
        """
        This function is used to process the model before training.
        You should assign some values for later use in training
        such as optimizer, criterion, etc. Attention: this function
        is called in the trainer class.

        Args:
            **kwargs: any arguments that you want to pass.
        """
        pass

    def get_net(self):
        """
        This function is used to get the model object.

        Returns:
            out (Any): the model object.
        """
        return self.net

    def get_net_ensemble(self):
        """
        This function is used to get the ensemble model object.

        Returns:
            out (Any): the ensemble model object.
        """
        return self.net_ensemble

    def get_model_type(self) -> str:
        """
        This function is used to get the model type.

        Returns:
            out (str): the model type.
        """
        return self.model_type

    def get_ensemble_number(self) -> int:
        """
        This function is used to get the ensemble number.

        Returns:
            out (int): the ensemble number.
        """
        return self.ensemble_num

    def get_device(self) -> Literal["cpu", "cuda"]:
        """
        This function is used to get the device that you use for training.

        Returns:
            out (Literal["cpu", "cuda"]): the device that you use for training.
        """
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
        """
        Args:
            dataloader (Dataloader): the dataloader object that contains the dataset.
            ensemble (int): the number of models in the ensemble.
            device (Literal["cpu", "cuda"]): the device that you want to use for training.
        """
        super().__init__(dataloader, ensemble, device)
        self.model_type = "mlp"
        # initialization for MLP model
        hidden_layers = [32, 16, 8]
        self.net = FcNet(self.column_count, hidden_layers, self.output_dim).to(device)
        self.net_ensemble = [
            FcNet(self.column_count, hidden_layers, self.output_dim)
            for i in range(ensemble)
        ]

    def process_model(self, lr: float, **kwargs):
        """
        This function is used to process the model before training.

        Args:
            lr (float): the learning rate of the optimizer.
            **kwargs: other arguments that you want to pass.
        """
        # choose optimizer for tree models
        self.optimizer = torch.optim.SGD(
            filter(lambda p: p.requires_grad, self.net.parameters()),
            lr=lr,
        )

        # choose criterion for different tasks
        if self.task == "classification":
            self.criterion = nn.CrossEntropyLoss().to(self.device)
            self.loss = classification_loss(self.net)
        elif self.task == "regression":
            self.criterion = nn.MSELoss().to(self.device)
            self.loss = regression_loss(self.net)
        else:
            logging.error(f"Task not supported: {self.task}")
            raise ValueError("Task not supported.")

    def train_naive(
        self,
        X: torch.Tensor,
        y: torch.Tensor,
        y_outlier: torch.Tensor,
        batch_size: int,
        epochs: int,
        need_test: bool = False,
    ):
        """
        In this function, we will train the model using the input data.

        Args:
            X (torch.Tensor): the input data.
            y (torch.Tensor): the target value.
            y_outlier (torch.Tensor): the outlier data.
            batch_size (int): the batch size for training.
            epochs (int): the number of epochs for training.
            need_test (bool): if this parameter is True, the accurate loss
                will be calculated during training.
        """
        X, y, y_outlier, batch_size, epochs = self.__train_naive_header(
            X, y, y_outlier, batch_size, epochs, need_test
        )
        self.__train_naive_body(X, y, y_outlier, batch_size, epochs, need_test)
        self.__train_naive_footer(X, y, y_outlier, batch_size, epochs, need_test)

    def __train_naive_header(
        self,
        X: torch.Tensor,
        y: torch.Tensor,
        y_outlier: torch.Tensor,
        batch_size: int,
        epochs: int,
        need_test: bool = False,
    ):
        """
        This function is used to preprocess the input data.

        Args:
            X (torch.Tensor): the input data.
            y (torch.Tensor): the target value.
            y_outlier (torch.Tensor): the outlier data.
            batch_size (int): the batch size for training.
            epochs (int): the number of epochs for training.
            need_test (bool): if this parameter is True, the accurate loss
                will be calculated during training.

        Returns:
            out (tuple): the preprocessed data.
        """
        # if the task is classification, we should convert the data type of y to long
        if self.task == "classification":
            y = y.long()

        return (X, y, y_outlier, batch_size, epochs)

    def __train_naive_body(
        self,
        X: torch.Tensor,
        y: torch.Tensor,
        y_outlier: torch.Tensor,
        batch_size: int,
        epochs: int,
        need_test: bool = False,
    ):
        """
        This function is the main part of training.

        Args:
            X (torch.Tensor): the input data.
            y (torch.Tensor): the target value.
            y_outlier (torch.Tensor): the outlier data.
            batch_size (int): the batch size for training.
            epochs (int): the number of epochs for training.
            need_test (bool): if this parameter is True, the accurate loss
                will be calculated during training.
        """
        # training for mlp and armnet with epochs times
        for epoch in range(epochs):
            # logging some information for each epoch
            logging.info(f"Starting epoch {epoch + 1}/{epochs}")
            # use torch.utils.data.DataLoader to load the data
            x_loader = TorchDataLoader(X, batch_size=batch_size)
            y_loader = TorchDataLoader(y, batch_size=batch_size)

            for x_batch, y_batch in zip(x_loader, y_loader):
                # get a batch of x and y
                x_batch = x_batch.to(self.device).float()
                y_batch = y_batch.to(self.device).float()

                # using gradient descent to optimize the parameters
                self.optimizer.zero_grad()
                # the loss function require 0D or 1D tensors for input
                out: torch.Tensor = self.net(x_batch).reshape(-1)
                ref: torch.Tensor = y_batch.reshape(-1)
                loss = self.criterion(out, ref)
                loss.backward()

                # using gradient clipping to avoid gradient explosion
                torch.nn.utils.clip_grad_norm_(self.net.parameters(), max_norm=1.0)
                self.optimizer.step()

    def __train_naive_footer(
        self,
        X: torch.Tensor,
        y: torch.Tensor,
        y_outlier: torch.Tensor,
        batch_size: int,
        epochs: int,
        need_test: bool = False,
    ):
        """
        This function is used to postprocess the model after training.

        Args:
            X (torch.Tensor): the input data.
            y (torch.Tensor): the target value.
            y_outlier (torch.Tensor): the outlier data.
            batch_size (int): the batch size for training.
            epochs (int): the number of epochs for training.
            need_test (bool): if this parameter is True, the accurate loss
                will be calculated during training.
        """
        # test preparation
        if need_test:
            accuracy_loss = self.loss.loss(X, y, y_outlier)
            logging.info(f"Current accuracy loss: {accuracy_loss}")

    def train_icarl(
        self,
        X: torch.Tensor,
        y: torch.Tensor,
        y_outlier: torch.Tensor,
        batch_size: int,
        epochs: int,
        buffer_size: int,
        need_test: bool = False,
    ):
        """
        This function iCaRL implements the training process for MLP model.

        Args:
            X (torch.Tensor): the input data.
            y (torch.Tensor): the target value.
            y_outlier (torch.Tensor): the outlier data.
            batch_size (int): the batch size for training.
            epochs (int): the number of epochs for training.
            buffer_size (int): the size of the exemplar buffer.
            need_test (bool): if this parameter is True, the accurate loss
                will be calculated during training.
        """
        X, y, y_outlier, batch_size, epochs, buffer_size = self.__train_icarl_header(
            X, y, y_outlier, batch_size, epochs, buffer_size, need_test
        )
        self.__train_icarl_body(
            X, y, y_outlier, batch_size, epochs, buffer_size, need_test
        )
        self.__train_icarl_footer(
            X, y, y_outlier, batch_size, epochs, buffer_size, need_test
        )

    def __train_icarl_header(
        self,
        X: torch.Tensor,
        y: torch.Tensor,
        y_outlier: torch.Tensor,
        batch_size: int,
        epochs: int,
        buffer_size: int,
        need_test: bool = False,
    ):
        """
        This function is used to preprocess the input data.

        Args:
            X (torch.Tensor): the input data.
            y (torch.Tensor): the target value.
            y_outlier (torch.Tensor): the outlier data.
            batch_size (int): the batch size for training.
            epochs (int): the number of epochs for training.
            buffer_size (int): the size of the exemplar buffer.
            need_test (bool): if this parameter is True, the accurate loss
                will be calculated during training.

        Returns:
            out (tuple): the preprocessed data.
        """
        self.x_example = None
        self.y_example = None
        # TODO: need_test is not used in this function

        return (X, y, y_outlier, batch_size, epochs, buffer_size)

    def __train_icarl_body(
        self,
        X: torch.Tensor,
        y: torch.Tensor,
        y_outlier: torch.Tensor,
        batch_size: int,
        epochs: int,
        buffer_size: int,
        need_test: bool = False,
    ):
        """
        This function is the main part of training.

        Args:
            X (torch.Tensor): the input data.
            y (torch.Tensor): the target value.
            y_outlier (torch.Tensor): the outlier data.
            batch_size (int): the batch size for training.
            epochs (int): the number of epochs for training.
            buffer_size (int): the size of the exemplar buffer.
            need_test (bool): if this parameter is True, the accurate loss
                will be calculated during training.
        """
        # train epochs times
        for epoch in range(epochs):
            logging.info(f"Starting epoch {epoch + 1}/{epochs}")
            # use torch.utils.data.DataLoader to load the data
            x_loader = TorchDataLoader(X, batch_size=batch_size)
            y_loader = TorchDataLoader(y, batch_size=batch_size)

            for x_batch, y_batch in zip(x_loader, y_loader):
                # get a batch of x and y
                x_batch = x_batch.to(self.device)
                y_batch = y_batch.to(self.device)

                # zero the gradient
                self.optimizer.zero_grad()
                out = self.net(x_batch)
                if self.task == "regression":
                    out = out.reshape(-1)
                # calculate the loss
                loss = self.criterion(out, y_batch)
                if self.x_example is not None:
                    out_e = self.net(self.x_example)
                    loss += self.criterion(out_e, self.y_example)
                # backward and optimize
                loss.backward()

                # using gradient clipping to avoid gradient explosion
                torch.nn.utils.clip_grad_norm_(self.net.parameters(), max_norm=1.0)
                self.optimizer.step()

    def __train_icarl_footer(
        self,
        X: torch.Tensor,
        y: torch.Tensor,
        y_outlier: torch.Tensor,
        batch_size: int,
        epochs: int,
        buffer_size: int,
        need_test: bool = False,
    ):
        """
        This function is used to postprocess the model after training.

        Args:
            X (torch.Tensor): the input data.
            y (torch.Tensor): the target value.
            y_outlier (torch.Tensor): the outlier data.
            batch_size (int): the batch size for training.
            epochs (int): the number of epochs for training.
            buffer_size (int): the size of the exemplar buffer.
            need_test (bool): if this parameter is True, the accurate loss
                will be calculated during training.
        """
        # prepare some variables here
        examples = [[] for _ in range(self.output_dim)]
        buffer_class = int(buffer_size / self.output_dim)

        # post processing
        if self.task == "regression":
            if self.x_example is None:
                features = self.net.feature_extractor(X).detach()
                avg = torch.mean(features, dim=0).unsqueeze(0)
                distance = torch.norm(features - avg, dim=1)
                chosen_k = min(buffer_size, len(distance))
                _, indices = torch.topk(distance, k=chosen_k, largest=False)
                self.x_example = X[indices].to(self.device)
                self.y_example = y[indices].to(self.device)
        elif self.task == "classification":
            update = False
            # iterate over all output dimensions
            for i in range(self.output_dim):
                if len(examples[i]) < buffer_class and i in y:
                    # get some information of the current class
                    targeted_data_list = [X[j] for j in range(len(y)) if y[j][i] == 1]
                    data_class = (
                        torch.stack(targeted_data_list).to(self.device)
                        if len(targeted_data_list) > 0
                        else torch.tensor([]).to(self.device)
                    )
                    features = self.net.feature_extractor(data_class).detach()
                    avg = torch.mean(features, dim=0).unsqueeze(0)
                    distance = torch.norm(features - avg, dim=1)
                    chosen_k = min(buffer_class - len(examples[i]), len(distance))
                    _, indices = torch.topk(distance, k=chosen_k, largest=False)
                    examples[i] = (
                        data_class[indices]
                        if len(examples[i]) == 0
                        else torch.cat((examples[i], data_class[indices]), dim=0)
                    )
                    update = True

            # if update is needed
            if update:
                self.x_example = []
                for i in range(self.output_dim):
                    if len(examples[i]) == 0:
                        continue
                    if len(self.x_example) == 0:
                        self.x_example = examples[i]
                        self.y_example = torch.ones(examples[i].shape[0]) * i
                    else:
                        self.x_example = torch.cat((self.x_example, examples[i]), dim=0)
                        self.y_example = torch.cat(
                            (self.y_example, torch.ones(examples[i].shape[0]) * i),
                            dim=0,
                        )

                # convert the data type of y to long and move the data to device
                self.x_example = self.x_example.to(self.device)
                self.y_example = self.y_example.long().to(self.device)

    def calculate_loss(self, X: torch.Tensor, y: torch.Tensor) -> float:
        """
        This function is used to calculate the loss of the model.

        Args:
            X (torch.Tensor): the input data.
            y (torch.Tensor): the target value.

        Returns:
            out (float): the calculated loss.
        """
        # calculate the loss
        out = self.net(X)
        if self.task == "regression":
            out = out.reshape(-1)
        loss = self.criterion(out, y)

        return loss.item()


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
        """
        Args:
            dataloader (Dataloader): the dataloader object that contains the dataset.
            ensemble (int): the number of models in the ensemble.
            device (Literal["cpu"]): the device that you want to use for training.
        """
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

    def process_model(self, **kwargs):
        """
        This function is used to process the model before training.

        Args:
            **kwargs: any arguments that you want to pass.
        """
        # choose criterion for different tasks
        if self.task == "classification":
            self.criterion = nn.CrossEntropyLoss().to(self.device)
            self.loss = classification_loss_tree(self.net)
        elif self.task == "regression":
            self.criterion = nn.MSELoss().to(self.device)
            self.loss = regression_loss_tree(self.net)
        else:
            logging.error(f"Task not supported: {self.task}")
            raise ValueError("Task not supported.")

    def train_naive(
        self,
        X: torch.Tensor,
        y: torch.Tensor,
        y_outlier: torch.Tensor,
        batch_size: int,
        epochs: int,
        need_test: bool = False,
    ):
        """
        In this function, we will train the model using the input data.

        Args:
            X (torch.Tensor): the input data.
            y (torch.Tensor): the target value.
            y_outlier (torch.Tensor): the outlier data.
            batch_size (int): the batch size for training.
            epochs (int): the number of epochs for training.
            need_test (bool): if this parameter is True, the accurate loss
                will be calculated during training.
        """
        X, y, y_outlier, batch_size, epochs = self.__train_naive_header(
            X, y, y_outlier, batch_size, epochs, need_test
        )
        self.__train_naive_body(X, y, y_outlier, batch_size, epochs, need_test)
        self.__train_naive_footer(X, y, y_outlier, batch_size, epochs, need_test)

    def __train_naive_header(
        self,
        X: torch.Tensor,
        y: torch.Tensor,
        y_outlier: torch.Tensor,
        batch_size: int,
        epochs: int,
        need_test: bool = False,
    ):
        """
        This function is used to preprocess the input data.

        Args:
            X (torch.Tensor): the input data.
            y (torch.Tensor): the target value.
            y_outlier (torch.Tensor): the outlier data.
            batch_size (int): the batch size for training.
            epochs (int): the number of epochs for training.
            need_test (bool): if this parameter is True, the accurate loss
                will be calculated during training.

        Returns:
            out (tuple): the preprocessed data.
        """
        # if the task is classification, we should convert the data type of y to long
        if self.task == "classification":
            y = y.long()

        return (X, y, y_outlier, batch_size, epochs)

    def __train_naive_body(
        self,
        X: torch.Tensor,
        y: torch.Tensor,
        y_outlier: torch.Tensor,
        batch_size: int,
        epochs: int,
        need_test: bool = False,
    ):
        """
        This function is the main part of training.

        Args:
            X (torch.Tensor): the input data.
            y (torch.Tensor): the target value.
            y_outlier (torch.Tensor): the outlier data.
            batch_size (int): the batch size for training.
            epochs (int): the number of epochs for training.
            need_test (bool): if this parameter is True, the accurate loss
                will be calculated during training.
        """
        # training for mlp and armnet with epochs times
        self.net.fit(X, y)

    def __train_naive_footer(
        self,
        X: torch.Tensor,
        y: torch.Tensor,
        y_outlier: torch.Tensor,
        batch_size: int,
        epochs: int,
        need_test: bool = False,
    ):
        """
        This function is used to postprocess the model after training.

        Args:
            X (torch.Tensor): the input data.
            y (torch.Tensor): the target value.
            y_outlier (torch.Tensor): the outlier data.
            batch_size (int): the batch size for training.
            epochs (int): the number of epochs for training.
            need_test (bool): if this parameter is True, the accurate loss
                will be calculated during training.
        """
        # test preparation
        if need_test:
            accuracy_loss = self.loss.loss(X, y, y_outlier)
            logging.info(f"Current accuracy loss: {accuracy_loss}")

    def train_icarl(
        self,
        X: torch.Tensor,
        y: torch.Tensor,
        y_outlier: torch.Tensor,
        batch_size: int,
        epochs: int,
        buffer_size: int,
        need_test: bool = False,
    ):
        """
        iCaRL training process only supports neural network models.

        Args:
            X (torch.Tensor): the input data.
            y (torch.Tensor): the target value.
            y_outlier (torch.Tensor): the outlier data.
            batch_size (int): the batch size for training.
            epochs (int): the number of epochs for training.
            buffer_size (int): the size of the exemplar buffer.
            need_test (bool): if this parameter is True, the accurate loss
                will be calculated during training.
        """
        logging.error(f"Model not supported: {self.model_type}")
        raise ValueError("ICaRL only supports NN model.")

    def calculate_loss(self, X: torch.Tensor, y: torch.Tensor) -> float:
        """
        This function is used to calculate the loss of the model.

        Args:
            X (torch.Tensor): the input data.
            y (torch.Tensor): the target value.

        Returns:
            out (float): the calculated loss.
        """
        # calculate the loss
        out = torch.tensor(self.net.predict(X)).reshape(-1).to(self.device).float()
        loss = self.criterion(out, y.reshape(-1).float())

        return loss.item()


class GdbtModel(ModelTemplate):
    """
    A simple GBDT model for classification and regression tasks.
    """

    def __init__(
        self,
        dataloader: Dataloader,
        ensemble: int = 1,
        device: Literal["cpu"] = "cpu",
    ):
        """
        Args:
            dataloader (Dataloader): the dataloader object that contains the dataset.
            ensemble (int): the number of models in the ensemble.
            device (Literal["cpu"]): the device that you want to use for training.
        """
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

    def process_model(self, **kwargs):
        """
        This function is used to process the model before training.

        Args:
            **kwargs: any arguments that you want to pass.
        """
        # choose criterion for different tasks
        if self.task == "classification":
            self.criterion = nn.CrossEntropyLoss().to(self.device)
            self.loss = classification_loss_tree(self.net)
        elif self.task == "regression":
            self.criterion = nn.MSELoss().to(self.device)
            self.loss = regression_loss_tree(self.net)
        else:
            logging.error(f"Task not supported: {self.task}")
            raise ValueError("Task not supported.")

    def train_naive(
        self,
        X: torch.Tensor,
        y: torch.Tensor,
        y_outlier: torch.Tensor,
        batch_size: int,
        epochs: int,
        need_test: bool = False,
    ):
        """
        In this function, we will train the model using the input data.

        Args:
            X (torch.Tensor): the input data.
            y (torch.Tensor): the target value.
            y_outlier (torch.Tensor): the outlier data.
            batch_size (int): the batch size for training.
            epochs (int): the number of epochs for training.
            need_test (bool): if this parameter is True, the accurate loss
                will be calculated during training.
        """
        X, y, y_outlier, batch_size, epochs = self.__train_naive_header(
            X, y, y_outlier, batch_size, epochs, need_test
        )
        self.__train_naive_body(X, y, y_outlier, batch_size, epochs, need_test)
        self.__train_naive_footer(X, y, y_outlier, batch_size, epochs, need_test)

    def __train_naive_header(
        self,
        X: torch.Tensor,
        y: torch.Tensor,
        y_outlier: torch.Tensor,
        batch_size: int,
        epochs: int,
        need_test: bool = False,
    ):
        """
        This function is used to preprocess the input data.

        Args:
            X (torch.Tensor): the input data.
            y (torch.Tensor): the target value.
            y_outlier (torch.Tensor): the outlier data.
            batch_size (int): the batch size for training.
            epochs (int): the number of epochs for training.
            need_test (bool): if this parameter is True, the accurate loss
                will be calculated during training.

        Returns:
            out (tuple): the preprocessed data.
        """
        # if the task is classification, we should convert the data type of y to long
        if self.task == "classification":
            y = y.argmax(dim=1)

        return (X, y, y_outlier, batch_size, epochs)

    def __train_naive_body(
        self,
        X: torch.Tensor,
        y: torch.Tensor,
        y_outlier: torch.Tensor,
        batch_size: int,
        epochs: int,
        need_test: bool = False,
    ):
        """
        This function is the main part of training.

        Args:
            X (torch.Tensor): the input data.
            y (torch.Tensor): the target value.
            y_outlier (torch.Tensor): the outlier data.
            batch_size (int): the batch size for training.
            epochs (int): the number of epochs for training.
            need_test (bool): if this parameter is True, the accurate loss
                will be calculated during training.
        """
        # training for mlp and armnet with epochs times
        self.net.fit(X, y)

    def __train_naive_footer(
        self,
        X: torch.Tensor,
        y: torch.Tensor,
        y_outlier: torch.Tensor,
        batch_size: int,
        epochs: int,
        need_test: bool = False,
    ):
        """
        This function is used to postprocess the model after training.

        Args:
            X (torch.Tensor): the input data.
            y (torch.Tensor): the target value.
            y_outlier (torch.Tensor): the outlier data.
            batch_size (int): the batch size for training.
            epochs (int): the number of epochs for training.
            need_test (bool): if this parameter is True, the accurate loss
                will be calculated during training.
        """
        # test preparation
        if need_test:
            accuracy_loss = self.loss.loss(X, y, y_outlier)
            logging.info(f"Current accuracy loss: {accuracy_loss}")

    def train_icarl(
        self,
        X: torch.Tensor,
        y: torch.Tensor,
        y_outlier: torch.Tensor,
        batch_size: int,
        epochs: int,
        buffer_size: int,
        need_test: bool = False,
    ):
        """
        iCaRL training process only supports neural network models.

        Args:
            X (torch.Tensor): the input data.
            y (torch.Tensor): the target value.
            y_outlier (torch.Tensor): the outlier data.
            batch_size (int): the batch size for training.
            epochs (int): the number of epochs for training.
            need_test (bool): if this parameter is True, the accurate loss
                will be calculated during training.
        """
        logging.error(f"Model not supported: {self.model_type}")
        raise ValueError("ICaRL only supports NN model.")

    def calculate_loss(self, X: torch.Tensor, y: torch.Tensor) -> float:
        """
        This function is used to calculate the loss of the model.

        Args:
            X (torch.Tensor): the input data.
            y (torch.Tensor): the target value.

        Returns:
            out (float): the calculated loss.
        """
        # predict the target value
        if self.task == "classification":
            out = self.net.predict_proba(X)
        elif self.task == "regression":
            out = self.net.predict(X)
        else:
            logging.error(f"Task not supported: {self.task}")
            raise ValueError("Task not supported.")

        # calculate the loss
        loss = self.criterion(torch.tensor(out).reshape(-1), y.reshape(-1))
        return loss.item()


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
        """
        Args:
            dataloader (Dataloader): the dataloader object that contains the dataset.
            ensemble (int): the number of models in the ensemble.
            device (Literal["cpu"]): the device that you want to use for training.
        """
        super().__init__(dataloader, ensemble, device)
        self.model_type = "tabnet"
        # initialization for TabNet model
        if self.task == "classification":
            self.net = TabNetClassifier()
        elif self.task == "regression":
            self.net = TabNetRegressor()
        else:
            logging.error(f"Task {self.task} not supported.")
            raise ValueError("Task not supported.")

    def process_model(self, **kwargs):
        """
        This function is used to process the model before training.

        Args:
            **kwargs: any arguments that you want to pass.
        """
        # choose criterion for different tasks
        if self.task == "classification":
            self.criterion = nn.CrossEntropyLoss().to(self.device)
            self.loss = classification_loss_tree(self.net)
        elif self.task == "regression":
            self.criterion = nn.MSELoss().to(self.device)
            self.loss = regression_loss_tree(self.net)
        else:
            logging.error(f"Task not supported: {self.task}")
            raise ValueError("Task not supported.")

    def train_naive(
        self,
        X: torch.Tensor,
        y: torch.Tensor,
        y_outlier: torch.Tensor,
        batch_size: int,
        epochs: int,
        need_test: bool = False,
    ):
        """
        In this function, we will train the model using the input data.

        Args:
            X (torch.Tensor): the input data.
            y (torch.Tensor): the target value.
            y_outlier (torch.Tensor): the outlier data.
            batch_size (int): the batch size for training.
            epochs (int): the number of epochs for training.
            need_test (bool): if this parameter is True, the accurate loss
                will be calculated during training.
        """
        X, y, y_outlier, batch_size, epochs = self.__train_naive_header(
            X, y, y_outlier, batch_size, epochs, need_test
        )
        self.__train_naive_body(X, y, y_outlier, batch_size, epochs, need_test)
        self.__train_naive_footer(X, y, y_outlier, batch_size, epochs, need_test)

    def __train_naive_header(
        self,
        X: torch.Tensor,
        y: torch.Tensor,
        y_outlier: torch.Tensor,
        batch_size: int,
        epochs: int,
        need_test: bool = False,
    ):
        """
        This function is used to preprocess the input data.

        Args:
            X (torch.Tensor): the input data.
            y (torch.Tensor): the target value.
            y_outlier (torch.Tensor): the outlier data.
            batch_size (int): the batch size for training.
            epochs (int): the number of epochs for training.
            need_test (bool): if this parameter is True, the accurate loss
                will be calculated during training.

        Returns:
            out (tuple): the preprocessed data.
        """
        # preprocess the input data according to the task
        if self.task == "regression":
            y = y.reshape(-1, 1)
        # if the task is classification, we should convert the one hot data to 1D tensor
        elif self.task == "classification":
            y = y.argmax(dim=1)

        # transit the data type of X and y to numpy array
        X: np.ndarray = X.cpu().numpy()
        y: np.ndarray = y.cpu().numpy()

        return (X, y, y_outlier, batch_size, epochs)

    def __train_naive_body(
        self,
        X: np.ndarray,
        y: np.ndarray,
        y_outlier: np.ndarray,
        batch_size: int,
        epochs: int,
        need_test: bool = False,
    ):
        """
        This function is the main part of training.

        Args:
            X (torch.Tensor): the input data.
            y (torch.Tensor): the target value.
            y_outlier (torch.Tensor): the outlier data.
            batch_size (int): the batch size for training.
            epochs (int): the number of epochs for training.
            need_test (bool): if this parameter is True, the accurate loss
                will be calculated during training.
        """
        self.net.fit(
            X,
            y,
            batch_size=batch_size,
            virtual_batch_size=batch_size,
            max_epochs=epochs,
        )

    def __train_naive_footer(
        self,
        X: torch.Tensor,
        y: torch.Tensor,
        y_outlier: torch.Tensor,
        batch_size: int,
        epochs: int,
        need_test: bool = False,
    ):
        """
        This function is used to postprocess the model after training.

        Args:
            X (torch.Tensor): the input data.
            y (torch.Tensor): the target value.
            y_outlier (torch.Tensor): the outlier data.
            batch_size (int): the batch size for training.
            epochs (int): the number of epochs for training.
            need_test (bool): if this parameter is True, the accurate loss
                will be calculated during training.
        """
        # test preparation
        if need_test:
            accuracy_loss = self.loss.loss(X, y, y_outlier)
            logging.info(f"Current accuracy loss: {accuracy_loss}")

    def train_icarl(
        self,
        X: torch.Tensor,
        y: torch.Tensor,
        y_outlier: torch.Tensor,
        batch_size: int,
        epochs: int,
        buffer_size: int,
        need_test: bool = False,
    ):
        """
        iCaRL training process only supports neural network models.

        Args:
            X (torch.Tensor): the input data.
            y (torch.Tensor): the target value.
            y_outlier (torch.Tensor): the outlier data.
            batch_size (int): the batch size for training.
            epochs (int): the number of epochs for training.
            need_test (bool): if this parameter is True, the accurate loss
                will be calculated during training.
        """
        logging.error(f"Model not supported: {self.model_type}")
        raise ValueError("ICaRL only supports NN model.")

    def calculate_loss(self, X: torch.Tensor, y: torch.Tensor) -> float:
        """
        This function is used to calculate the loss of the model.

        Args:
            X (torch.Tensor): the input data.
            y (torch.Tensor): the target value.

        Returns:
            out (float): the calculated loss.
        """
        # predict the target value
        if self.task == "classification":
            out = self.net.predict_proba(X)
        elif self.task == "regression":
            out = self.net.predict(X)
        else:
            logging.error(f"Task not supported: {self.task}")
            raise ValueError("Task not supported.")

        # calculate the loss
        out = torch.tensor(out).reshape(-1)
        loss = self.criterion(out.float(), y.reshape(-1).float())

        return loss.item()


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
        """
        Args:
            dataloader (Dataloader): the dataloader object that contains the dataset.
            ensemble (int): the number of models in the ensemble.
            device (Literal["cpu", "cuda"]): the device that you want to use for training.
        """
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

    def process_model(self, lr: float, **kwargs):
        """
        This function is used to process the model before training.

        Args:
            lr (float): the learning rate for the optimizer.
            **kwargs: any arguments that you want to pass.
        """
        # choose optimizer for tree models
        self.optimizer = torch.optim.SGD(
            filter(lambda p: p.requires_grad, self.net.parameters()),
            lr=lr,
        )

        # choose criterion for different tasks
        if self.task == "classification":
            self.criterion = nn.CrossEntropyLoss().to(self.device)
            self.loss = classification_loss(self.net)
        elif self.task == "regression":
            self.criterion = nn.MSELoss().to(self.device)
            self.loss = regression_loss(self.net)
        else:
            logging.error(f"Task not supported: {self.task}")
            raise ValueError("Task not supported.")

    def __preprocess_x(self, X: torch.Tensor) -> dict:
        """
        This function is used to preprocess the input data. The built-in ArmNet model
        requires a dictionary (which contains the value and id of the input data) as
        the input.

        Args:
            X (torch.Tensor): the input data.

        Returns:
            out (dict): the preprocessed data.
        """
        return {
            "value": X,
            "id": torch.arange(X.shape[1])
            .repeat(X.shape[0])
            .view(X.shape[0], -1)
            .to(self.device),
        }

    def train_naive(
        self,
        X: torch.Tensor,
        y: torch.Tensor,
        y_outlier: torch.Tensor,
        batch_size: int,
        epochs: int,
        need_test: bool = False,
    ):
        """
        In this function, we will train the model using the input data.

        Args:
            X (torch.Tensor): the input data.
            y (torch.Tensor): the target value.
            y_outlier (torch.Tensor): the outlier data.
            batch_size (int): the batch size for training.
            epochs (int): the number of epochs for training.
            need_test (bool): if this parameter is True, the accurate loss
                will be calculated during training.
        """
        X, y, y_outlier, batch_size, epochs = self.__train_naive_header(
            X, y, y_outlier, batch_size, epochs, need_test
        )
        self.__train_naive_body(X, y, y_outlier, batch_size, epochs, need_test)
        self.__train_naive_footer(X, y, y_outlier, batch_size, epochs, need_test)

    def __train_naive_header(
        self,
        X: torch.Tensor,
        y: torch.Tensor,
        y_outlier: torch.Tensor,
        batch_size: int,
        epochs: int,
        need_test: bool = False,
    ):
        """
        This function is used to preprocess the input data.

        Args:
            X (torch.Tensor): the input data.
            y (torch.Tensor): the target value.
            y_outlier (torch.Tensor): the outlier data.
            batch_size (int): the batch size for training.
            epochs (int): the number of epochs for training.
            need_test (bool): if this parameter is True, the accurate loss
                will be calculated during training.

        Returns:
            out (tuple): the preprocessed data.
        """
        # if the task is classification, we should convert the data type of y to long
        if self.task == "classification":
            y = y.long()

        return (X, y, y_outlier, batch_size, epochs)

    def __train_naive_body(
        self,
        X: torch.Tensor,
        y: torch.Tensor,
        y_outlier: torch.Tensor,
        batch_size: int,
        epochs: int,
        need_test: bool = False,
    ):
        """
        This function is the main part of training.

        Args:
            X (torch.Tensor): the input data.
            y (torch.Tensor): the target value.
            y_outlier (torch.Tensor): the outlier data.
            batch_size (int): the batch size for training.
            epochs (int): the number of epochs for training.
            need_test (bool): if this parameter is True, the accurate loss
                will be calculated during training.
        """
        # training for mlp and armnet with epochs times
        for epoch in range(epochs):
            # logging some information for each epoch
            logging.info(f"Starting epoch {epoch + 1}/{epochs}")
            # use torch.utils.data.DataLoader to load the data
            x_loader = TorchDataLoader(X, batch_size=batch_size)
            y_loader = TorchDataLoader(y, batch_size=batch_size)

            for x_batch, y_batch in zip(x_loader, y_loader):
                # get a batch of x and y
                x_batch = self.__preprocess_x(x_batch.to(self.device).float())
                y_batch = y_batch.to(self.device).float()

                # using gradient descent to optimize the parameters
                self.optimizer.zero_grad()
                # the loss function require 0D or 1D tensors for input
                out: torch.Tensor = self.net(x_batch).reshape(-1)
                ref: torch.Tensor = y_batch.reshape(-1)
                loss = self.criterion(out, ref)
                loss.backward()

                # using gradient clipping to avoid gradient explosion
                torch.nn.utils.clip_grad_norm_(self.net.parameters(), max_norm=1.0)
                self.optimizer.step()

    def __train_naive_footer(
        self,
        X: torch.Tensor,
        y: torch.Tensor,
        y_outlier: torch.Tensor,
        batch_size: int,
        epochs: int,
        need_test: bool = False,
    ):
        """
        This function is used to postprocess the model after training.

        Args:
            X (torch.Tensor): the input data.
            y (torch.Tensor): the target value.
            y_outlier (torch.Tensor): the outlier data.
            batch_size (int): the batch size for training.
            epochs (int): the number of epochs for training.
            need_test (bool): if this parameter is True, the accurate loss
                will be calculated during training.
        """
        # test preparation
        if need_test:
            accuracy_loss = self.loss.loss(self.__preprocess_x(X), y, y_outlier)
            logging.info(f"Current accuracy loss: {accuracy_loss}")

    def train_icarl(
        self,
        X: torch.Tensor,
        y: torch.Tensor,
        y_outlier: torch.Tensor,
        batch_size: int,
        epochs: int,
        buffer_size: int,
        need_test: bool = False,
    ):
        """
        iCaRL training process only supports neural network models.

        Args:
            X (torch.Tensor): the input data.
            y (torch.Tensor): the target value.
            y_outlier (torch.Tensor): the outlier data.
            batch_size (int): the batch size for training.
            epochs (int): the number of epochs for training.
            need_test (bool): if this parameter is True, the accurate loss
                will be calculated during training.
        """
        logging.error(f"Model not supported: {self.model_type}")
        raise ValueError("ICaRL only supports NN model.")

    def calculate_loss(self, X: torch.Tensor, y: torch.Tensor) -> float:
        """
        This function is used to calculate the loss of the model.

        Args:
            X (torch.Tensor): the input data.
            y (torch.Tensor): the target value.

        Returns:
            out (float): the calculated loss.
        """
        # calculate the loss
        out = self.net(self.__preprocess_x(X))
        loss = self.criterion(out.reshape(-1), y.reshape(-1))

        return loss.item()


class CluStreamModel(ModelTemplate):
    """
    A simple CluStream model for clustering tasks.
    """

    def __init__(
        self,
        dataloader: Dataloader,
        ensemble: int = 1,
        device: Literal["cpu"] = "cpu",
    ):
        """
        Args:
            dataloader (Dataloader): the dataloader object that contains the dataset.
            ensemble (int): the number of models in the ensemble.
            device (Literal["cpu"]): the device that you want to use for training.
        """
        super().__init__(dataloader, ensemble, device)
        self.model_type = "clustream"
        self.net = CluStreamNet()

    def process_model(self, **kwargs):
        """
        This function is used to process the model before training.
        In this model, we don't need to process the model before training.

        Args:
            **kwargs: any arguments that you want to pass.
        """
        pass

    def train_cluster(self, X: torch.Tensor):
        """
        Train the model with the input data.

        Args:
            X (torch.Tensor): features of the input data.
        """
        self.net.fit(X)

    def predict_cluster(self, X: torch.Tensor) -> torch.Tensor:
        """
        Predict the cluster of the input data.

        Args:
            X (torch.Tensor): features of the input data.

        Returns:
            out (torch.Tensor): the predicted cluster of the input data.
        """
        return self.net(X)


class DbStreamModel(ModelTemplate):
    """
    A simple DBStream model for clustering tasks.
    """

    def __init__(
        self,
        dataloader: Dataloader,
        ensemble: int = 1,
        device: Literal["cpu"] = "cpu",
    ):
        """
        Args:
            dataloader (Dataloader): the dataloader object that contains the dataset.
            ensemble (int): the number of models in the ensemble.
            device (Literal["cpu"]): the device that you want to use for training.
        """
        super().__init__(dataloader, ensemble, device)
        self.model_type = "dbstream"
        self.net = DbStreamNet()

    def process_model(self, **kwargs):
        """
        This function is used to process the model before training.
        In this model, we don't need to process the model before training.

        Args:
            **kwargs: any arguments that you want to pass.
        """
        pass

    def train_cluster(self, X: torch.Tensor):
        """
        Train the model with the input data.

        Args:
            X (torch.Tensor): features of the input data.
        """
        self.net.fit(X)

    def predict_cluster(self, X: torch.Tensor) -> torch.Tensor:
        """
        Predict the cluster of the input data.

        Args:
            X (torch.Tensor): features of the input data.

        Returns:
            out (torch.Tensor): the predicted cluster of the input data.
        """
        return self.net(X)


class DenStreamModel(ModelTemplate):
    """
    A simple DenStream model for clustering tasks.
    """

    def __init__(
        self,
        dataloader: Dataloader,
        ensemble: int = 1,
        device: Literal["cpu"] = "cpu",
    ):
        """
        Args:
            dataloader (Dataloader): the dataloader object that contains the dataset.
            ensemble (int): the number of models in the ensemble.
            device (Literal["cpu"]): the device that you want to use for training.
        """
        super().__init__(dataloader, ensemble, device)
        self.model_type = "denstream"
        self.net = DenStreamNet()

    def process_model(self, **kwargs):
        """
        This function is used to process the model before training.
        In this model, we don't need to process the model before training.

        Args:
            **kwargs: any arguments that you want to pass.
        """
        pass

    def train_cluster(self, X: torch.Tensor):
        """
        Train the model with the input data.

        Args:
            X (torch.Tensor): features of the input data.
        """
        self.net.fit(X)

    def predict_cluster(self, X: torch.Tensor) -> torch.Tensor:
        """
        Predict the cluster of the input data.

        Args:
            X (torch.Tensor): features of the input data.

        Returns:
            out (torch.Tensor): the predicted cluster of the input data.
        """
        return self.net(X)


class StreamKMeansModel(ModelTemplate):
    """
    A simple StreamKMeans model for clustering tasks.
    """

    def __init__(
        self,
        dataloader: Dataloader,
        ensemble: int = 1,
        device: Literal["cpu"] = "cpu",
    ):
        """
        Args:
            dataloader (Dataloader): the dataloader object that contains the dataset.
            ensemble (int): the number of models in the ensemble.
            device (Literal["cpu"]): the device that you want to use for training.
        """
        super().__init__(dataloader, ensemble, device)
        self.model_type = "streamkmeans"
        self.net = StreamKMeansNet()

    def process_model(self, **kwargs):
        """
        This function is used to process the model before training.
        In this model, we don't need to process the model before training.

        Args:
            **kwargs: any arguments that you want to pass.
        """
        pass

    def train_cluster(self, X: torch.Tensor):
        """
        Train the model with the input data.

        Args:
            X (torch.Tensor): features of the input data.
        """
        self.net.fit(X)

    def predict_cluster(self, X: torch.Tensor) -> torch.Tensor:
        """
        Predict the cluster of the input data.

        Args:
            X (torch.Tensor): features of the input data.

        Returns:
            out (torch.Tensor): the predicted cluster of the input data.
        """
        return self.net(X)


class XStreamDetectorModel(ModelTemplate):
    """
    A simple XStreamDetector model for outlier detection tasks.
    """

    def __init__(
        self,
        dataloader: Dataloader,
        ensemble: int = 1,
        device: Literal["cpu"] = "cpu",
    ):
        """
        Args:
            dataloader (Dataloader): the dataloader object that contains the dataset.
            ensemble (int): the number of models in the ensemble.
            device (Literal["cpu"]): the device that you want to use for training.
        """
        super().__init__(dataloader, ensemble, device)
        self.model_type = "xstream"
        self.net = XStreamDetectorNet()

    def process_model(self, **kwargs):
        """
        This function is used to process the model before training.
        In this model, we don't need to process the model before training.

        Args:
            **kwargs: any arguments that you want to pass.
        """
        pass

    def get_outlier_with_stream_model(self, X: torch.Tensor) -> torch.Tensor:
        """
        Get the outlier score of the input data using the XStream model.

        Args:
            X (torch.Tensor): the input data.

        Returns:
            out (torch.Tensor): the outlier score of the input data.
        """
        return self.net.get_model_score(X)

    def get_outlier(self, X: torch.Tensor) -> torch.Tensor:
        """
        Get the outlier points using PYOD models.

        Args:
            X (torch.Tensor): the input data.

        Returns:
            out (torch.Tensor): a 0-1 vector representing the outlier points.
        """
        return self.net(X)


class RShashDetectorModel(ModelTemplate):
    """
    A simple RShashDetector model for outlier detection tasks.
    """

    def __init__(
        self,
        dataloader: Dataloader,
        ensemble: int = 1,
        device: Literal["cpu"] = "cpu",
    ):
        """
        Args:
            dataloader (Dataloader): the dataloader object that contains the dataset.
            ensemble (int): the number of models in the ensemble.
            device (Literal["cpu"]): the device that you want to use for training.
        """
        super().__init__(dataloader, ensemble, device)
        self.model_type = "rshash"
        self.net = RShashDetectorNet()

    def process_model(self, **kwargs):
        """
        This function is used to process the model before training.
        In this model, we don't need to process the model before training.

        Args:
            **kwargs: any arguments that you want to pass.
        """
        pass

    def get_outlier_with_stream_model(self, X: torch.Tensor) -> torch.Tensor:
        """
        Get the outlier score of the input data using the RShash model.

        Args:
            X (torch.Tensor): the input data.

        Returns:
            out (torch.Tensor): the outlier score of the input data.
        """
        return self.net.get_model_score(X)

    def get_outlier(self, X: torch.Tensor) -> torch.Tensor:
        """
        Get the outlier points using PYOD models.

        Args:
            X (torch.Tensor): the input data.

        Returns:
            out (torch.Tensor): a 0-1 vector representing the outlier points.
        """
        return self.net(X)


class HSTreeDetectorModel(ModelTemplate):
    """
    A simple HSTreeDetector model for outlier detection tasks.
    """

    def __init__(
        self,
        dataloader: Dataloader,
        ensemble: int = 1,
        device: Literal["cpu"] = "cpu",
    ):
        """
        Args:
            dataloader (Dataloader): the dataloader object that contains the dataset.
            ensemble (int): the number of models in the ensemble.
            device (Literal["cpu"]): the device that you want to use for training.
        """
        super().__init__(dataloader, ensemble, device)
        self.model_type = "hstree"
        self.net = HSTreeDetectorNet()

    def process_model(self, **kwargs):
        """
        This function is used to process the model before training.
        In this model, we don't need to process the model before training.

        Args:
            **kwargs: any arguments that you want to pass.
        """
        pass

    def get_outlier_with_stream_model(self, X: torch.Tensor) -> torch.Tensor:
        """
        Get the outlier score of the input data using the HSTree model.

        Args:
            X (torch.Tensor): the input data.

        Returns:
            out (torch.Tensor): the outlier score of the input data.
        """
        return self.net.get_model_score(X)

    def get_outlier(self, X: torch.Tensor) -> torch.Tensor:
        """
        Get the outlier points using PYOD models.

        Args:
            X (torch.Tensor): the input data.

        Returns:
            out (torch.Tensor): a 0-1 vector representing the outlier points.
        """
        return self.net(X)


class LodaDetectorModel(ModelTemplate):
    """
    A simple LodaDetector model for outlier detection tasks.
    """

    def __init__(
        self,
        dataloader: Dataloader,
        ensemble: int = 1,
        device: Literal["cpu"] = "cpu",
    ):
        """
        Args:
            dataloader (Dataloader): the dataloader object that contains the dataset.
            ensemble (int): the number of models in the ensemble.
            device (Literal["cpu"]): the device that you want to use for training.
        """
        super().__init__(dataloader, ensemble, device)
        self.model_type = "loda"
        self.net = LodaDetectorNet()

    def process_model(self, **kwargs):
        """
        This function is used to process the model before training.
        In this model, we don't need to process the model before training.

        Args:
            **kwargs: any arguments that you want to pass.
        """
        pass

    def get_outlier_with_stream_model(self, X: torch.Tensor) -> torch.Tensor:
        """
        Get the outlier score of the input data using the LODA model.

        Args:
            X (torch.Tensor): the input data.

        Returns:
            out (torch.Tensor): the outlier score of the input data.
        """
        return self.net.get_model_score(X)

    def get_outlier(self, X: torch.Tensor) -> torch.Tensor:
        """
        Get the outlier points using PYOD models.

        Args:
            X (torch.Tensor): the input data.

        Returns:
            out (torch.Tensor): a 0-1 vector representing the outlier points.
        """
        return self.net(X)


class RrcfDetectorModel(ModelTemplate):
    """
    A simple RrcfDetector model for outlier detection tasks.
    """

    def __init__(
        self,
        dataloader: Dataloader,
        ensemble: int = 1,
        device: Literal["cpu"] = "cpu",
    ):
        """
        Args:
            dataloader (Dataloader): the dataloader object that contains the dataset.
            ensemble (int): the number of models in the ensemble.
            device (Literal["cpu"]): the device that you want to use for training.
        """
        super().__init__(dataloader, ensemble, device)
        self.model_type = "rrcf"
        self.net = RrcfDetectorNet()

    def process_model(self, **kwargs):
        """
        This function is used to process the model before training.
        In this model, we don't need to process the model before training.

        Args:
            **kwargs: any arguments that you want to pass.
        """
        pass

    def get_outlier_with_stream_model(self, X: torch.Tensor) -> torch.Tensor:
        """
        Get the outlier score of the input data using the RRCF model.

        Args:
            X (torch.Tensor): the input data.

        Returns:
            out (torch.Tensor): the outlier score of the input data.
        """
        return self.net.get_model_score(X)

    def get_outlier(self, X: torch.Tensor) -> torch.Tensor:
        """
        Get the outlier points using PYOD models.

        Args:
            X (torch.Tensor): the input data.

        Returns:
            out (torch.Tensor): a 0-1 vector representing the outlier points.
        """
        return self.net(X)
