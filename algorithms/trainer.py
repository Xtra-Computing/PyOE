import os
import time
import torch
import logging
import torch.distributed as dist
from abc import abstractmethod
from torch.utils.data import DataLoader as TorchDataLoader
from torch.utils.data.distributed import DistributedSampler
from .loss import *
from ..preprocessors import Preprocessor
from ..models import ModelTemplate
from ..dataloaders import Dataloader, DataloaderWrapper


class TrainerTemplate:
    """
    This is a template for training function. We implemented some common training
    functions and you can also implement your own training function using this template.
    """

    def __init__(
        self,
        dataloader: Dataloader,
        model: ModelTemplate,
        preprocessor: Preprocessor,
        lr: float = 0.01,
        epochs: int = 1,
        batch_size: int = 64,
        buffer_size: int = 100,
        **kargws,
    ) -> None:
        """
        This is the constructor of the TrainerTemplate class.

        Args:
            dataloader (Dataloader): The dataloader object.
            model (ModelTemplate): The model object.
            preprocessor (Preprocessor): The preprocessor object.
            lr (float): The learning rate.
            epochs (int): The number of epochs.
            batch_size (int): The batch size.
            buffer_size (int): The buffer size.
            **kwargs: Additional optional parameters.
        """
        # basic assignments
        self.dataloader = dataloader
        self.model = model
        self.preprocessor = preprocessor
        self.lr = lr
        self.epochs = epochs
        self.batch_size = batch_size
        self.buffer_size = buffer_size
        # device, task, model_type and net
        self.device = self.model.get_device()
        self.task = self.dataloader.get_task()
        self.model_type = self.model.get_model_type()
        self.net = self.model.get_net()
        # results of the last time
        self.running_time: float | None = None

    @abstractmethod
    def train(self, need_test: bool = False) -> None:
        """
        This is the training function. You can implement your own training function here.

        Args:
            need_test (bool): If this parameter is True, the accurate loss will be calculated during training.
        """
        pass

    def _time_start(self):
        """
        This function is used to record the start time of the training process.
        """
        self.start_time = time.time()

    def _time_end(self):
        """
        This function is used to record the end time of the training process.
        """
        self.running_time = time.time() - self.start_time
        logging.info(f"Training time: {self.running_time:.2f} seconds.")

    def get_last_training_time(self) -> Optional[float]:
        """
        This function is used to get the last training time.

        Returns:
            out (Optional[float]): The last training time.
        """
        return self.running_time


class NaiveTrainer(TrainerTemplate):
    """
    This class is a training wrapper for the model. It will call the
    model's preprocessing and training function to train the model.
    """

    def __init__(
        self,
        dataloader: Dataloader,
        model: ModelTemplate,
        preprocessor: Preprocessor,
        lr: float = 0.01,
        epochs: int = 1,
        batch_size: int = 64,
        buffer_size: int = 100,
        **kargws,
    ) -> None:
        super().__init__(
            dataloader,
            model,
            preprocessor,
            lr,
            epochs,
            batch_size,
            buffer_size,
            **kargws,
        )
        # preprocess the model
        self.model.process_model(lr=self.lr)

    def _train(
        self,
        X: torch.Tensor,
        y: torch.Tensor,
        y_outlier: torch.Tensor,
        need_test: bool = False,
        **kwargs,
    ) -> None:
        """
        This function is used to train the model for a single batch of data.

        Args:
            X (torch.Tensor): The input data.
            y (torch.Tensor): The target data.
            y_outlier (torch.Tensor): The outlier target data.
            need_test (bool): If this parameter is True, the accurate loss will be calculated during training.
            **kwargs: Additional optional parameters.
        """
        # use the correct device for training
        X, y, y_outlier = (
            torch.tensor(X, dtype=torch.float).to(self.device),
            torch.tensor(y, dtype=torch.float).to(self.device),
            (
                None
                if y_outlier is None
                else torch.tensor(y_outlier, dtype=torch.float).to(self.device)
            ),
        )

        # train the model
        self.model.train_naive(X, y, y_outlier, self.batch_size, self.epochs, need_test)

    def train(self, need_test: bool = False) -> None:
        """
        This function is used to train the model with regression or classification task using naive algorithm.

        Args:
            need_test (bool): If this parameter is True, the accurate loss will be calculated during training.
        """
        self._time_start()

        # load data using the dataloader
        torch_dataloader = TorchDataLoader(
            DataloaderWrapper(self.dataloader, need_test),
            batch_size=self.batch_size,
            shuffle=True,
        )

        # train the model
        for X, y, y_outlier in torch_dataloader:
            X = self.preprocessor.fill(X)
            self._train(X, y, y_outlier, need_test=need_test)

        self._time_end()


class IcarlTrainer(TrainerTemplate):
    """
    This class is a training wrapper for the model. It will call the
    model's preprocessing and training function to train the model.
    """

    def __init__(
        self,
        dataloader: Dataloader,
        model: ModelTemplate,
        preprocessor: Preprocessor,
        lr: float = 0.01,
        epochs: int = 1,
        batch_size: int = 128,
        buffer_size: int = 100,
        **kargws,
    ) -> None:
        super().__init__(
            dataloader,
            model,
            preprocessor,
            lr,
            epochs,
            batch_size,
            buffer_size,
            **kargws,
        )
        # preprocess the model
        self.model.process_model(lr=self.lr)

    def _train(
        self,
        X: torch.Tensor,
        y: torch.Tensor,
        y_outlier: torch.Tensor = None,
        need_test: bool = False,
        **kargws,
    ) -> None:
        """
        This function is used to train the model for a single batch of data.

        Args:
            X (torch.Tensor): The input data.
            y (torch.Tensor): The target data.
            y_outlier (torch.Tensor): The outlier target data.
            need_test (bool): If this parameter is True, the accurate loss will be calculated during training.
            **kwargs: Additional optional parameters.
        """
        # use the correct device for training
        X, y, y_outlier = (
            torch.tensor(X, dtype=torch.float).to(self.device),
            torch.tensor(y, dtype=torch.float).to(self.device),
            (
                None
                if y_outlier is None
                else torch.tensor(y_outlier, dtype=torch.float).to(self.device)
            ),
        )

        # train the model
        self.model.train_icarl(
            X, y, y_outlier, self.batch_size, self.epochs, self.buffer_size, need_test
        )

    def train(self, need_test: bool = False) -> None:
        """
        This function is used to train the model with regression or classification task using iCaRL algorithm.

        Args:
            need_test (bool): If this parameter is True, the accurate loss will be calculated during training.
        """
        self._time_start()

        # load data using the dataloader
        torch_dataloader = TorchDataLoader(
            DataloaderWrapper(self.dataloader, need_test),
            batch_size=self.batch_size,
            shuffle=True,
        )

        # train the model
        for X, y, y_outlier in torch_dataloader:
            X = self.preprocessor.fill(X)
            self._train(X, y, y_outlier, need_test=need_test)

        self._time_end()


class ClusterTrainer(TrainerTemplate):
    """
    This class is a training wrapper for the model. It will call the
    model's training function to train the model. Actually ClusterModel
    is trained in an online manner, so we don't need to train it in a batch
    manner.
    """

    def __init__(
        self,
        dataloader: Dataloader,
        model: ModelTemplate,
        preprocessor: Preprocessor,
        **kargws,
    ) -> None:
        super().__init__(dataloader, model, preprocessor, **kargws)

    def _train(self, X: torch.Tensor, **kwargs) -> None:
        """
        This function is used to train the model for a single batch of data.

        Args:
            X (torch.Tensor): Features of the input data.
            **kwargs: Additional optional parameters.
        """
        self.model.train_cluster(X)

    def train(self, need_test: bool = False) -> None:
        """
        This function is used to train the model with clustering task.

        Args:
            need_test (bool): If this parameter is True, the accurate loss will be calculated during training.
        """
        self._time_start()

        # load data using the dataloader
        torch_dataloader = TorchDataLoader(
            DataloaderWrapper(self.dataloader, need_test),
            batch_size=self.batch_size,
            shuffle=True,
        )

        # train the model
        for X, _, _ in torch_dataloader:
            X = self.preprocessor.fill(X)
            self._train(X, need_test=need_test)

        self._time_end()


"""
The class below is not used in the current version. Previouly it was used to train
the OutlierModel, but now we could call those training and prediction functions
directly from the model. We keep it commented here for future use.
"""

# class OutlierTrainer(TrainerTemplate):
#     """
#     This class is a training wrapper for the model. It will call the
#     model's training function to train the model. Actually OutlierModel
#     is trained in an online manner, so we don't need to train it in a batch
#     manner.
#     """

#     def __init__(
#         self,
#         dataloader: Dataloader,
#         model: ModelTemplate,
#         preprocessor: Preprocessor,
#         **kargws,
#     ) -> None:
#         super().__init__(dataloader, model, preprocessor, **kargws)

#     def _train(self, X: torch.Tensor, **kwargs) -> None:
#         self.model.train_outlier(X)

#     def train(self, need_test: bool = False) -> None:
#         self._time_start()

#         # load data using the dataloader
#         torch_dataloader = TorchDataLoader(
#             DataloaderWrapper(self.dataloader, need_test),
#             batch_size=self.batch_size,
#             shuffle=True,
#         )

#         # train the model
#         for X, _, _ in torch_dataloader:
#             X = self.preprocessor.fill(X)
#             self._train(X, need_test=need_test)

#         self._time_end()


class MultiProcessTrainer(TrainerTemplate):
    """
    This is a multi-process training function. It will create a distributed
    dataloader and train the model using the distributed data.
    """

    def __init__(
        self,
        world_size: int,
        dataloader: Dataloader,
        trainer: TrainerTemplate,
        preprocessor: Preprocessor,
    ):
        self.world_size = world_size
        self.dataloader = dataloader
        self.trainer = trainer
        self.preprocessor = preprocessor

    def _train(self, rank: int, need_test: bool):
        """
        This function is a wrapper function used by one sub-process to train the model.

        Args:
            rank (int): The rank of the sub-process.
            need_test (bool): If this parameter is True, the accurate loss will be calculated during training.
        """

        # initialize the process group
        dist.init_process_group("gloo", rank=rank, world_size=self.world_size)
        os.environ["CUDA_VISIBLE_DEVICES"] = str(rank)

        # using a wrapper dataloader to handle the parameter need_test
        wrapper = DataloaderWrapper(self.dataloader, need_test)

        # create the dataloader using DistributedSampler
        sampler = DistributedSampler(wrapper, num_replicas=self.world_size, rank=rank)
        torch_dataloader = TorchDataLoader(
            wrapper, sampler=sampler, batch_size=self.trainer.batch_size
        )

        logging.info(f"Process {rank} starts with {len(torch_dataloader)} batches.")
        # train the model
        for X, y, outlier_label in torch_dataloader:
            X = self.preprocessor.fill(X)
            self.trainer._train(X, y, outlier_label, need_test=need_test)

    def train(self, need_test=False):
        """
        This function is used to train the model using multi-process.

        Args:
            need_test (bool): If this parameter is True, the accurate loss will be calculated during training.
        """

        self._time_start()

        # start the multi-process training
        torch.multiprocessing.spawn(
            self._train, nprocs=self.world_size, args=(need_test,)
        )

        self._time_end()
