import torch
import logging
import torch.distributed as dist
from abc import abstractmethod
from torch.utils.data import DataLoader
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
        y_outlier: torch.Tensor = None,
        need_test: bool = False,
        **kargws,
    ) -> None:
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
        # preprocess the model
        self.model.process_model(lr=self.lr)

    def train(
        self,
        X: torch.Tensor,
        y: torch.Tensor,
        y_outlier: torch.Tensor = None,
        need_test: bool = False,
        **kargws,
    ) -> None:
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


class ClusterTrainer(TrainerTemplate):
    """
    This class is a training wrapper for the model. It will call the
    model's training function to train the model. Actually ClusterModel
    is trained in an online manner, so we don't need to train it in a batch
    manner.
    """

    def __init__(self, dataloader: Dataloader, model: ModelTemplate, **kargws) -> None:
        super().__init__(dataloader, model, **kargws)

    def train(self, X: torch.Tensor, **kwargs) -> None:
        self.model.train_cluster(X)


class MultiProcessTrainer:
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
        # initialize the process group
        dist.init_process_group("gloo", rank=rank, world_size=self.world_size)

        # using a wrapper dataloader to handle the parameter need_test
        wrapper = DataloaderWrapper(self.dataloader, need_test)

        # create the dataloader using DistributedSampler
        sampler = DistributedSampler(wrapper, num_replicas=self.world_size, rank=rank)
        torch_dataloader = DataLoader(
            wrapper, sampler=sampler, batch_size=self.trainer.batch_size
        )

        logging.info(f"Process {rank} starts with {len(torch_dataloader)} batches.")
        # train the model
        for X, y, outlier_label in torch_dataloader:
            X = self.preprocessor.fill(X)
            self.trainer.train(X, y, outlier_label, need_test=need_test)

    def train(self, need_test=False):
        torch.multiprocessing.spawn(
            self._train, nprocs=self.world_size, args=(need_test,)
        )
