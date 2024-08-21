import os
import pyoe
import logging

# INFO level logging for more detailed output
logger = logging.getLogger()
logger.setLevel(logging.INFO)


def set_env_vars():
    # Set environment variables for distributed training
    os.environ["MASTER_ADDR"] = "localhost"
    os.environ["MASTER_PORT"] = "12345"
    os.environ["WORLD_SIZE"] = str(world_size)


# if using multi-process for training, all codes with regard to training should
# be put in the `if __name__ == "__main__":` block
if __name__ == "__main__":
    # using pre-prepared dataset and load them into a dataloader
    dataloader = pyoe.Dataloader(dataset_name="dataset_experiment_info/beijingPM2.5")

    # initialize the model, trainer and preprocessor
    model = pyoe.MlpModel(dataloader=dataloader, device="cuda")
    preprocessor = pyoe.Preprocessor(missing_fill="knn2")
    trainer = pyoe.NaiveTrainer(
        dataloader=dataloader, model=model, preprocessor=preprocessor, epochs=16
    )

    # train the model using single process
    # trainer.train()

    # train the model using multiple processes
    world_size = 4
    set_env_vars()
    pyoe.MultiProcessTrainer(world_size, dataloader, trainer, preprocessor).train()

    # using a effective metric to evaluate the model
    print(
        f"Average MSELoss: {pyoe.metrics.EffectivenessMetric(dataloader, model).measure()}"
    )
