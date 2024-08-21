import pyoe
from torch.utils.data import DataLoader as TorchDataLoader

dataloader = pyoe.Dataloader(dataset_name="dataset_experiment_info/beijingPM2.5")
model = pyoe.XStreamDetectorModel(dataloader=dataloader)
torch_dataloader = TorchDataLoader(dataloader, batch_size=10240)
for X, y, _ in torch_dataloader:
    print(model.get_outlier(X), model.get_outlier_with_stream_model(X))
