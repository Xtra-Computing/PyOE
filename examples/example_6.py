import pyoe
from torch.utils.data import DataLoader as TorchDataLoader

# prepare dataloader, model, preprocessor and trainer, and then train the model
dataloader = pyoe.Dataloader(dataset_name="OD_datasets/AT")
model = pyoe.CluStreamModel(dataloader=dataloader)
preprocessor = pyoe.Preprocessor(missing_fill="knn2")
trainer = pyoe.ClusterTrainer(dataloader=dataloader, model=model, preprocessor=preprocessor, epochs=16)
trainer.train()

# predict which cluster these data points belong to
torch_dataloader = TorchDataLoader(dataloader, batch_size=32, shuffle=True)
for X, y, _ in torch_dataloader:
    print(X, model.predict_cluster(X))
