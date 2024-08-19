import PyOE
from torch.utils.data import DataLoader as TorchDataLoader

dataloader = PyOE.Dataloader(dataset_name="OD_datasets/AT")
model = PyOE.CluStreamModel(dataloader=dataloader)
preprocessor = PyOE.Preprocessor(missing_fill="knn2")
trainer = PyOE.ClusterTrainer(dataloader=dataloader, model=model, preprocessor=preprocessor, epochs=16)
trainer.train()

torch_dataloader = TorchDataLoader(dataloader, batch_size=32, shuffle=True)
for X, y, _ in torch_dataloader:
    print(X, model.predict_cluster(X))
