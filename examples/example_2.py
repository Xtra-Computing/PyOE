import pyoe

dataloader = pyoe.Dataloader(dataset_name="dataset_experiment_info/beijingPM2.5")
model = pyoe.MlpModel(dataloader=dataloader, device="cuda")
preprocessor = pyoe.Preprocessor(missing_fill="knn2")
trainer = pyoe.IcarlTrainer(dataloader=dataloader, model=model, preprocessor=preprocessor)
trainer.train()
print(f"Average MSELoss: {pyoe.metrics.EffectivenessMetric(dataloader, model).measure()}")
