import PyOE

dataloader = PyOE.Dataloader(dataset_name="dataset_experiment_info/beijingPM2.5")
model = PyOE.MlpModel(dataloader=dataloader, device="cuda")
preprocessor = PyOE.Preprocessor(missing_fill="knn2")
trainer = PyOE.IcarlTrainer(dataloader=dataloader, model=model, preprocessor=preprocessor)
trainer.train()
print(f"Average MSELoss: {PyOE.metrics.EffectivenessMetric(dataloader, model).measure()}")
