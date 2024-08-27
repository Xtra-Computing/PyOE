import pyoe

# prepare dataloader, model, preprocessor and trainer
dataloader = pyoe.TimeSeriesDataloader(
    dataset_name="dataset_experiment_info/5cities/beijing"
)
model = pyoe.ChronosModel(
    dataloader, device="cuda", prediction_length="16", model_path="tiny"
)

# train the model and then predict
X, y = dataloader.get_data(), dataloader.get_target()
model.train_forecast(X, y)
print(model.predict_forecast(X, y), y)
