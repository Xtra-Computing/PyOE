import pyoe
import matplotlib.pyplot as plt

# prepare dataloader, model, preprocessor and trainer
dataloader = pyoe.TimeSeriesDataloader(
    dataset_name="dataset_experiment_info/5cities/beijing"
)
model = pyoe.ChronosModel(
    dataloader, device="cuda", prediction_length=16, model_path="tiny"
)

# train the model and then predict
X, y = dataloader.get_data(), dataloader.get_target()
model.train_forecast(X, y)
y_pred = model.predict_forecast(X, y)
print(y["target"].values[-20:], y_pred["mean"].to_numpy())

# plot the prediction
y_prev = y["target"].values[-20:]
x_prev = range(len(y_prev))
y_pred = y_pred["mean"].to_numpy()
x_pred = range(len(y_prev), len(y_prev) + len(y_pred))

plt.plot(x_prev, y_prev, label="Previous")
plt.plot(x_pred, y_pred, label="Predicted")
plt.savefig("example_7.png")
