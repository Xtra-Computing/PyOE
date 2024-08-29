import pyoe
import matplotlib.pyplot as plt

# prepare dataloader, model, preprocessor and trainer
prediction_length = 16
dataloader = pyoe.TimeSeriesDataloader(dataset_name="financial_datasets/AAA")
model = pyoe.ChronosModel(
    dataloader, device="cuda", prediction_length=prediction_length, model_path="tiny"
)

# train the model
X, y = dataloader.get_data(), dataloader.get_target()
model.train_forecast(X, y)

# split the data and predict
X_front, y_front = X[:-prediction_length], y[:-prediction_length]
y_pred = model.predict_forecast(X_front, y_front)
print(y["target"].values[-prediction_length:], y_pred["mean"].to_numpy())

# plot the prediction
y_prev = y["target"].values[-prediction_length * 2 :]
x_prev = range(len(y_prev))
y_pred = y_pred["mean"].to_numpy()
x_pred = range(prediction_length, prediction_length + len(y_pred))

plt.plot(x_prev, y_prev, label="Previous")
plt.plot(x_pred, y_pred, label="Predicted")
plt.savefig("example_7.png")
