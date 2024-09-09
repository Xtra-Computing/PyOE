import pyoe
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

"""
Prepare a time series dataset.
"""
# spawn a time series dataset using 'y = x**2 + r' and save it to a file
x = np.linspace(0, 10, 100)
y = x**2 + np.random.normal(0, 1, 100)
# spawn the 'date' column with 1-day intervals
date = pd.date_range(start="2020-01-01", periods=100, freq="D")
df = pd.DataFrame({"date": date, "y": y})
df.to_csv("./data/financial_datasets/dataset_test.csv", index=False)

"""
Train a model on the dataset and predict the next 10 days.
"""
# prepare dataloader, model, preprocessor and trainer
prediction_length = 10
dataloader = pyoe.TimeSeriesDataloader(dataset_name="financial_datasets/dataset_test", predicted_label="y")
model = pyoe.ChronosModel(dataloader, device="cuda", prediction_length=prediction_length, model_path="tiny")

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
plt.savefig("example_9.png")
