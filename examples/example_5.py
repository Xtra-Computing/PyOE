# import the library PyOE
import pyoe

# load data and detect concept drift
dataloader = pyoe.Dataloader(dataset_name="dataset_experiment_info/beijingPM2.5")
# change the list below with ground truth...
ground_truth_example = [100, 1000, 10000]

print(pyoe.metrics.DriftDelayMetric(dataloader).measure(ground_truth_example))
