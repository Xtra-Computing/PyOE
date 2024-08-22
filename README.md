# PyOE

[PyOE](https://github.com/Xtra-Computing/PyOE) is a Machine Learning System inspired by [OEBench](https://github.com/Xtra-Computing/OEBench). With this system, you can train models on our datasets with just a few lines of code. For more details, you can visit our [PyOE website](https://pyoe.xtra.science).

The structure of our system is shown below:

![Structure of PyOE](images/pyoe.png)

## How to Start with PyOE

First, you need to install the dependencies required by PyOE. You can do this by running the following commands:

```shell
pip install streamad==0.3.1
pip install pyoe
```

We have some examples in the examples folder. You can copy one of these examples to the parent directory of this README file and try running them. If they run successfully, it means the installation is complete.

## Five Main Tasks of PyOE System

Our PyOE system currently supports 5 types of tasks: regression analysis, classification, outlier detection, and concept drift detection. In the following, we will provide examples of code for each of these 5 tasks.

### Regression

To perform *Open Environment* regression analysis using PyOE, follow the steps mentioned earlier: first, load the dataset, then select a model that supports this task, choose a method for handling missing values, and finally, pass all these as parameters to the trainer. By calling the ```train``` method, the model will be trained automatically.

After that, if you want to make regression predictions, you can get the model using ```model.get_net()```, and then pass the corresponding data to make predictions.

Here is a complete example:

```python
# import the library PyOE
import pyoe

# choose the targeted dataset, model, preprocessor, and trainer
dataloader = pyoe.Dataloader(dataset_name="dataset_experiment_info/beijingPM2.5")
model = pyoe.MlpModel(dataloader=dataloader, device="cuda")
preprocessor = pyoe.Preprocessor(missing_fill="knn2")
trainer = pyoe.NaiveTrainer(dataloader=dataloader, model=model, preprocessor=preprocessor, epochs=16)

# get the trained net
net = model.get_net()
# predict here...
```

### Classification

The classification task is almost identical to the regression task, except that some different loss functions and models are used internally, but this does not affect the usage on the user side. Note that in classification tasks, we use OneHot encoding for the prediction results. Therefore, depending on the number of classes, each prediction result should be a 0-1 vector of the corresponding dimension. Here is a simple example:

```python
# import the library PyOE
import pyoe

# choose the targeted dataset, model, preprocessor, and trainer
dataloader = pyoe.Dataloader(dataset_name="dataset_experiment_info/room_occupancy")
model = pyoe.MlpModel(dataloader=dataloader, device="cuda")
preprocessor = pyoe.Preprocessor(missing_fill="knn2")
trainer = pyoe.NaiveTrainer(dataloader=dataloader, model=model, preprocessor=preprocessor, epochs=1024)

# get the trained net
net = model.get_net()
# predict here...
```

### Clustering

Our system supports clustering analysis of data points. The task is done by training a clustering stream model on the data points and then using the model to make predictions after training is complete. The model will return integer label values representing which cluster the current data point belongs to. Below is a sample code:

```python
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
```

### Outlier Detection

Our PyOE also supports outlier analysis of data. Since outlier detection is solely dependent on the data, we directly call the ```get_outlier``` method of stream models. This method is nearly identical to the outlier detection methods in OEBench, using PyOD's ECOD and IForest models. We consider a point to be an outlier if and only if both of these models agree.

In real-world scenarios, many data are streaming data (e.g., time series data) and need to be updated online. In such cases, streaming algorithms are useful. Therefore, in the streaming models we provide, there is a ```get_model_score``` method that can be used to get the score assigned to data points by the streaming model. By comparing this score with the previous global algorithm or ground truth, you can determine the effectiveness of the streaming algorithm.

Here is an example:

```python
import pyoe
from torch.utils.data import DataLoader as TorchDataLoader

dataloader = pyoe.Dataloader(dataset_name="dataset_experiment_info/beijingPM2.5")
model = pyoe.XStreamDetectorModel(dataloader=dataloader)
# use TorchDataLoader to enumerate X and y
torch_dataloader = TorchDataLoader(dataloader, batch_size=10240)
for X, y, _ in torch_dataloader:
    print(model.get_outlier(X), model.get_outlier_with_stream_model(X))
```

### Concept Drift Detection

Concept drift detection is mainly divided into two scenarios. The first scenario is when the ground truth is unknown. In that case, we use the code below to obtain the detected drift points:

```python
# import the library PyOE
import pyoe

# load data and detect concept drift
dataloader = pyoe.Dataloader(dataset_name="dataset_experiment_info/beijingPM2.5")
print(pyoe.metrics.DriftDelayMetric(dataloader).measure())
```

It will print a list containing all the detected concept drift points. It should be noted that ```pyoe.metrics.DriftDelayMetric``` also contains many parameters that can be used to define the model, sensitivity of detection, and so on.

The second scenario is when the ground truth is known. Use the following code to measure the *Average Concept Drift Delay*:

```python
# import the library PyOE
import pyoe

# load data and detect concept drift
dataloader = pyoe.Dataloader(dataset_name="dataset_experiment_info/beijingPM2.5")
# change the list below with ground truth...
ground_truth_example = [100, 1000, 10000]

print(pyoe.metrics.DriftDelayMetric(dataloader).measure())
```

It will print a floating-point number representing the *Average Concept Drift Delay*.

## Repos We Refer To

- https://github.com/Minqi824/ADBench
- https://github.com/messaoudia/AdaptiveRandomForest
- https://github.com/moskomule/ewc.pytorch

## Citation
If you find this repository useful, please cite our paper:

```
@article{diao2024oebench,
      title={OEBench: Investigating Open Environment Challenges in Real-World Relational Data Streams}, 
      author={Diao, Yiqun and Yang, Yutong and Li, Qinbin and He, Bingsheng and Lu, Mian},
      journal={Proceedings of the VLDB Endowment},
      volume={17},
      number={6},
      pages={1283--1296},
      year={2024},
      publisher={VLDB Endowment}
}
```
