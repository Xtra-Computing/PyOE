# OEBench

This is the code for our paper [OEBench: Investigating Open Environment Challenges in Real-World Relational Data Streams](https://arxiv.org/abs/2308.15059).

Relational datasets are widespread in real-world scenarios and are usually delivered in a streaming fashion. This type of data stream can present unique challenges, such as distribution drifts, outliers, emerging classes, and changing features, which have recently been described as [open environment](https://academic.oup.com/nsr/article/9/8/nwac123/6626031) challenges for machine learning. 

We develop an Open Environment Benchmark named **OEBench** to evaluate open environment challenges in relational data streams. Specifically, we investigate 55 real-world streaming datasets and establish that open environment scenarios are indeed widespread in real-world datasets, which presents significant challenges for stream learning algorithms. 

![](figures/flowchart2.png?raw=true)

## Open environment statistics extraction pipeline

This data processing pipeline is specifically designed for open environment learning, providing a comprehensive analysis of datasets, including missing values statistics, anomaly detection, multi-dimensional and one-dimensional drift detection, and concept drift detection. The pipeline is designed to process multiple datasets and provide a detailed report on various metrics.

The whole datasets can be downloaded from https://drive.google.com/file/d/1m7eKbycaEh38OxB7gJibUZ2kNqzVzYMf/view?usp=sharing. 

### Dependencies

This project requires the following Python packages:

- numpy
- pandas
- scikit-learn
- scikit-multiflow
- scipy
- pyod
- Keras
- tensorflow-gpu
- torch
- rtdl
- delu
- lightgbm
- xgboost
- catboost
- copulas
- menelaus (need Python >= 3.9)
- pytorch-tabnet

If `import keras` reports error in ADBench, please replace it with `import tensorflow.keras`.


### Usage

1. Prepare `info.json` and `schema.json` for your datasets and place them in a folder named `dataset_experiment_info` in the same directory as this script. For each dataset, create a subfolder with the dataset's name.

2. If only the statistics for selected datasets are desired, in the script, update the `dataset_prefix_list` variable to include the desired dataset subfolders' names from the `dataset_experiment_info` folder. statistics for all datasets are desired, current code can remain unchanged as all dataset subfolders under the `dataset_experiment_info` folder will be iterated.

3. Run the script, and the pipeline will process each dataset in the specified list, generating various statistics and saving the results in separate CSV files within each dataset's subfolder. An `overall_stats.csv` file will also be generated, containing aggregated statistics for all datasets.


```
python pipeline.py
```

### Adding a new dataset

To add a new dataset to the pipeline, follow these steps:

1. Create a new subfolder within the `dataset_experiment_info` folder, named after the dataset.
2. Place the dataset file (e.g., CSV or Excel) in the `dataset` folder.
3. Create a schema file `schema.json` and an dataset information file `info.json` for the dataset and place it in the same subfolder. 

4. If needed, add the dataset subfolder's name to the `dataset_prefix_list` variable in the script.

For example, to add a dataset called `my_new_dataset`, you should:

- Create a subfolder named `my_new_dataset` inside the `dataset_experiment_info` folder.
- Place the `my_new_dataset.csv` file (or any other supported format) inside the `dataset` subfolder.
- Create a schema file `schema.json` and a information file `info.json` and place them inside the `my_new_dataset` subfolder.
- If needed, manually add 'my_new_dataset' to the `dataset_prefix_list` variable in the script.

Template of `schema.json` of a dataset is as follows:
```json
{
    "numerical": ["num1", "num2"],
    "categorical": ["cat1", "cat2"],
    "target": ["target"],
    "timestamp": ["date", "time"],
    "replace_with_null": ["column_to_be_replaced_by_null"],
    "window size": 0,
    "unnecessary": ["unnecessary1", "unnecessary2"]
}
```

Template of `info.json` of a dataset is as follows:
```json
{
    "schema": "schema.json",
    "data": "dataset/my_new_.csv",
    "task": "classification"
}
```

### Function: run_pipeline

#### Parameters

- `dataset_prefix_list`: A list of dataset path prefixes to process.
- `done`: A list of already processed datasets.

#### Description

The `run_pipeline` function iterates through each dataset path prefix in the `dataset_prefix_list` and processes the dataset. For each dataset, the function performs the following steps:

1. Pre-processes the dataset and extracts its schema.
2. Processes missing values and calculates various missing value statistics.
3. Detect outliers using IForest and ECOD methods.
4. Detect multi-dimensional data drift using HDDDM, kdqTree and KS Statistics.
5. Detect one-dimensional data drift using KS Statistics, HDDDM, kdsTree, CBDB, and PCA-CD methods.
6. Detect concept drift using the PERM, ADWIN, DDM and EDDM method.

After processing each dataset, the function saves the calculated statistics in separate CSV files within each dataset's subfolder. Additionally, the `overall_stats.csv` file is generated, containing aggregated statistics for all datasets.

### Clustering visualization
`cluster.py` visualizes the clusters of datasets according to our calculated statistics for three open environment problems (missing values, drifts, outliers). The purpose is to select representative datasets for further experiments on 10 stream learning algorithms.

![](figures/cluster.png?raw=true)


## Run our benchmark of selected datasets (or other specified datasets)

Please refer to `run.sh` as an example. 

| Parameter                      | Description                                 |
| ----------------------------- | ---------------------------------------- |
| `model` | The model architecture. Options: `mlp`, `tree`. Default = `mlp`. |
| `gbdt` | Whether to use gbdt for tree model. Options: `0`, `1`. Default = `0`. |
| `dataset` | Dataset to use. Options: `selected` or others from the `pipeline.py` (like `dataset_experiment_info/airlines`, etc). Default = `selected`. |
| `alg` | The training algorithm. Options: `naive`, `ewc`, `lwf`, `icarl`, `sea`, `arf`. Default = `naive`. |
| `lr` | Learning rate for MLP models, default = `0.01`. |
| `batch-size` | Batch size for MLP models, default = `64`. |
| `epochs` | Number of training epochs in local window for MLP models, default = `10`. |
| `layers` | The number of layers in MLP models, default = `3`. |
| `reg` | The regularization factor, default = `1`. |
| `buffer` | The number of examplars allowed to store, default = `100`. |
| `ensemble` | The ensemble size for GBDT and SEA, default = `1`. |
| `window-factor` | The factor to multiply the default window size, default = `1`. |
| `missing-fill` | The method to fill missing value. Options: `knn_` (`_` is the number of K in KNN), `regression`, `avg`, `zero`. Default = `knn2`. |
| `logdir` | The path to store the logs, default = `./logs/`. |
| `device` | Specify the device to run the program, default = `cpu`. |
| `init_seed` | The initial seed, default = `0`. |

## Some repos we refer to

- https://github.com/Minqi824/ADBench
- https://github.com/messaoudia/AdaptiveRandomForest
- https://github.com/moskomule/ewc.pytorch

## Citation
If you find this repository useful, please cite our paper:

```
@misc{diao2023oebench,
      title={OEBench: Investigating Open Environment Challenges in Real-World Relational Data Streams}, 
      author={Yiqun Diao and Yutong Yang and Qinbin Li and Bingsheng He and Mian Lu},
      year={2023},
      eprint={2308.15059},
      archivePrefix={arXiv}
}
```
