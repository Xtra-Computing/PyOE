import os
import json
import logging
import numpy as np
import pandas as pd
from sklearn.impute import KNNImputer
from sklearn.preprocessing import OneHotEncoder


def __schema_parser(path: str):
    """
    Parse the schema file and return the data path, schema path and task

    Args:
        path (str): the path of the dataset folder.

    Returns:
        data_path (str): the path of the data file.
        schema_path (str): the path of the schema file.
        task (str): the task of the data.
    """
    info_path = path + "/info.json"
    logging.info(f"Start to read data info from {info_path}")

    with open(info_path, "r") as f:
        info = json.load(f)
        schema_path = path + "/" + info["schema"]
        data_path = info["data"]
        task = info["task"]
    return data_path, schema_path, task


def __data_preprocessing(
    dataset_path_prefix: str,
    data_path: str,
    schema_path: str,
    task: str,
    reload: bool = False,
):
    """
    Preprocess the data and return the target data, data before one hot encoding,
    data after one hot encoding, original columns, window size, row count,
    original column count, new columns, new column count, data one hot.

    Args:
        dataset_path_prefix (str): the prefix of the dataset path.
        data_path (str): the path of the data file.
        schema_path (str): the path of the schema file.
        task (str): the task of the data.

    Returns:
        target_data_nonnull (pd.DataFrame): the target data without null values.
        data_before_onehot (pd.DataFrame): the data before one hot encoding.
        data_one_hot (pd.DataFrame): the data after one hot encoding.
        data_onehot_nonnull (pd.DataFrame): the data after one hot encoding without null values.
        total_columns (pd.Index): the original columns.
        window_size (int): the window size.
        row_count (int): the row count of the data.
        original_column_count (int): the original column count.
        new_columns (pd.Index): the new columns after one hot encoding.
        new_column_count (int): the new column count after one hot encoding.
    """
    # open the schema.json file
    with open(schema_path, "r") as f:
        schema: dict = json.load(f)
        categorical = schema["categorical"]
        target = schema["target"]
        timestamp = schema["timestamp"]
        unnecessary = schema["unnecessary"]
        window_size = schema["window size"]
        replace_with_null = schema.get("replace_with_null", [])
        time_interval = schema.get("time_interval", None)

    # use pandas to read the data (extension name will be judged)
    if data_path.endswith(".csv"):
        data = pd.read_csv(data_path)
    elif data_path.endswith(".xlsx") or data_path.endswith(".xls"):
        data = pd.read_excel(data_path)
    else:
        logging.error(f'The format of data file "{data_path}" is not supported')
        raise ValueError(f"{data_path}: data format not supported")

    total_columns = data.columns
    original_column_count = data.shape[1]
    original_data = data

    # replace some values with null
    logging.info("Start to replace some values with null")
    for value in replace_with_null:
        data = data.replace(value, np.nan)
    data = data.dropna(subset=target)
    logging.info(f"These values have been replaced with null: {replace_with_null}")

    # sort the data by timestamp
    logging.info("Start to sort the data by timestamp")
    if not pd.api.types.is_datetime64_any_dtype(data[timestamp]):
        for col in timestamp:
            data[col] = pd.to_datetime(data[col], errors="ignore")
    data = data.sort_values(timestamp, ascending=True)
    logging.info("Sorting finished")

    original_data = data
    target_data = data[target]
    timestamp_data = pd.to_datetime(data[timestamp], errors="coerce")

    # drop unnecessary columns
    data = data.drop(unnecessary, axis=1)
    data = data.drop(timestamp, axis=1)

    data_one_hot = data.drop(target, axis=1)  # data without target
    data_one_hot = pd.get_dummies(data_one_hot, columns=categorical)  # no target

    new_columns = data_one_hot.columns
    new_column_count = new_columns.shape[0]
    logging.info(f"Columns after one hot encoding: {new_columns.tolist()}")

    # one hot encoding for different tasks
    if task == "classification":
        # factorize the target data
        target_data[target[0]] = pd.factorize(target_data[target[0]])[0]
        # define one hot encoder
        one_hot_encoder = OneHotEncoder(sparse_output=False)
        y_one_hot = one_hot_encoder.fit_transform(target_data[[target[0]]])
        # update the output dimension
        output_dim = y_one_hot.shape[1]
        target = [f"{target[0]}_{i}" for i in range(output_dim)]
        target_data = pd.DataFrame(y_one_hot, columns=target)
    elif task == "regression":
        output_dim = 1
    elif task == "forecasting":
        output_dim = 1
        data_one_hot["timestamp"] = timestamp_data
        if time_interval is None:
            logging.error("Time interval must be specified for forecasting task")
            raise ValueError("Time interval is not specified")
    else:
        logging.error(f"Task {task} is not supported")
        raise ValueError(f"{task}: task not supported")

    logging.info("Start null values processing")
    # check for the existence of the cache file
    whole_data_one_hot_path = dataset_path_prefix + "/onehot_nonnull.csv"
    if os.path.exists(whole_data_one_hot_path) and reload is False:
        logging.info("Loading the processed data from previouly saved file")
        # load the data from the file
        whole_data_one_hot = pd.read_csv(whole_data_one_hot_path, index_col=0)
        if task == "forecasting":
            # forecast task needs timestamp additionally
            whole_data_one_hot["timestamp"] = pd.to_datetime(
                whole_data_one_hot["timestamp"]
            )
        # divide the data into target and data without target
        target_data = whole_data_one_hot[target]
        data_onehot_nonnull = whole_data_one_hot.drop(target, axis=1)
        logging.info("Loading finished")
    elif data_one_hot.isna().values.any():
        # join target columns to the one hot data
        logging.info("The dataset has null values")
        data_one_hot[target] = target_data  # add target to one hot data

        # use KNNImputer to fill the null values
        imp = KNNImputer(n_neighbors=2, weights="uniform")
        # drop the rows with null target
        data_one_hot = data_one_hot.dropna(subset=(target + ["timestamp"]))
        target_data, timestamp_data = data_one_hot[target], data_one_hot["timestamp"]
        data_one_hot = data_one_hot.drop(target + ["timestamp"], axis=1)

        # convert all columns to numeric
        non_numeric_columns = data_one_hot.select_dtypes(exclude=[np.number]).columns
        for col in non_numeric_columns:
            data_one_hot[col] = pd.to_numeric(data_one_hot[col], errors="coerce")

        # inpute the null values
        data_onehot_nonnull = imp.fit_transform(data_one_hot)
        data_onehot_nonnull = pd.DataFrame(data_onehot_nonnull, columns=new_columns)
        assert not data_onehot_nonnull.isnull().values.any()

        # add the timestamp back to the data
        data_onehot_nonnull["timestamp"] = timestamp_data.reset_index(drop=True)
    else:
        logging.info("The dataset has no null values")
        data_onehot_nonnull = data_one_hot
    logging.info("Null values processing finished")

    # add the target back to the data
    concat_data = [
        data_onehot_nonnull.reset_index(drop=True),
        target_data.reset_index(drop=True),
    ]
    whole_data_one_hot = pd.concat(concat_data, axis=1)
    # output the data without null values to a csv file
    whole_data_one_hot.to_csv(whole_data_one_hot_path, mode="w")

    if task == "forecasting":
        logging.info("Start processing data for time series")
        # from concat_data, get the target and the data without target
        whole_data_one_hot.set_index("timestamp", inplace=True)
        whole_data_one_hot = whole_data_one_hot.asfreq(time_interval)
        whole_data_one_hot.reset_index(inplace=True)

        # time series data needs item_id and timestamp (for target)
        target_data = whole_data_one_hot[target + ["timestamp"]]
        target_data.rename(columns={target[0]: "target"}, inplace=True)
        data_onehot_nonnull = whole_data_one_hot.drop(target + ["timestamp"], axis=1)
        logging.info("Time series data processing finished")

    return (
        pd.DataFrame(target_data),
        pd.DataFrame(original_data),
        pd.DataFrame(data_one_hot),
        pd.DataFrame(data_onehot_nonnull),
        total_columns,
        window_size,
        data_onehot_nonnull.shape[0],
        original_column_count,
        new_columns,
        new_column_count,
    )


def load_data(dataset_path: str, prefix: str = "", reload: bool = False):
    """
    Load the data and return the target data, data before one hot encoding,
    data after one hot encoding, window size, output dimension, data one hot,
    and task.

    Args:
        dataset_path (str): the path of the dataset folder.
        prefix (str): the prefix of the dataset path.

    Returns:
        target_data_nonnull (pd.DataFrame): the target data without null values.
        data_before_onehot (pd.DataFrame): the data before one hot encoding.
        data_one_hot (pd.DataFrame): the data after one hot encoding.
        data_onehot_nonnull (pd.DataFrame): the data after one hot encoding without null values.
        window_size (int): the window size.
        output_dim (int): the output dimension.
        task (str): the task of the data.
    """

    logging.info(f"Processing data with prefix {dataset_path}")

    # conbime the prefix and the dataset path
    data_path, schema_path, task = __schema_parser(prefix + dataset_path)
    data_path = prefix + data_path

    logging.info(f"Start data pre-processing for {dataset_path}")
    (
        target_data_nonnull,
        data_before_onehot,
        data_one_hot,
        data_onehot_nonnull,
        original_columns,
        window_size,
        row_count,
        original_column_count,
        new_columns,
        new_column_count,
    ) = __data_preprocessing(
        prefix + dataset_path,
        data_path,
        schema_path,
        task,
        reload,
    )
    logging.info(f"Data preprocessing for {dataset_path} has been done")

    # output the data info
    output_dim = len(target_data_nonnull.columns)

    return (
        target_data_nonnull,
        data_before_onehot,
        data_one_hot,
        data_onehot_nonnull,
        window_size,
        output_dim,
        task,
    )
