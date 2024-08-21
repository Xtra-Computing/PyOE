# Use StreamAD library: https://github.com/Fengrui-Liu/StreamAD
# The first three algorithms have lower scores for outliers, while the others have higher scores for outliers. Thus, the AUC ROC score x should be 1-x for the first three algorithms.
# To test these stream outlier dectors, just run `python outliers.py`.

import numpy as np
import logging
import datetime
from pipeline import *
from streamad.model import (
    xStreamDetector,
    RShashDetector,
    HSTreeDetector,
    LodaDetector,
    RrcfDetector,
)
from sklearn.metrics import roc_auc_score


def outlier_detector_marker(data):

    seed = 0

    model_dict = {
        "ECOD": PYOD,
        "IForest": PYOD,
    }

    anomaly_list = []

    for name, clf in model_dict.items():
        # logger.info(name)
        clf = clf(seed=seed, model_name=name)
        clf = clf.fit(data, [])
        # output predicted anomaly score on testing set
        score = clf.predict_score(data)
        # logger.info(score)
        t = score.mean() + 2 * score.std()
        # logger.info(t)
        anomaly_list.append(np.where(score > t, 1, 0))

    anomaly_index = anomaly_list[0]
    for i in range(1, len(anomaly_list)):
        anomaly_index = anomaly_index * anomaly_list[i]

    # logger.info(sum(anomaly_index)/len(anomaly_index))

    return anomaly_index
    # return outlier_stats_each_window, outlier_stats_overall


if __name__ == "__main__":
    mkdirs("logs/")

    for handler in logging.root.handlers[:]:
        logging.root.removeHandler(handler)

    log_file_name = "logs/experiment_log-%s.log" % (
        datetime.datetime.now().strftime("%Y-%m-%d-%H:%M-%S")
    )

    logging.basicConfig(
        filename=log_file_name,
        format="%(asctime)s %(levelname)-8s %(message)s",
        datefmt="%m-%d %H:%M",
        level=logging.INFO,
        filemode="w",
    )

    logger = logging.getLogger()
    logger.setLevel(logging.INFO)

    dataset_prefix_list = [
        "dataset_experiment_info/room_occupancy",
        "dataset_experiment_info/electricity_prices",
        "dataset_experiment_info/insects/incremental_reoccurring_balanced",
        "dataset_experiment_info/beijing_multisite/shunyi",
        "dataset_experiment_info/tetouan",
    ]
    done = []

    for dataset_path_prefix in dataset_prefix_list:
        if dataset_path_prefix in done:
            continue

        streamad_models = [
            xStreamDetector(depth=10),
            RShashDetector(components_num=10),
            HSTreeDetector(),
            LodaDetector(),
            RrcfDetector(),
        ]

        logger.info(dataset_path_prefix)
        try:
            data_path, schema_path, task = schema_parser(dataset_path_prefix)
        except:
            continue

        logger.info("start pre-processing")

        (
            target_data_nonnull,
            data_before_onehot,
            data_onehot_nonnull,
            original_columns,
            window_size,
            row_count,
            original_column_count,
            new_columns,
            new_column_count,
        ) = data_preprocessing(
            dataset_path_prefix, data_path, schema_path, task, logger
        )
        logger.info("preprocessing done")

        outlier_labels = outlier_detector_marker(data_onehot_nonnull)

        data_onehot_nonnull = data_onehot_nonnull.to_numpy()

        for model in streamad_models:
            scores = []
            for i in range(len(data_onehot_nonnull)):
                score = model.fit_score(data_onehot_nonnull[i])
                scores.append(score)
            # logger.info(scores)
            auc = roc_auc_score(outlier_labels[100:], scores[100:])
            logger.info(auc)
