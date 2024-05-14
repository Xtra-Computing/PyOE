from river import cluster
from river import stream
import time

# Use StreamAD library: https://github.com/Fengrui-Liu/StreamAD
# The first three algorithms have lower scores for outliers, while the others have higher scores for outliers. Thus, the AUC ROC score x should be 1-x for the first three algorithms.
# To test these stream outlier dectors, just run `python outliers.py`.

import numpy as np
import logging
import datetime
from pipeline import *

from sklearn import metrics

if __name__ == "__main__":
    mkdirs("logs/")

    for handler in logging.root.handlers[:]:
        logging.root.removeHandler(handler)

    log_file_name = 'logs/experiment_log-%s.log' % (datetime.datetime.now().strftime("%Y-%m-%d-%H:%M-%S"))

    logging.basicConfig(
        filename=log_file_name,
        format='%(asctime)s %(levelname)-8s %(message)s',
        datefmt='%m-%d %H:%M', level=logging.INFO, filemode='w')

    logger = logging.getLogger()
    logger.setLevel(logging.INFO)

    dataset_prefix_list = ['dataset_experiment_info/room_occupancy', 'dataset_experiment_info/electricity_prices', 'dataset_experiment_info/insects/incremental_reoccurring_balanced', 'dataset_experiment_info/beijing_multisite/shunyi', 'dataset_experiment_info/tetouan']
    done = []

    
    for dataset_path_prefix in dataset_prefix_list:      
        if dataset_path_prefix in done:
            continue
        
        cluster_models = [cluster.CluStream(),
                          cluster.DBSTREAM(),
                          cluster.DenStream(),
                          cluster.STREAMKMeans()
                          ]

        logger.info(dataset_path_prefix)
        try:
            data_path, schema_path, task = schema_parser(dataset_path_prefix)
        except:
            continue
        
        logger.info("start pre-processing")
        
        target_data_nonnull, data_before_onehot, data_onehot_nonnull, original_columns, window_size, row_count, original_column_count, new_columns, new_column_count = data_preprocessing(dataset_path_prefix, data_path, schema_path, task, logger)
        logger.info("preprocessing done")

        data_onehot_nonnull = data_onehot_nonnull.to_numpy().tolist()
        
        for model in cluster_models:
            st = time.time()
            logger.info(model)
            assigned = []
            for x,_ in stream.iter_array(data_onehot_nonnull):
                model.learn_one(x)
                assigned.append(model.predict_one(x))

            X = np.array(data_onehot_nonnull)

            logger.info("time=%f"%(time.time()-st))
            logger.info(metrics.silhouette_score(X, assigned, metric='euclidean'))
            logger.info(metrics.calinski_harabasz_score(X, assigned))
            logger.info(metrics.davies_bouldin_score(X, assigned))
