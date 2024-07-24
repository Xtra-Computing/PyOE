import os
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.experimental import enable_iterative_imputer
from sklearn.impute import IterativeImputer, KNNImputer
from sklearn.decomposition import PCA
from sklearn.preprocessing import OneHotEncoder
from sklearn.preprocessing import StandardScaler
from sklearn.manifold import TSNE
from sklearn import manifold
import matplotlib.pyplot as plt
import numpy as np
from sklearn.datasets import make_blobs
from sklearn.cluster import MeanShift, estimate_bandwidth
import csv
import json
import matplotlib.pyplot as plt
from sklearn.naive_bayes import GaussianNB
from sklearn.linear_model import SGDClassifier
from sklearn import svm
from sklearn.base import clone
from menelaus.concept_drift import LinearFourRates, ADWINAccuracy, DDM, EDDM, STEPD, MD3
from menelaus.data_drift import HDDDM, KdqTreeBatch, CDBD, PCACD
from menelaus.datasets import fetch_circle_data, fetch_rainfall_data
from menelaus import *
from menelaus.ensemble.election import SimpleMajorityElection
from menelaus.ensemble.ensemble import BatchEnsemble
from statistics import mean
from scipy import stats
import logging
import datetime
import os

from skmultiflow.data import SEAGenerator, HyperplaneGenerator, STAGGERGenerator, RandomRBFGeneratorDrift, LEDGeneratorDrift, WaveformGenerator

def mkdirs(dirpath):
    try:
        os.makedirs(dirpath)
    except Exception as _:
        pass

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

def schema_parser(path):
    info_path = path+'/info.json'
    print(info_path)
    with open(info_path, 'r') as f:
        info = json.load(f)
        schema_path = path+"/"+info["schema"]
        data_path = info["data"]
        task = info["task"]
    return data_path, schema_path, task

def data_preprocessing(dataset_path_prefix, data_path, schema_path, task, delete_null_target=False, return_info=False):
    with open(schema_path, 'r') as f:
        schema = json.load(f)
        # numerical = schema["numerical"]
        categorical = schema["categorical"]
        target = schema["target"]
        timestamp = schema["timestamp"]
        unnecessary = schema["unnecessary"]
        window_size = schema["window size"]
        replace_with_null = []
        if "replace_with_null" in schema:
            replace_with_null = schema["replace_with_null"]
                
    try:
        data = pd.read_csv(data_path)
    except:
        data = pd.read_excel(data_path)
        
    total_columns = data.columns
    row_count = data.shape[0]
    original_column_count = data.shape[1]
    original_data = data
    
    print("not replaced yet")
    for value in replace_with_null:
        print(value)
        data = data.replace(value, np.nan)
        print("done")
    print("replaced with null")

    data = data.dropna(subset=target)
    
    print("start sorting")
    try:
        data = data.sort_values(timestamp, ascending=True)
    except:
        try:
            data[timestamp] = pd.to_datetime(data[timestamp])
        except:
            pass
    print("finished sorting")
    
    original_data = data
    
    # try:
    #     data = data.sort_values(timestamp, ascending=True)
    # except:
    #     for col in timestamp:
    #         data[col] = pd.to_datetime(data[col])
    #     data = data.sort_values(timestamp, ascending=True)
    data_before_onehot = data
    #print(data)
    target_data = data[target]
    print(target_data.isnull().values.any())
    if task == "classification":
        #print(target_data[target[0]])
        target_data[target[0]] = pd.factorize(target_data[target[0]])[0]
        #print(target_data)
        output_dim = max(target_data[target[0]])+1
    else:
        output_dim = 1
    
    data = data.drop(unnecessary, axis=1)
    data = data.drop(timestamp, axis=1)
    
    data_temp = data
    data_temp=data_temp.drop(target, axis=1)#data without target
    
    data_one_hot = pd.get_dummies(data_temp, columns=categorical)#no target
    
    new_columns = data_one_hot.columns
    print(new_columns)
    new_column_count = new_columns.shape[0]
    if delete_null_target:
        return pd.DataFrame(data_one_hot), pd.DataFrame(target_data), window_size, task, new_column_count, output_dim
    
    #try:
    #    data_onehot_nonnull_path = dataset_path_prefix+'/onehot_nonnull.csv'
    #    data_onehot_nonnull = pd.read_csv(data_onehot_nonnull_path)
    #except:
    if True:
        if(data.isnull().values.any() or data.isna().values.any()):
            print("has null")
            temp_columns = []
            for col in new_columns:
                temp_columns.append(col)
            for col in target:
                temp_columns.append(col)
            # print(temp_columns)
            data_one_hot[target] = target_data#add target to one hot data
            # print(data_one_hot.info())
            
            imp = KNNImputer(n_neighbors=2, weights="uniform")
            try:
                try:
                    data_onehot_nonnull = imp.fit_transform(data_one_hot)
                    data_onehot_nonnull = pd.DataFrame(data_onehot_nonnull.values, columns=temp_columns)
                except:
                    for col in new_columns:
                        data_one_hot[col] = pd.to_numeric(data_one_hot[col], errors='coerce')
                    data_onehot_nonnull = imp.fit_transform(data_one_hot)
                    data_onehot_nonnull = pd.DataFrame(data_onehot_nonnull.values, columns=temp_columns)
            except:
                data_one_hot = data_one_hot.dropna(subset=target)
                target_data = data_one_hot[target]
                data_one_hot = data_one_hot.drop(target, axis=1)
                print(data_one_hot.columns)
                try:
                    print("all numeric")
                    data_onehot_nonnull = imp.fit_transform(data_one_hot)
                    data_onehot_nonnull = pd.DataFrame(data_onehot_nonnull)
                    print("done")
                    
                    # data_onehot_nonnull[target] = target_data
                    data_onehot_nonnull = pd.concat([data_onehot_nonnull.reset_index(drop=True), target_data.reset_index(drop=True)], axis=1)
                    print(data_onehot_nonnull.isnull().values.any())
                except:
                    print("non numeric")
                    for col in new_columns:
                        data_one_hot[col] = pd.to_numeric(data_one_hot[col], errors='coerce')
                    print("start inputing")
                    data_onehot_nonnull = imp.fit_transform(data_one_hot)
                    data_onehot_nonnull = pd.DataFrame(data_onehot_nonnull)
                    print("done")
                    
                    data_onehot_nonnull = pd.concat([data_onehot_nonnull.reset_index(drop=True), target_data.reset_index(drop=True)], axis=1)
                    print(data_onehot_nonnull.isnull().values.any())
            # temp_columns = pd.DataFrame(new_columns).append(target)

        else:
            data_onehot_nonnull = data_one_hot
            data_onehot_nonnull[target] = target_data

    print(data_onehot_nonnull.isnull().values.any())
    
    data_onehot_nonnull_path = dataset_path_prefix+'/onehot_nonnull.csv'
    data_onehot_nonnull.to_csv(data_onehot_nonnull_path, mode='w')  
    
    target_data = data_onehot_nonnull[target]
    target_data = pd.DataFrame(target_data, columns=target)
    
    data_onehot_nonnull=data_onehot_nonnull.drop(target, axis=1)
    data_onehot_nonnull.columns = new_columns
    print(data_onehot_nonnull.isnull().values.any())
    
    row_count = data_onehot_nonnull.shape[0]
    
    # imp = KNNImputer(n_neighbors=2, weights="uniform")
    
    # if(target_data.isnull().values.any()):
    #     target_data = imp.fit_transform(target_data)
    #     target_data = pd.DataFrame(target_data)
    #     target_data.head()
    
    # print("test")
    # if(data_one_hot.isnull().values.any()):
    #     print("has null")
    #     data_onehot_nonnull = imp.fit_transform(data_one_hot)
    #     print("processed")
    #     data_onehot_nonnull = pd.DataFrame(data_onehot_nonnull, columns=new_columns) 
    # else:
    #     print("no null")
    #     data_onehot_nonnull = data_one_hot
    
    if return_info:
        return pd.DataFrame(target_data), pd.DataFrame(original_data), pd.DataFrame(data_onehot_nonnull), total_columns, window_size, row_count, original_column_count, new_columns, new_column_count, pd.DataFrame(data_one_hot)
    else:
        return pd.DataFrame(target_data), pd.DataFrame(original_data), pd.DataFrame(data_onehot_nonnull), total_columns, window_size, row_count, original_column_count, new_columns, new_column_count
       


def missing_value_processor(data, window_size, total_columns, window_count, row_count):
    missing_value_stats_by_window = pd.DataFrame(index=list(range(window_count)), 
                                                 columns=[
                                                     "columns_with_null", "col_null_dict", "empty_cells_num", "ave_null_columns", 
                                                     "ave_null_columns_ratio", "missing_value_ratio"])
    column_count =len(total_columns) 
    ave_null_columns_ratio_list = []
    missing_value_ratio_list = []
    rows_with_missing_values_ratio_list = []
    for n in range(window_count):
        
        rows_with_missing_values = window_size - len(data[n*window_size:(n+1)*window_size].dropna())
        current_rows_with_missing_values_ratio = rows_with_missing_values/window_size
        rows_with_missing_values_ratio_list.append(current_rows_with_missing_values_ratio)
        
        columns_with_null = []
        col_null_dict = {}
        empty_cells_num = 0
        ave_null_columns = 0        
        for col in total_columns:
            if(data[n*window_size:(n+1)*window_size][col].isnull().values.any()):
                columns_with_null.append(col)
                current = data[n*window_size:(n+1)*window_size][col].isnull().sum()
                col_null_dict[col] = current
                empty_cells_num = empty_cells_num+current
        if(len(columns_with_null)!=0):
            ave_null_columns = empty_cells_num/window_size
            ave_null_columns_ratio = ave_null_columns/column_count
            missing_value_ratio = empty_cells_num/(column_count*window_size)
            
            missing_value_stats_by_window.at[n, "columns_with_null"] = columns_with_null
            missing_value_stats_by_window.at[n, "col_null_dict"] = col_null_dict
            missing_value_stats_by_window.at[n, "empty_cells_num"] = empty_cells_num
            missing_value_stats_by_window.at[n, "ave_null_columns"] = ave_null_columns
            missing_value_stats_by_window.at[n, "ave_null_columns_ratio"] = ave_null_columns_ratio
            missing_value_stats_by_window.at[n, "missing_value_ratio"] = missing_value_ratio
            
            ave_null_columns_ratio_list.append(ave_null_columns_ratio)
            missing_value_ratio_list.append(missing_value_ratio)
            # 1 2 3 4
            # 0 n 0 0
            # 0 n n 0
            # 0 n n n 
    try:
        ave_ave_null_columns_ratio = mean(ave_null_columns_ratio_list)
        max_ave_null_columns_ratio = max(ave_null_columns_ratio_list)
        ave_missing_value_ratio = mean(missing_value_ratio_list)
        max_missing_value_ratio = max(missing_value_ratio_list)
    except:
        ave_ave_null_columns_ratio = None
        max_ave_null_columns_ratio = None
        ave_missing_value_ratio = None
        max_missing_value_ratio = None
    
    ave_rows_with_missing_values_ratio_per_window = mean(rows_with_missing_values_ratio_list)
    max_rows_with_missing_values_ratio_per_window = max(rows_with_missing_values_ratio_list)
    
    total_rows_with_missing_value = len(data)-len(data.dropna())
    total_rows_with_missing_values_ratio = total_rows_with_missing_value/len(data)
        

    missing_value_stats_overall = pd.DataFrame(index=["columns_with_null", "col_null_dict", "empty_cells_num", "ave_null_columns"],
                                               columns=["overall"])
    print("overall stats")
    columns_with_null = []
    col_null_dict = {}
    empty_cells_num = 0
    ave_null_columns = 0
    for col in total_columns:
        if(data[col].isnull().values.any()):
            columns_with_null.append(col)
            current = data[col].isnull().sum()
            col_null_dict[col] = current
            empty_cells_num +=current
            
    ave_null_columns = empty_cells_num/row_count
    
    missing_value_stats_overall.at["columns_with_null","overall"] = columns_with_null
    missing_value_stats_overall.at["col_null_dict","overall"] = col_null_dict
    missing_value_stats_overall.at["empty_cells_num","overall"] = empty_cells_num
    missing_value_stats_overall.at["ave_null_columns","overall"] = ave_null_columns  
    
    overall_ave_null_columns_ratio = ave_null_columns/column_count
    overall_missing_value_ratio = empty_cells_num/(row_count*column_count)
    
    
    return ave_rows_with_missing_values_ratio_per_window, max_rows_with_missing_values_ratio_per_window, total_rows_with_missing_values_ratio, ave_ave_null_columns_ratio, max_ave_null_columns_ratio, ave_missing_value_ratio, max_missing_value_ratio, overall_ave_null_columns_ratio, overall_missing_value_ratio     
    
    
    # return missing_value_stats_overall, missing_value_stats_by_window
    
    
def data_drift_detector_multi_dimensional(data, window_size, window_num):
    #print(data)
    detectors_dict = {
        'kdq': KdqTreeBatch(bootstrap_samples=500),
        'hdddm': HDDDM(),
        'cdbd': CDBD(),
        'pcacd': PCACD(window_size=window_size)
    }

    training_size = window_size

    reference = data.iloc[0:training_size]

    # ensemble.set_reference(reference)
    #print(f"Batch #{0} | Ensemble reference set")

    drift_detector_list = ['hdddm','kdq']

    drift_stats = pd.DataFrame([])
    
    hdddm_drift_percentage = 0
    kdq_drift_percentage = 0
    
    hdddm_warning_percentage = 0
    kdq_warning_percentage = 0
    for algo in drift_detector_list:
        #print(algo)
        drift_count = 0
        drift_percentage = 0
        detected_drift = []
        
        warning_count = 0
        warning_percentage = 0
        detected_warning = []
        
        detectors = {algo : detectors_dict[algo]}
        election = SimpleMajorityElection()
        ensemble = BatchEnsemble(detectors, election)
        #print("start reference")
        #print(reference)
        ensemble.set_reference(reference)
        #print("start updating")
        #print(window_num)
        for n in range(1, window_num):
            #print(n)
            ensemble.update(data[n*window_size:(n+1)*window_size])
            detected_drift.append(ensemble.drift_state)
            if(ensemble.drift_state == 'drift'):
                drift_count +=1
            elif(ensemble.drift_state == 'warning'):
                warning_count+=1
        
        #print(detected_drift)
        
        drift_percentage = drift_count/(window_num-1)
        warning_percentage = warning_count/(window_num-1)
        drift_stats[algo] = [drift_percentage]
        if algo=="hdddm":
            hdddm_drift_percentage = drift_percentage
            hdddm_warning_percentage = warning_percentage
        else:
            kdq_drift_percentage = drift_percentage
            kdq_warning_percentage = warning_percentage
        
        #print(drift_percentage)
    ave_drift_percentage = (hdddm_drift_percentage+kdq_drift_percentage)/2
    ave_warning_percentage = (hdddm_warning_percentage+kdq_warning_percentage)/2
    
    print(drift_stats)
    return hdddm_drift_percentage, kdq_drift_percentage, ave_drift_percentage, hdddm_warning_percentage, kdq_warning_percentage, ave_warning_percentage
    # return drift_stats

def data_drift_detector_one_dimensional(data, window_size, window_num, columns):
    detectors_dict = {
        'kdq': KdqTreeBatch(bootstrap_samples=500),
        'hdddm': HDDDM(),
        'cdbd': CDBD(),
    }
    training_size = window_size
 
    drift_stats_each_column = pd.DataFrame(columns=columns, index=["hdddm", "kdq", "cdbd"])
    
    drift_detector_list = ['hdddm','kdq', 'cdbd']
    
    #print(data)
    hdddm_ave_drift_percentage = 0
    hdddm_max_drift_percentage = 0
    kdq_ave_drift_percentage = 0
    kdq_max_drift_percentage = 0
    cbdb_ave_drift_percentage = 0
    cbdb_max_drift_percentage = 0
    ks_ave_drift_percentage = 0
    ks_max_drift_percentage = 0
    
    ave_drift_percentage = 0
    max_drift_percentage = 0
    
    hdddm_ave_warning_percentage = 0
    hdddm_max_warning_percentage = 0
    kdq_ave_warning_percentage = 0
    kdq_max_warning_percentage = 0
    cbdb_ave_warning_percentage = 0
    cbdb_max_warning_percentage = 0
    ks_ave_warning_percentage = 0
    ks_max_warning_percentage = 0
    
    ks_drift_percentage_list = []
    ks_warning_percentage_list = []
    for col in columns:
        reference = data[col].iloc[0:training_size]
        
        pvalue = 0.05
            
        ks_detected_drift = []
        ks_drift_count = 0
        ks_drift_percentage = 0
            
        ks_warning_count = 0
        ks_warning_percentage = 0
        for n in range(1, window_num):
            current_data = data[col].iloc[n*window_size:(n+1)*window_size]
            test = stats.ks_2samp(reference, current_data)
            if test[1] < pvalue:#drift
                ks_drift_count+=1
                ks_detected_drift.append('drift')
            elif test[1] < pvalue*2:#warning
                ks_warning_count+=1
                ks_detected_drift.append('warning')
            else:
                ks_detected_drift.append(None)
        ks_drift_percentage = ks_drift_count/(window_num-1)
        ks_drift_percentage_list.append(ks_drift_percentage)
        ks_warning_percentage = ks_warning_count/(window_num-1)
        ks_warning_percentage_list.append(ks_warning_percentage)
    
    ks_ave_drift_percentage = mean(ks_drift_percentage_list)
    ks_max_drift_percentage = max(ks_drift_percentage_list)
    
    print("ks warning")
    print(ks_warning_percentage_list)
    ks_ave_warning_percentage = mean(ks_warning_percentage_list)
    ks_max_warning_percentage = max(ks_warning_percentage_list)
    print(ks_ave_warning_percentage)
    print(ks_max_warning_percentage)
    
    
    ave_warning_percentage = 0
    max_warning_percentage = 0
    for algo in drift_detector_list:
        current_detector = detectors_dict[algo] 
        drift_percentage_list = []
        warning_percentage_list = []
        for col in columns:
            #print("column: "+col)
            reference = data[col].iloc[0:training_size]
            #print(reference)
            current_detector.set_reference(reference) 
            
            detected_drift = []
            drift_count = 0
            drift_percentage = 0
            
            warning_count = 0
            warning_percentage = 0
            for n in range(1, window_num):
                #print(n)
                curr = data[col].iloc[n*window_size:(n+1)*window_size]
                #print(curr)
                try:
                    current_detector.update(curr)
                    detected_drift.append(current_detector.drift_state)
                    if(current_detector.drift_state == 'drift'):
                        drift_count +=1
                    elif(current_detector.drift_state == 'warning'):
                        warning_count+=1
                except:
                    continue
            drift_percentage = drift_count/(window_num-1)
            drift_stats_each_column.loc[algo][col] = drift_percentage
            drift_percentage_list.append(drift_percentage)
            
            warning_percentage = warning_count/(window_num-1)
            warning_percentage_list.append(warning_percentage)
        
        ave = mean(drift_percentage_list)
        maximum = max(drift_percentage_list)
        
        #print(algo)
        #print("warning")
        #print(warning_percentage_list)
        warning_ave = mean(warning_percentage_list)
        warning_maximum = max(warning_percentage_list)
        #print(warning_ave)
        #print(warning_maximum)
        
        if algo == "hdddm":
            hdddm_ave_drift_percentage = ave
            hdddm_max_drift_percentage = maximum
            
            hdddm_ave_warning_percentage = warning_ave
            hdddm_max_warning_percentage = warning_maximum
        elif algo == "kdq":
            kdq_ave_drift_percentage = ave
            kdq_max_drift_percentage = maximum
            
            kdq_ave_warning_percentage = warning_ave
            kdq_max_warning_percentage = warning_maximum
        else:
            cbdb_ave_drift_percentage = ave
            cbdb_max_drift_percentage = maximum
            
            cbdb_ave_warning_percentage = warning_ave
            cbdb_max_warning_percentage = warning_maximum
            
    ave_drift_percentage = mean([ks_ave_drift_percentage, hdddm_ave_drift_percentage, kdq_ave_drift_percentage, cbdb_ave_drift_percentage])
    max_drift_percentage = max([ks_max_drift_percentage, hdddm_ave_drift_percentage, kdq_ave_drift_percentage, cbdb_ave_drift_percentage])
    
    ave_warning_percentage = mean([ks_ave_warning_percentage, hdddm_ave_warning_percentage, kdq_ave_warning_percentage, cbdb_ave_warning_percentage])
    max_warning_percentage = max([ks_max_warning_percentage, hdddm_ave_warning_percentage, kdq_ave_warning_percentage, cbdb_ave_warning_percentage]) 
    
    # status = pd.DataFrame(columns=["index", "var1", "var2", "drift_detected"])
    
    pca_cd = PCACD(window_size=window_size, divergence_metric="intersection")  
    pca = PCA(n_components=2)
    pca_features = pca.fit_transform(data)
    pca_df = pd.DataFrame(data=pca_features, columns=['var1', 'var2'])
    print("pca df:")
    print(pca_df)
    
    drift_stats_pca = pd.DataFrame(columns=['var1', 'var2'])
    
    var1_drift_percentage = 0
    var2_drift_percentage = 0
    pca_ave_drift_percentage = 0
    pca_max_drift_percentage = 0
    
    var1_warning_percentage = 0
    var2_warning_percentage = 0
    pca_ave_warning_percentage = 0
    pca_max_warning_percentage = 0
    
    for col in ['var1', 'var2']:
        detected_drift = []
        drift_count = 0
        drift_percentage = 0
        for n in range(0, window_num):
            #print(n)
            # print(pca_df[n*window_size:(n+1)*window_size, col])
            pca_cd.update(pca_df[col].iloc[n*window_size:(n+1)*window_size])
            detected_drift.append(pca_cd.drift_state)
            if(pca_cd.drift_state == 'drift'):
                drift_count +=1
            elif(pca_cd.drift_state == 'warning'):
                warning_count +=1
        drift_percentage = drift_count/(window_num)
        if col =="var1":
            var1_drift_percentage = drift_percentage
            var1_warning_percentage = warning_percentage
        else:
            var2_drift_percentage = drift_percentage
            var2_warning_percentage = warning_percentage
        drift_stats_pca.loc[col] = drift_percentage
    pca_ave_drift_percentage = (var1_drift_percentage+var2_drift_percentage)/2
    pca_max_drift_percentage = max([var1_drift_percentage, var2_drift_percentage])    
    
    pca_ave_warning_percentage = (var1_warning_percentage+var2_warning_percentage)/2
    pca_max_warning_percentage = max([var1_warning_percentage, var2_warning_percentage]) 
    
        
    return ks_ave_drift_percentage, ks_max_drift_percentage, hdddm_ave_drift_percentage, hdddm_max_drift_percentage, kdq_ave_drift_percentage, kdq_max_drift_percentage, cbdb_ave_drift_percentage, cbdb_max_drift_percentage, pca_ave_drift_percentage, pca_max_drift_percentage, ave_drift_percentage, max_drift_percentage, hdddm_ave_warning_percentage, hdddm_max_warning_percentage, kdq_ave_warning_percentage, kdq_max_warning_percentage, cbdb_ave_warning_percentage, cbdb_max_warning_percentage, pca_ave_warning_percentage, pca_max_warning_percentage, ks_ave_warning_percentage, ks_max_warning_percentage, ave_warning_percentage, max_warning_percentage
    # return drift_stats_each_column, drift_stats_pca
    
    
from menelaus.ensemble import BatchEnsemble, StreamingEnsemble
from menelaus.concept_drift import LinearFourRates, ADWINAccuracy, DDM, EDDM, STEPD, MD3

from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import log_loss
from sklearn.metrics import mean_squared_error
from sklearn.metrics import accuracy_score
from sklearn.metrics import r2_score
from sklearn.linear_model import LinearRegression
from sklearn.linear_model import SGDRegressor, SGDClassifier

def PERM(task, data, target, window_size, window_count, number_of_permutation, sensitivity, significance_rate):
    t = 0
    k = window_size
    # drift_percentage = 0
    # drift_detected_list = []
    drift_count = 0
    while k < len(data):
        reference = data[t:k]
        reference_target = target[t:k]
        current = data[k:k+window_size]
        current_target = target[k:k+window_size]
        detected = perm_detect(task, reference, current, reference_target, current_target, number_of_permutation, sensitivity, significance_rate)
        if detected==True:
            drift_count+=1
            t = k
        k = k+window_size
    
    drift_percentage = drift_count/(window_count-1)
    print("perm")
    print(drift_percentage)
    
    return drift_percentage


from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import log_loss
from sklearn.metrics import mean_squared_error
from sklearn.metrics import accuracy_score
from sklearn.metrics import r2_score, mean_absolute_error
from sklearn.linear_model import SGDRegressor, SGDClassifier, LinearRegression

def perm_detect(task, reference, current, reference_target, current_target, number_of_permutation, sensitivity, significance_rate):
    reference_size = len(reference)
    current_size = len(current)
    total_size = reference_size+current_size
    data = pd.concat([reference, current])
    target = pd.concat([reference_target, current_target])
    test_size_ratio = current_size / total_size
    
    ordered_loss=1
    if(task=="regression"):
        rf = LinearRegression()
        try:
            rf.fit(reference, reference_target)
        except:
            return False
        ordered_y_pred = rf.predict(current)
        ordered_y_pred = pd.DataFrame(ordered_y_pred)
        # print("reference")
        # print(reference)
        
        # print("reference target")
        # print(reference_target)
        
        # print("current")
        # print(current)
        
        # print("current target")
        # print(current_target)
        
        # print("predicted")
        # print(ordered_y_pred)
        # print("regression")
        ordered_loss = mean_absolute_error(np.array(current_target), np.array(ordered_y_pred))
        #print(ordered_loss)
        y_range = np.max(np.array(current_target)) - np.min(np.array(current_target))
        ordered_loss = ordered_loss/y_range
    else:
        #print("reference")
        #print(reference)
        
        #print("reference target")
        #print(reference_target)
        # rf = SGDClassifier(random_state = 42)
        try:
            rf.fit(reference, reference_target)
        except:
            return False
        rf.fit(reference, reference_target)
        
        ordered_y_pred = rf.predict(current)
        ordered_y_pred = pd.DataFrame(ordered_y_pred)
        # ordered_y_pred = ordered_y_pred.reshape()
        #print("classification")
        # ordered_loss = log_loss(current_target, ordered_y_pred)
        ordered_loss = 1-accuracy_score(current_target, ordered_y_pred)
    
    #print("ordered loss")
    #print(ordered_loss)

    count = 0
    for i in range(number_of_permutation):
        X_train, X_test, y_train, y_test = train_test_split(data, target, test_size = test_size_ratio, random_state=None, shuffle=True)
        
        if(task=="regression"):
            rf = LinearRegression()
            rf.fit(X_train, y_train)
            current_y_pred = rf.predict(X_test)
            #print("regression")
            current_y_pred=pd.DataFrame(current_y_pred)
            current_loss = mean_absolute_error(np.array(y_test), np.array(current_y_pred))
            y_range = np.max(np.array(y_test)) - np.min(np.array(y_test))
            current_loss = current_loss/y_range
            #print(current_loss)
        else:
            rf = SGDClassifier(random_state = 42)
            rf.fit(X_train, y_train)
            try:
                rf.fit(X_train, y_train)
            except:
                return False
            current_y_pred = rf.predict(X_test)
            current_y_pred=pd.DataFrame(current_y_pred)
            # current_y_pred=current_y_pred.reshape(-1,1)
            #print("classification")
            # current_loss = log_loss(y_test, current_y_pred)
            current_loss = 1-accuracy_score(y_test, current_y_pred)
            #print(current_loss)
        
        if(abs(ordered_loss-current_loss) <= sensitivity):
            count+=1
    
    #print("count")
    #print(count)
    
    value = (1+count)/(1+number_of_permutation)
            
    if value <= significance_rate:
        return True
    else:
        return False

from ADBench.baseline.DevNet.run import DevNet
from ADBench.baseline.PyOD import PYOD
from ADBench.baseline.DAGMM.run import DAGMM
from ADBench.baseline.GANomaly.run import GANomaly
from ADBench.baseline.DeepSAD.src.run import DeepSAD
from ADBench.baseline.REPEN.run import REPEN
from ADBench.baseline.PReNet.run import PReNet	
from ADBench.baseline.FEAWAD.run import FEAWAD
from ADBench.baseline.Supervised import supervised
from ADBench.baseline.FTTransformer.run import FTTransformer

import warnings
warnings.filterwarnings("ignore")
from ADBench.data_generator import DataGenerator
from ADBench.myutils import Utils


def outlier_detector(data, window_size, window_count):
    
    seed=42
    
    model_dict = {
        'ECOD':PYOD,
        'IForest':PYOD,
    }
    
    outlier_stats_each_window = pd.DataFrame(index = ['ECOD', 'IForest', "mean"],
                                             columns = list(range(1, window_count)))
    
    # outlier_stats_overall = pd.DataFrame(columns = ['ECOD', 'IForest', "mean"])
    outlier_stats_overall = {}
    
    anomaly_count_list = []
    anomaly_index = []
    anomaly_sum = 0
    
    IForest_ave_anomaly_ratio_list = []
    ECOD_ave_anomaly_ratio_list = []
    
    IForest_overall_anomaly_ratio = 0
    ECOD_overall_anomaly_ratio = 0
    
    for name, clf in model_dict.items():
        #print(name)
        clf = clf(seed=seed, model_name=name)
        clf = clf.fit(data, [])
        # output predicted anomaly score on testing set
        score = clf.predict_score(data)
        #print(score)
        t = score.mean()+3*score.std()
        #print(t)
        anomaly_index = np.where(score>t)[0]
        #print(anomaly_index)
        anomaly_count = len(anomaly_index)
        anomaly_sum = anomaly_sum + anomaly_count
        anomaly_count_list.append(anomaly_count)
        #print(anomaly_count)
        outlier_stats_overall[name] = anomaly_count
        anomaly_ratio = anomaly_count/(len(data))
        
        if name =="ECOD":
            ECOD_overall_anomaly_ratio = anomaly_ratio
        else:
            IForest_overall_anomaly_ratio = anomaly_ratio
        
        for n in range(window_count):
            anomaly_count_current_window = 0
            for pos in anomaly_index:
                start = n*window_size
                end = (n+1)*window_size
                if pos >= start and pos < end:
                    anomaly_count_current_window+=1
        
            #print(anomaly_count_current_window)
            outlier_stats_each_window.loc[name][n] = anomaly_count_current_window   
            anomaly_ratio_current_window = anomaly_count_current_window/window_size
            if name=="IForest":
                IForest_ave_anomaly_ratio_list.append(anomaly_ratio_current_window)
            else:
                ECOD_ave_anomaly_ratio_list.append(anomaly_ratio_current_window)
                
    IForest_ave_anomaly_ratio = mean(IForest_ave_anomaly_ratio_list)
    IForest_max_anomaly_ratio = max(IForest_ave_anomaly_ratio_list)    
    
    ECOD_ave_anomaly_ratio = mean(ECOD_ave_anomaly_ratio_list)
    ECOD_max_anomaly_ratio = max(ECOD_ave_anomaly_ratio_list)       
        
    ave = anomaly_sum/2
    ave_overall_anomaly_ratio = ave/(len(data))
    outlier_stats_overall["mean"] = ave
    
    ave_anomaly_ratio_list= []
    for n in range(1, window_count):
        outlier_stats_each_window.loc["mean"][n] = (outlier_stats_each_window.loc["IForest"][n]+outlier_stats_each_window.loc["ECOD"][n])/2
        ave_anomaly_ratio_list.append((outlier_stats_each_window.loc["mean"][n])/window_size)
    
    mean_ave_anomaly_ratio = mean(ave_anomaly_ratio_list)
    max_ave_anomaly_ratio = max(ave_anomaly_ratio_list)
    
    # print(outlier_stats_overall)
    
    return IForest_ave_anomaly_ratio, IForest_max_anomaly_ratio, ECOD_ave_anomaly_ratio, ECOD_max_anomaly_ratio, mean_ave_anomaly_ratio, max_ave_anomaly_ratio, ECOD_overall_anomaly_ratio, IForest_overall_anomaly_ratio, ave_overall_anomaly_ratio
    # return outlier_stats_each_window, outlier_stats_overall


from sklearn.naive_bayes import GaussianNB
def concept_drift(task, data, target, window_size, window_count):
    training_size = window_size
    
    # X_test = data.iloc[training_size:len(data)]
    # y_true = target.iloc[training_size:len(data)]
    start=0
    end=training_size
    
    if task == "regression":
        clf = LinearRegression()
        X_train = data.iloc[start:end]
        
        y_train = target.iloc[start:end]
        clf.fit(X_train, y_train)
    else:
        clf = GaussianNB()
        for i in range(1, window_count):
            end = end*i
            X_train = data.iloc[start:end]
            y_train = target.iloc[start:end]
            try:
                clf.fit(X_train, y_train)
                break
            except:
                continue
        
    
    
    adwin = ADWINAccuracy()
    ddm = DDM(n_threshold=100, warning_scale=7, drift_scale=10)
    eddm = EDDM(n_threshold=30, warning_thresh=0.7, drift_thresh=0.5)
    
    adwin_count = 0
    ddm_count = 0
    eddm_count = 0
    adwin_warning_count = 0
    ddm_warning_count = 0
    eddm_warning_count = 0
    
    for i in range(end, len(data)):
        X_test = data.iloc[i]
        X_test = np.array(X_test).reshape(1, -1)
        #print(X_test)
        y_pred = clf.predict(X_test)
        # if(task=="classification"):
        #     y_true = target.iloc[i]
        # else:
        #     y_true = int(target.iloc[i])
        y_true = target.iloc[i]
        #print(y_pred)
        
        adwin.update(y_true, y_pred)
        ddm.update(y_true, y_pred)
        eddm.update(y_true, y_pred)
        
        if adwin.drift_state == "warning":
            adwin_warning_count+=1
        
        if ddm.drift_state == "warning":
            ddm_warning_count+=1
            
        if eddm.drift_state == "warning":
            eddm_warning_count+=1
        
        if adwin.drift_state == "drift":
            adwin_count+=1
            
            retrain_start = adwin.retraining_recs[0] + training_size
            retrain_end = adwin.retraining_recs[1] + training_size
            
            X_train = data.iloc[retrain_start:retrain_end]
            y_train = target.iloc[retrain_start:retrain_end]
        
            try:
                if task == "classification":
                    clf = GaussianNB()
                else:
                    clf = LinearRegression()
                clf.fit(X_train, y_train)
            except:
                continue
        
        if ddm.drift_state == "drift":
            ddm_count+=1
            
            retrain_start = ddm.retraining_recs[0] + training_size
            retrain_end = ddm.retraining_recs[1] + training_size
            if (retrain_start == retrain_end):  # minimum retraining window in case of sudden drift
                retrain_start = max(0, retrain_start - 300)

            # If retraining is not desired, omit the next four lines.
            X_train = data.iloc[retrain_start:retrain_end]
            y_train = target.iloc[retrain_start:retrain_end]
            
            try:
                if task == "classification":
                    clf = GaussianNB()
                else:
                    clf = LinearRegression()
                clf.fit(X_train, y_train)
            except:
                continue
            
        if eddm.drift_state == "drift":
            eddm_count+=1
            try:
                retrain_start = eddm.retraining_recs[0] + training_size
                retrain_end = eddm.retraining_recs[1] + training_size
            except:
                retrain_start = eddm.retraining_recs[0] + training_size
                retrain_end = eddm.retraining_recs[1] + training_size
            if (retrain_start == retrain_end):  # minimum retraining window in case of sudden drift
                retrain_start = max(0, retrain_start - 300)

            # If retraining is not desired, omit the next four lines.
            X_train = data.iloc[retrain_start:retrain_end]
            y_train = target.iloc[retrain_start:retrain_end]
            
            try:
                if task == "classification":
                    clf = GaussianNB()
                else:
                    clf = LinearRegression()
                clf.fit(X_train, y_train)
            except:
                continue
    
    n = (len(data)-end)
    adwin_drift_percentage = adwin_count/n
    ddm_drift_percentage = ddm_count/n
    eddm_drift_percentage = eddm_count/n
    ave = (adwin_drift_percentage + ddm_drift_percentage + eddm_drift_percentage)/3
    
    adwin_warning_percentage = adwin_warning_count/n
    ddm_warning_percentage = ddm_warning_count/n
    eddm_warning_percentage = eddm_warning_count/n
    warning_ave = (adwin_warning_percentage + ddm_warning_percentage + eddm_warning_percentage)/3
    
    print(adwin_drift_percentage)
    print(ddm_drift_percentage)
    print(eddm_drift_percentage)
    print(ave)
    
    print(adwin_warning_percentage)
    print(ddm_warning_percentage)
    print(eddm_warning_percentage)
    print(warning_ave)
    
    return adwin_drift_percentage, ddm_drift_percentage, eddm_drift_percentage, ave, adwin_warning_percentage, ddm_warning_percentage, eddm_warning_percentage, warning_ave



def run_pipeline(dataset_prefix_list, done=[], generated=False, return_info=False, prefix=""):   
    if generated:
        dataset_prefix_list = ["SEAGenerator", "HyperplaneGenerator", "STAGGERGenerator", "RandomRBFGeneratorDrift", "LEDGeneratorDrift", "WaveformGenerator"]
        generated_dataset = dict({})
        for i in range(4):
            stream = SEAGenerator(classification_function = i, balance_classes = False)
            if i==0:
                x_total, y_total = stream.next_sample(12500)
            else:
                x, y = stream.next_sample(12500)
                x_total = np.concatenate((x_total,x),axis=0)
                y_total = np.concatenate((y_total,y),axis=0)
        generated_dataset["SEAGenerator"] = dict({})
        generated_dataset["SEAGenerator"]["input"] = x_total
        generated_dataset["SEAGenerator"]["target"] = y_total

        stream = HyperplaneGenerator(mag_change=0.1)
        x, y = stream.next_sample(50000)
        generated_dataset["HyperplaneGenerator"] = dict({})
        generated_dataset["HyperplaneGenerator"]["input"] = x
        generated_dataset["HyperplaneGenerator"]["target"] = y

        for i in range(3):
            stream = STAGGERGenerator(classification_function = i, balance_classes = False)
            if i==0:
                x_total, y_total = stream.next_sample(16700)
            else:
                x, y = stream.next_sample(16700)
                x_total = np.concatenate((x_total,x),axis=0)
                y_total = np.concatenate((y_total,y),axis=0)
        generated_dataset["STAGGERGenerator"] = dict({})
        generated_dataset["STAGGERGenerator"]["input"] = x_total
        generated_dataset["STAGGERGenerator"]["target"] = y_total
        
        stream = RandomRBFGeneratorDrift(n_classes=4, change_speed=0.87)
        x, y = stream.next_sample(50000)
        generated_dataset["RandomRBFGeneratorDrift"] = dict({})
        generated_dataset["RandomRBFGeneratorDrift"]["input"] = x
        generated_dataset["RandomRBFGeneratorDrift"]["target"] = y

        stream = LEDGeneratorDrift(noise_percentage = 0.28,has_noise= True, n_drift_features=4)
        x, y = stream.next_sample(50000)
        generated_dataset["LEDGeneratorDrift"] = dict({})
        generated_dataset["LEDGeneratorDrift"]["input"] = x
        generated_dataset["LEDGeneratorDrift"]["target"] = y

        stream = WaveformGenerator(has_noise= True)
        x, y = stream.next_sample(50000)
        generated_dataset["WaveformGenerator"] = dict({})
        generated_dataset["WaveformGenerator"]["input"] = x
        generated_dataset["WaveformGenerator"]["target"] = y

    overall_stats = pd.DataFrame(index=dataset_prefix_list, columns=["size", "#columns", 
                                                                 "ave_rows_with_missing_values_ratio_per_window", "max_rows_with_missing_values_ratio_per_window", 
                                                                 "total_rows_with_missing_values_ratio",
                                                                 "overall missing value ratio", "overall average null columns ratio", 
                                                                 "average null columns ratio among all windows", "max average #null columns ratio among all windows", 
                                                                 "average missing value ratio among all windows", "max missing value ration among all windows",
                                                                 "IForest_ave_anomaly_ratio", "IForest_max_anomaly_ratio", 
                                                                 "ECOD_ave_anomaly_ratio", "ECOD_max_anomaly_ratio", 
                                                                 "mean_ave_anomaly_ratio", "max_ave_anomaly_ratio", 
                                                                 "ECOD_overall_anomaly_ratio", "IForest_overall_anomaly_ratio", 
                                                                 "ave_overall_anomaly_ratio",
                                                                 "hdddm_drift_percentage", "kdq_drift_percentage", "ave_drift_percentage",
                                                                 "hdddm_warning_percentage", "kdq_warning_percentage", "ave_warning_percentage",
                                                                 "ks_ave_drift_percentage", "ks_max_drift_percentage",
                                                                 "hdddm_ave_drift_percentage", "hdddm_max_drift_percentage", 
                                                                 "kdq_ave_drift_percentage", "kdq_max_drift_percentage", 
                                                                 "cbdb_ave_drift_percentage", "cbdb_max_drift_percentage", 
                                                                 "pca_ave_drift_percentage", "pca_max_drift_percentage",
                                                                 "ave_drift_percentage", "max_drift_percentage",
                                                                 "ks_ave_warning_percentage", "ks_max_warning_percentage",
                                                                 "hdddm_ave_warning_percentage", "hdddm_max_warning_percentage", 
                                                                 "kdq_ave_warning_percentage", "kdq_max_warning_percentage", 
                                                                 "cbdb_ave_warning_percentage", "cbdb_max_warning_percentage", 
                                                                 "pca_ave_warning_percentage", "pca_max_warning_percentage", 
                                                                 "ave_warning_percentage", "max_warning_percentage",
                                                                 "concept_drift_ratio", "adwin", "ddm", "eddm", "ave", "adwin_warning", "ddm_warning", "eddm_warning", "ave_warning"])
            
    for dataset_path_prefix in dataset_prefix_list:
        if dataset_path_prefix in done:
            continue

        logging.info(f"Processing data with prefix {dataset_path_prefix}")
        if generated:
            window_size = 500
            input = generated_dataset[dataset_path_prefix]["input"]
            target = generated_dataset[dataset_path_prefix]["target"]
            task = "classification"
            column_count = input.shape[1]
            output_dim = np.max(target) + 1
            target_data_nonnull = pd.DataFrame(target)
            data_before_onehot = pd.DataFrame(input)
            data_onehot_nonnull = pd.DataFrame(input) 
            original_columns = data_before_onehot.columns
            row_count = input.shape[0]
            original_column_count = column_count 
            new_columns = data_before_onehot.columns
        else:
            try:
                data_path, schema_path, task = schema_parser(prefix + dataset_path_prefix)
                data_path = prefix + data_path
            except:
                continue

            logging.info(f"Start data pre-processing for {dataset_path_prefix}")

            if return_info:
                target_data_nonnull, data_before_onehot, data_onehot_nonnull, original_columns, window_size, row_count, original_column_count, new_columns, new_column_count, data_one_hot = data_preprocessing(prefix + dataset_path_prefix, data_path, schema_path, task, return_info=return_info)
            else:
                target_data_nonnull, data_before_onehot, data_onehot_nonnull, original_columns, window_size, row_count, original_column_count, new_columns, new_column_count = data_preprocessing(prefix + dataset_path_prefix, data_path, schema_path, task, return_info=return_info)
            
            logging.info(f"Data preprocessing for {dataset_path_prefix} has been done")

            if task == "regression":
                output_dim = 1
            else:
                output_dim = np.max(target_data_nonnull) + 1

        current_stats = pd.DataFrame(index=dataset_prefix_list, columns=["size", "#columns", 
                                                                    "ave_rows_with_missing_values_ratio_per_window", "max_rows_with_missing_values_ratio_per_window", 
                                                                    "total_rows_with_missing_values_ratio",
                                                                    "overall missing value ratio", "overall average null columns ratio", 
                                                                    "average null columns ratio among all windows", "max average #null columns ratio among all windows", 
                                                                    "average missing value ratio among all windows", "max missing value ration among all windows",
                                                                    "IForest_ave_anomaly_ratio", "IForest_max_anomaly_ratio", 
                                                                    "ECOD_ave_anomaly_ratio", "ECOD_max_anomaly_ratio", 
                                                                    "mean_ave_anomaly_ratio", "max_ave_anomaly_ratio", 
                                                                    "ECOD_overall_anomaly_ratio", "IForest_overall_anomaly_ratio", 
                                                                    "ave_overall_anomaly_ratio",
                                                                    "hdddm_drift_percentage", "kdq_drift_percentage", "ave_drift_percentage",
                                                                    "hdddm_warning_percentage", "kdq_warning_percentage", "ave_warning_percentage",
                                                                    "ks_ave_drift_percentage", "ks_max_drift_percentage",
                                                                    "hdddm_ave_drift_percentage", "hdddm_max_drift_percentage", 
                                                                    "kdq_ave_drift_percentage", "kdq_max_drift_percentage", 
                                                                    "cbdb_ave_drift_percentage", "cbdb_max_drift_percentage", 
                                                                    "pca_ave_drift_percentage", "pca_max_drift_percentage",
                                                                    "ave_drift_percentage", "max_drift_percentage",
                                                                    "ks_ave_warning_percentage", "ks_max_warning_percentage",
                                                                    "hdddm_ave_warning_percentage", "hdddm_max_warning_percentage", 
                                                                    "kdq_ave_warning_percentage", "kdq_max_warning_percentage", 
                                                                    "cbdb_ave_warning_percentage", "cbdb_max_warning_percentage", 
                                                                    "pca_ave_warning_percentage", "pca_max_warning_percentage", 
                                                                    "ave_warning_percentage", "max_warning_percentage",
                                                                    "concept_drift_ratio", "adwin", "ddm", "eddm", "ave", "adwin_warning", "ddm_warning", "eddm_warning", "ave_warning"])
        

        current_stats.loc[dataset_path_prefix]["size"] = row_count
        current_stats.loc[dataset_path_prefix]["#columns"] = original_column_count

        overall_stats.loc[dataset_path_prefix]["size"] = row_count
        overall_stats.loc[dataset_path_prefix]["#columns"] = original_column_count

        window_count = int(int(row_count)/int(window_size))
        print(window_count)

        ave_rows_with_missing_values_ratio_per_window, max_rows_with_missing_values_ratio_per_window, total_rows_with_missing_values_ratio, ave_ave_null_columns_ratio, max_ave_null_columns_ratio, ave_missing_value_ratio, max_missing_value_ratio, overall_ave_null_columns_ratio, overall_missing_value_ratio = missing_value_processor(data_before_onehot, window_size, original_columns, window_count, row_count)
        
        current_stats.loc[dataset_path_prefix]["ave_rows_with_missing_values_ratio_per_window"] = ave_rows_with_missing_values_ratio_per_window
        current_stats.loc[dataset_path_prefix]["max_rows_with_missing_values_ratio_per_window"] = max_rows_with_missing_values_ratio_per_window
        current_stats.loc[dataset_path_prefix]["total_rows_with_missing_values_ratio"] = total_rows_with_missing_values_ratio
        current_stats.loc[dataset_path_prefix]["overall missing value ratio"] = overall_missing_value_ratio
        current_stats.loc[dataset_path_prefix]["overall average null columns ratio"] = overall_ave_null_columns_ratio
        current_stats.loc[dataset_path_prefix]["average null columns ratio among all windows"] = ave_ave_null_columns_ratio
        current_stats.loc[dataset_path_prefix]["max average #null columns ratio among all windows"] = max_ave_null_columns_ratio
        current_stats.loc[dataset_path_prefix]["average missing value ratio among all windows"] = ave_missing_value_ratio
        current_stats.loc[dataset_path_prefix]["max missing value ration among all windows"] = max_missing_value_ratio
        current_stats.loc[dataset_path_prefix]["max missing value ration among all windows"] = max_missing_value_ratio
        
        overall_stats.loc[dataset_path_prefix]["ave_rows_with_missing_values_ratio_per_window"] = ave_rows_with_missing_values_ratio_per_window
        overall_stats.loc[dataset_path_prefix]["max_rows_with_missing_values_ratio_per_window"] = max_rows_with_missing_values_ratio_per_window
        overall_stats.loc[dataset_path_prefix]["total_rows_with_missing_values_ratio"] = total_rows_with_missing_values_ratio
        overall_stats.loc[dataset_path_prefix]["overall missing value ratio"] = overall_missing_value_ratio
        overall_stats.loc[dataset_path_prefix]["overall average null columns ratio"] = overall_ave_null_columns_ratio
        overall_stats.loc[dataset_path_prefix]["average null columns ratio among all windows"] = ave_ave_null_columns_ratio
        overall_stats.loc[dataset_path_prefix]["max average #null columns ratio among all windows"] = max_ave_null_columns_ratio
        overall_stats.loc[dataset_path_prefix]["average missing value ratio among all windows"] = ave_missing_value_ratio
        overall_stats.loc[dataset_path_prefix]["max missing value ration among all windows"] = max_missing_value_ratio
        
        IForest_ave_anomaly_ratio, IForest_max_anomaly_ratio, ECOD_ave_anomaly_ratio, ECOD_max_anomaly_ratio, mean_ave_anomaly_ratio, max_ave_anomaly_ratio, ECOD_overall_anomaly_ratio, IForest_overall_anomaly_ratio, ave_overall_anomaly_ratio = outlier_detector(data_onehot_nonnull, window_size, window_count)
        
        current_stats.loc[dataset_path_prefix]["IForest_ave_anomaly_ratio"] = IForest_ave_anomaly_ratio
        current_stats.loc[dataset_path_prefix]["IForest_max_anomaly_ratio"] = IForest_max_anomaly_ratio
        current_stats.loc[dataset_path_prefix]["ECOD_ave_anomaly_ratio"] = ECOD_ave_anomaly_ratio
        current_stats.loc[dataset_path_prefix]["ECOD_max_anomaly_ratio"] = ECOD_max_anomaly_ratio
        current_stats.loc[dataset_path_prefix]["mean_ave_anomaly_ratio"] = mean_ave_anomaly_ratio
        current_stats.loc[dataset_path_prefix]["max_ave_anomaly_ratio"] = max_ave_anomaly_ratio
        current_stats.loc[dataset_path_prefix]["ECOD_overall_anomaly_ratio"] = ECOD_overall_anomaly_ratio
        current_stats.loc[dataset_path_prefix]["IForest_overall_anomaly_ratio"] = IForest_overall_anomaly_ratio
        current_stats.loc[dataset_path_prefix]["ave_overall_anomaly_ratio"] = ave_overall_anomaly_ratio
        
        overall_stats.loc[dataset_path_prefix]["IForest_ave_anomaly_ratio"] = IForest_ave_anomaly_ratio
        overall_stats.loc[dataset_path_prefix]["IForest_max_anomaly_ratio"] = IForest_max_anomaly_ratio
        overall_stats.loc[dataset_path_prefix]["ECOD_ave_anomaly_ratio"] = ECOD_ave_anomaly_ratio
        overall_stats.loc[dataset_path_prefix]["ECOD_max_anomaly_ratio"] = ECOD_max_anomaly_ratio
        overall_stats.loc[dataset_path_prefix]["mean_ave_anomaly_ratio"] = mean_ave_anomaly_ratio
        overall_stats.loc[dataset_path_prefix]["max_ave_anomaly_ratio"] = max_ave_anomaly_ratio
        overall_stats.loc[dataset_path_prefix]["ECOD_overall_anomaly_ratio"] = ECOD_overall_anomaly_ratio
        overall_stats.loc[dataset_path_prefix]["IForest_overall_anomaly_ratio"] = IForest_overall_anomaly_ratio
        overall_stats.loc[dataset_path_prefix]["ave_overall_anomaly_ratio"] = ave_overall_anomaly_ratio
        
        logging.info(f"Start multi-dimensional drift detection for {dataset_path_prefix}")

        hdddm_drift_percentage, kdq_drift_percentage, ave_drift_percentage, hdddm_warning_percentage, kdq_warning_percentage, ave_warning_percentage = data_drift_detector_multi_dimensional(data_onehot_nonnull, window_size, window_count)
        
        current_stats.loc[dataset_path_prefix]["hdddm_drift_percentage"] = hdddm_drift_percentage
        current_stats.loc[dataset_path_prefix]["kdq_drift_percentage"] = kdq_drift_percentage
        current_stats.loc[dataset_path_prefix]["ave_drift_percentage"] = ave_drift_percentage
        current_stats.loc[dataset_path_prefix]["hdddm_warning_percentage"] = hdddm_warning_percentage
        current_stats.loc[dataset_path_prefix]["kdq_warning_percentage"] = kdq_warning_percentage
        current_stats.loc[dataset_path_prefix]["ave_warning_percentage"] = ave_warning_percentage
        
        overall_stats.loc[dataset_path_prefix]["hdddm_drift_percentage"] = hdddm_drift_percentage
        overall_stats.loc[dataset_path_prefix]["kdq_drift_percentage"] = kdq_drift_percentage
        overall_stats.loc[dataset_path_prefix]["ave_drift_percentage"] = ave_drift_percentage
        overall_stats.loc[dataset_path_prefix]["hdddm_warning_percentage"] = hdddm_warning_percentage
        overall_stats.loc[dataset_path_prefix]["kdq_warning_percentage"] = kdq_warning_percentage
        overall_stats.loc[dataset_path_prefix]["ave_warning_percentage"] = ave_warning_percentage

        print("start one-dimensional drift detection")
        ks_ave_drift_percentage, ks_max_drift_percentage, hdddm_ave_drift_percentage, hdddm_max_drift_percentage, kdq_ave_drift_percentage, kdq_max_drift_percentage, cbdb_ave_drift_percentage, cbdb_max_drift_percentage, pca_ave_drift_percentage, pca_max_drift_percentage, ave_drift_percentage, max_drift_percentage, hdddm_ave_warning_percentage, hdddm_max_warning_percentage, kdq_ave_warning_percentage, kdq_max_warning_percentage, cbdb_ave_warning_percentage, cbdb_max_warning_percentage, pca_ave_warning_percentage, pca_max_warning_percentage, ks_ave_warning_percentage, ks_max_warning_percentage, ave_warning_percentage, max_warning_percentage = data_drift_detector_one_dimensional(data_onehot_nonnull, window_size, window_count, new_columns)
        print("ks stats")
        print(ks_ave_drift_percentage)
        print(ks_max_drift_percentage)
        current_stats.loc[dataset_path_prefix]["ks_ave_drift_percentage"] = ks_ave_drift_percentage
        current_stats.loc[dataset_path_prefix]["ks_max_drift_percentage"] = ks_max_drift_percentage
        current_stats.loc[dataset_path_prefix]["hdddm_ave_drift_percentage"] = hdddm_ave_drift_percentage
        current_stats.loc[dataset_path_prefix]["hdddm_max_drift_percentage"] = hdddm_max_drift_percentage
        current_stats.loc[dataset_path_prefix]["kdq_ave_drift_percentage"] = kdq_ave_drift_percentage
        current_stats.loc[dataset_path_prefix]["kdq_max_drift_percentage"] = kdq_max_drift_percentage
        current_stats.loc[dataset_path_prefix]["cbdb_ave_drift_percentage"] = cbdb_ave_drift_percentage
        current_stats.loc[dataset_path_prefix]["cbdb_max_drift_percentage"] = cbdb_max_drift_percentage
        current_stats.loc[dataset_path_prefix]["ave_drift_percentage"] = ave_drift_percentage
        current_stats.loc[dataset_path_prefix]["max_drift_percentage"] = max_drift_percentage
        current_stats.loc[dataset_path_prefix]["pca_ave_drift_percentage"] = pca_ave_drift_percentage
        current_stats.loc[dataset_path_prefix]["pca_max_drift_percentage"] = pca_max_drift_percentage
        
        current_stats.loc[dataset_path_prefix]["ks_ave_warning_percentage"] = ks_ave_warning_percentage
        current_stats.loc[dataset_path_prefix]["ks_max_warning_percentage"] = ks_max_warning_percentage
        current_stats.loc[dataset_path_prefix]["hdddm_ave_warning_percentage"] = hdddm_ave_warning_percentage
        current_stats.loc[dataset_path_prefix]["hdddm_max_warning_percentage"] = hdddm_max_warning_percentage
        current_stats.loc[dataset_path_prefix]["kdq_ave_warning_percentage"] = kdq_ave_warning_percentage
        current_stats.loc[dataset_path_prefix]["kdq_max_warning_percentage"] = kdq_max_warning_percentage
        current_stats.loc[dataset_path_prefix]["cbdb_ave_warning_percentage"] = cbdb_ave_warning_percentage
        current_stats.loc[dataset_path_prefix]["cbdb_max_warning_percentage"] = cbdb_max_warning_percentage
        current_stats.loc[dataset_path_prefix]["ave_warning_percentage"] = ave_warning_percentage
        current_stats.loc[dataset_path_prefix]["max_warning_percentage"] = max_warning_percentage
        current_stats.loc[dataset_path_prefix]["pca_ave_warning_percentage"] = pca_ave_warning_percentage
        current_stats.loc[dataset_path_prefix]["pca_max_warning_percentage"] = pca_max_warning_percentage
        
        overall_stats.loc[dataset_path_prefix]["ks_ave_drift_percentage"] = ks_ave_drift_percentage
        overall_stats.loc[dataset_path_prefix]["ks_max_drift_percentage"] = ks_max_drift_percentage
        overall_stats.loc[dataset_path_prefix]["hdddm_ave_drift_percentage"] = hdddm_ave_drift_percentage
        overall_stats.loc[dataset_path_prefix]["hdddm_max_drift_percentage"] = hdddm_max_drift_percentage
        overall_stats.loc[dataset_path_prefix]["kdq_ave_drift_percentage"] = kdq_ave_drift_percentage
        overall_stats.loc[dataset_path_prefix]["kdq_max_drift_percentage"] = kdq_max_drift_percentage
        overall_stats.loc[dataset_path_prefix]["cbdb_ave_drift_percentage"] = cbdb_ave_drift_percentage
        overall_stats.loc[dataset_path_prefix]["cbdb_max_drift_percentage"] = cbdb_max_drift_percentage
        overall_stats.loc[dataset_path_prefix]["ave_drift_percentage"] = ave_drift_percentage
        overall_stats.loc[dataset_path_prefix]["max_drift_percentage"] = max_drift_percentage
        overall_stats.loc[dataset_path_prefix]["pca_ave_drift_percentage"] = pca_ave_drift_percentage
        overall_stats.loc[dataset_path_prefix]["pca_max_drift_percentage"] = pca_max_drift_percentage
        
        overall_stats.loc[dataset_path_prefix]["ks_ave_warning_percentage"] = ks_ave_warning_percentage
        overall_stats.loc[dataset_path_prefix]["ks_max_warning_percentage"] = ks_max_warning_percentage
        overall_stats.loc[dataset_path_prefix]["hdddm_ave_warning_percentage"] = hdddm_ave_warning_percentage
        overall_stats.loc[dataset_path_prefix]["hdddm_max_warning_percentage"] = hdddm_max_warning_percentage
        overall_stats.loc[dataset_path_prefix]["kdq_ave_warning_percentage"] = kdq_ave_warning_percentage
        overall_stats.loc[dataset_path_prefix]["kdq_max_warning_percentage"] = kdq_max_warning_percentage
        overall_stats.loc[dataset_path_prefix]["cbdb_ave_warning_percentage"] = cbdb_ave_warning_percentage
        overall_stats.loc[dataset_path_prefix]["cbdb_max_warning_percentage"] = cbdb_max_warning_percentage
        overall_stats.loc[dataset_path_prefix]["ave_warning_percentage"] = ave_warning_percentage
        overall_stats.loc[dataset_path_prefix]["max_warning_percentage"] = max_warning_percentage
        overall_stats.loc[dataset_path_prefix]["pca_ave_warning_percentage"] = pca_ave_warning_percentage
        overall_stats.loc[dataset_path_prefix]["pca_max_warning_percentage"] = pca_max_warning_percentage
        
        try:
            concept_drift_ratio = PERM(task, data_onehot_nonnull, target_data_nonnull, window_size, window_count, 20, 0.02, 0.05)
            
            current_concept_drift_stats = pd.DataFrame(index = dataset_prefix_list, columns=["concept_drift_ratio"])
            overall_concept_drift_stats = pd.DataFrame(index = dataset_prefix_list, columns=["concept_drift_ratio"])
            current_concept_drift_stats.loc[dataset_path_prefix]["concept_drift_ratio"] = concept_drift_ratio
            overall_concept_drift_stats.loc[dataset_path_prefix]["concept_drift_ratio"] = concept_drift_ratio
            overall_stats.loc[dataset_path_prefix]["concept_drift_ratio"] = concept_drift_ratio
            
            current_concept_drift_stats.to_csv(dataset_path_prefix + '/concept_drift_stats.csv', mode='w')
            
        except:
            pass
            #continue
        '''
        concept_drift_ratio = PERM(task, data_onehot_nonnull, target_data_nonnull, window_size, window_count, 20, 0.02, 0.05)
            
        current_concept_drift_stats = pd.DataFrame(index = dataset_prefix_list, columns=["concept_drift_ratio"])
        overall_concept_drift_stats = pd.DataFrame(index = dataset_prefix_list, columns=["concept_drift_ratio"])
        current_concept_drift_stats.loc[dataset_path_prefix]["concept_drift_ratio"] = concept_drift_ratio           
        overall_concept_drift_stats.loc[dataset_path_prefix]["concept_drift_ratio"] = concept_drift_ratio
            
            
        current_concept_drift_stats.to_csv(dataset_path_prefix + '/concept_drift_stats.csv', mode='w')
        '''
        
        adwin, ddm, eddm, ave, adwin_warning, ddm_warning, eddm_warning, warning_ave = concept_drift(task, data_onehot_nonnull, target_data_nonnull, window_size, window_count)
        
        current_concept_drift_stats = pd.DataFrame(index = dataset_prefix_list, columns=["adwin", "ddm", "eddm", "ave", "adwin_warning", "ddm_warning", "eddm_warning", "ave_warning"])
        current_concept_drift_stats.loc[dataset_path_prefix]["adwin"] = adwin
        current_concept_drift_stats.loc[dataset_path_prefix]["ddm"] = ddm
        current_concept_drift_stats.loc[dataset_path_prefix]["eddm"] = eddm
        current_concept_drift_stats.loc[dataset_path_prefix]["ave"] = ave
        
        current_concept_drift_stats.loc[dataset_path_prefix]["adwin_warning"] = adwin_warning
        current_concept_drift_stats.loc[dataset_path_prefix]["ddm_warning"] = ddm_warning
        current_concept_drift_stats.loc[dataset_path_prefix]["eddm_warning"] = eddm_warning
        current_concept_drift_stats.loc[dataset_path_prefix]["ave_warning"] = warning_ave

        overall_stats.loc[dataset_path_prefix]["adwin"] = adwin
        overall_stats.loc[dataset_path_prefix]["ddm"] = ddm
        overall_stats.loc[dataset_path_prefix]["eddm"] = eddm
        overall_stats.loc[dataset_path_prefix]["ave"] = ave
        
        overall_stats.loc[dataset_path_prefix]["adwin_warning"] = adwin_warning
        overall_stats.loc[dataset_path_prefix]["ddm_warning"] = ddm_warning
        overall_stats.loc[dataset_path_prefix]["eddm_warning"] = eddm_warning
        overall_stats.loc[dataset_path_prefix]["ave_warning"] = warning_ave
        
        
        # current_concept_drift_stats.to_csv(dataset_path_prefix + '/menelaus_concept_drift_stats.csv', mode='w')
        
        done.append(dataset_path_prefix)
        print(done)
        
        
        overall_stats.to_csv('overall_stats.csv', mode='w')

        if return_info:
            return overall_stats, target_data_nonnull, data_before_onehot, data_onehot_nonnull, window_size, output_dim, data_one_hot, task
    

# done = ['dataset_experiment_info/weather_indian_cities/lucknow', 'dataset_experiment_info/weather_indian_cities/mumbai', 'dataset_experiment_info/weather_indian_cities/Rajasthan', 'dataset_experiment_info/weather_indian_cities/Bhubhneshwar', 'dataset_experiment_info/weather_indian_cities/delhi', 'dataset_experiment_info/weather_indian_cities/chennai', 'dataset_experiment_info/taxi_ride_duration','dataset_experiment_info/election','dataset_experiment_info/energy_prediction','dataset_experiment_info/room_occupancy','dataset_experiment_info/covtype','dataset_experiment_info/news_popularity','dataset_experiment_info/traffic_volumn','dataset_experiment_info/KDDCUP99','dataset_experiment_info/powersupply','dataset_experiment_info/bitcoin', 'dataset_experiment_info/airlines', 'dataset_experiment_info/beijingPM2.5', 'dataset_experiment_info/household', 'dataset_experiment_info/luxembourg', 'dataset_experiment_info/safe_driver', 'dataset_experiment_info/5cities/shenyang', 'dataset_experiment_info/5cities/guangzhou', 'dataset_experiment_info/5cities/beijing', 'dataset_experiment_info/5cities/shanghai', 'dataset_experiment_info/5cities/chengdu','dataset_experiment_info', 'dataset_experiment_info/allstate_claims_severity', 'dataset_experiment_info/bike_sharing_demand', 'dataset_experiment_info/rssi', 'dataset_experiment_info/noaa', 'dataset_experiment_info/electricity_prices', 'dataset_experiment_info/tetouan', 'dataset_experiment_info/beijing_multisite', 'dataset_experiment_info/beijing_multisite/wanliu', 'dataset_experiment_info/beijing_multisite/wanshouxingong', 'dataset_experiment_info/beijing_multisite/gucheng', 'dataset_experiment_info/beijing_multisite/huairou', 'dataset_experiment_info/beijing_multisite/nongzhanguan', 'dataset_experiment_info/beijing_multisite/changping', 'dataset_experiment_info/beijing_multisite/dingling', 'dataset_experiment_info/beijing_multisite/aotizhongxin', 'dataset_experiment_info/beijing_multisite/dongsi', 'dataset_experiment_info/beijing_multisite/shunyi', 'dataset_experiment_info/beijing_multisite/guanyuan', 'dataset_experiment_info/beijing_multisite/tiantan', 'dataset_experiment_info/insects', 'dataset_experiment_info/insects/abrupt_imbalanced', 'dataset_experiment_info/insects/out-of-control', 'dataset_experiment_info/insects/incremental_imbalanced', 'dataset_experiment_info/insects/incremental_reoccurring_balanced', 'dataset_experiment_info/insects/incremental_balanced', 'dataset_experiment_info/insects/incremental_abrupt_balanced', 'dataset_experiment_info/insects/gradual_imbalanced', 'dataset_experiment_info/insects/abrupt_balanced', 'dataset_experiment_info/insects/incremental_abrupt_imbalanced', 'dataset_experiment_info/insects/incremental_reoccurring_imbalanced', 'dataset_experiment_info/insects/gradual_balanced', 'dataset_experiment_info/italian_city_airquality']
# done = ['dataset_experiment_info/rssi']
# done = ['dataset_experiment_info/household','dataset_experiment_info/luxembourg', 'dataset_experiment_info/beijingPM2.5', 'dataset_experiment_info/airbnb', 'dataset_experiment_info/bitcoin', 'dataset_experiment_info/airlines', 'dataset_experiment_info/italian_city_airquality', 'dataset_experiment_info/airbnb', 'dataset_experiment_info/allstate_claims_severity', 'dataset_experiment_info/bike_sharing_demand', 'dataset_experiment_info/noaa', 'dataset_experiment_info/electricity_prices', 'dataset_experiment_info/tetouan', 'dataset_experiment_info/beijing_multisite/wanliu', 'dataset_experiment_info/beijing_multisite/wanshouxingong', 'dataset_experiment_info/beijing_multisite/gucheng', 'dataset_experiment_info/beijing_multisite/huairou', 'dataset_experiment_info/beijing_multisite/nongzhanguan', 'dataset_experiment_info/beijing_multisite/changping', 'dataset_experiment_info/beijing_multisite/dingling', 'dataset_experiment_info/beijing_multisite/aotizhongxin', 'dataset_experiment_info/beijing_multisite/dongsi', 'dataset_experiment_info/beijing_multisite/shunyi', 'dataset_experiment_info/beijing_multisite/guanyuan', 'dataset_experiment_info/beijing_multisite/tiantan', 'dataset_experiment_info/insects/abrupt_imbalanced', 'dataset_experiment_info/insects/out-of-control', 'dataset_experiment_info/insects/incremental_imbalanced', 'dataset_experiment_info/insects/incremental_reoccurring_balanced', 'dataset_experiment_info/insects/incremental_balanced', 'dataset_experiment_info/insects/incremental_abrupt_balanced', 'dataset_experiment_info/insects/gradual_imbalanced', 'dataset_experiment_info/insects/abrupt_balanced', 'dataset_experiment_info/insects/incremental_abrupt_imbalanced', 'dataset_experiment_info/insects/incremental_reoccurring_imbalanced', 'dataset_experiment_info/insects/gradual_balanced']

if __name__ == "__main__":
    rootdir = "dataset_experiment_info"
    dataset_prefix_list = [x[0] for x in os.walk(rootdir)]

    # dataset_prefix_list = ['dataset_experiment_info/weather_indian_cities/lucknow']
    print(dataset_prefix_list)
    # done = ['dataset_experiment_info/airlines', 'dataset_experiment_info/airbnb', 'dataset_experiment_info/allstate_claims_severity', 'dataset_experiment_info/bike_sharing_demand', 'dataset_experiment_info/rssi', 'dataset_experiment_info/KDDCUP99', 'dataset_experiment_info/electricity_prices', 
    #         'dataset_experiment_info/tetouan', 'dataset_experiment_info/beijing_multisite/wanliu', 'dataset_experiment_info/beijing_multisite/wanshouxingong', 'dataset_experiment_info/beijing_multisite/gucheng', 'dataset_experiment_info/beijing_multisite/huairou', 
    #         'dataset_experiment_info/beijing_multisite/nongzhanguan', 'dataset_experiment_info/beijing_multisite/changping', 'dataset_experiment_info/beijing_multisite/dingling', 'dataset_experiment_info/beijing_multisite/aotizhongxin',
    #         'dataset_experiment_info/beijing_multisite/dongsi', 'dataset_experiment_info/beijing_multisite/shunyi', 'dataset_experiment_info/beijing_multisite/guanyuan', 'dataset_experiment_info/beijing_multisite/tiantan', 'dataset_experiment_info/weather_indian_cities/bangalore', 
    #         'dataset_experiment_info/weather_indian_cities/lucknow', 'dataset_experiment_info/weather_indian_cities/mumbai', 'dataset_experiment_info/weather_indian_cities/Rajasthan', 'dataset_experiment_info/weather_indian_cities/Bhubhneshwar', 
    #         'dataset_experiment_info/weather_indian_cities/delhi', 'dataset_experiment_info/weather_indian_cities/chennai', 'dataset_experiment_info/insects/abrupt_imbalanced', 'dataset_experiment_info/insects/out-of-control', 'dataset_experiment_info/insects/incremental_imbalanced', 
    #         'dataset_experiment_info/insects/incremental_balanced', 'dataset_experiment_info/insects/gradual_imbalanced', 'dataset_experiment_info/insects/incremental_abrupt_imbalanced', 'dataset_experiment_info/insects/incremental_reoccurring_imbalanced', 
    #         'dataset_experiment_info/italian_city_airquality', 'dataset_experiment_info/taxi_ride_duration', 'dataset_experiment_info/rialto', 'dataset_experiment_info/noaa']

    # done = ['dataset_experiment_info/airlines', 'dataset_experiment_info/allstate_claims_severity', 'dataset_experiment_info/bike_sharing_demand']
    # done = []
    # done = ['dataset_experiment_info/bitcoin', 'dataset_experiment_info/airbnb' ,'dataset_experiment_info/allstate_claims_severity', 'dataset_experiment_info/bike_sharing_demand', 'dataset_experiment_info/rssi', 'dataset_experiment_info/noaa', 'dataset_experiment_info/KDDCUP99', 'dataset_experiment_info/electricity_prices', 'dataset_experiment_info/tetouan', 'dataset_experiment_info/beijing_multisite/wanliu', 'dataset_experiment_info/beijing_multisite/wanshouxingong', 'dataset_experiment_info/beijing_multisite/gucheng', 'dataset_experiment_info/beijing_multisite/huairou', 'dataset_experiment_info/beijing_multisite/nongzhanguan', 'dataset_experiment_info/beijing_multisite/changping', 'dataset_experiment_info/beijing_multisite/dingling', 'dataset_experiment_info/beijing_multisite/aotizhongxin', 'dataset_experiment_info/beijing_multisite/dongsi', 'dataset_experiment_info/beijing_multisite/shunyi', 'dataset_experiment_info/beijing_multisite/guanyuan', 'dataset_experiment_info/beijing_multisite/tiantan', 'dataset_experiment_info/weather_indian_cities/bangalore', 'dataset_experiment_info/weather_indian_cities/lucknow', 'dataset_experiment_info/weather_indian_cities/mumbai', 'dataset_experiment_info/weather_indian_cities/Rajasthan', 'dataset_experiment_info/weather_indian_cities/Bhubhneshwar', 'dataset_experiment_info/weather_indian_cities/delhi', 'dataset_experiment_info/weather_indian_cities/chennai', 'dataset_experiment_info/insects/abrupt_imbalanced', 'dataset_experiment_info/insects/out-of-control', 'dataset_experiment_info/insects/incremental_imbalanced', 'dataset_experiment_info/insects/incremental_reoccurring_balanced', 'dataset_experiment_info/insects/incremental_balanced', 'dataset_experiment_info/insects/incremental_abrupt_balanced', 'dataset_experiment_info/insects/gradual_imbalanced', 'dataset_experiment_info/insects/abrupt_balanced', 'dataset_experiment_info/insects/incremental_abrupt_imbalanced', 'dataset_experiment_info/insects/incremental_reoccurring_imbalanced', 'dataset_experiment_info/insects/gradual_balanced', 'dataset_experiment_info/italian_city_airquality', 'dataset_experiment_info/taxi_ride_duration']
    # done = ['dataset_experiment_info/airlines', 'dataset_experiment_info/bitcoin', 'dataset_experiment_info/airbnb', 'dataset_experiment_info/allstate_claims_severity', 'dataset_experiment_info/bike_sharing_demand', 'dataset_experiment_info/rssi', 'dataset_experiment_info/noaa', 'dataset_experiment_info/KDDCUP99', 'dataset_experiment_info/electricity_prices', 'dataset_experiment_info/tetouan', 'dataset_experiment_info/beijing_multisite/wanliu', 'dataset_experiment_info/beijing_multisite/wanshouxingong', 'dataset_experiment_info/beijing_multisite/gucheng', 'dataset_experiment_info/beijing_multisite/huairou', 'dataset_experiment_info/beijing_multisite/nongzhanguan', 'dataset_experiment_info/beijing_multisite/changping', 'dataset_experiment_info/beijing_multisite/dingling', 'dataset_experiment_info/beijing_multisite/aotizhongxin', 'dataset_experiment_info/beijing_multisite/dongsi', 'dataset_experiment_info/beijing_multisite/shunyi', 'dataset_experiment_info/beijing_multisite/guanyuan', 'dataset_experiment_info/beijing_multisite/tiantan', 'dataset_experiment_info/weather_indian_cities/bangalore', 'dataset_experiment_info/weather_indian_cities/lucknow', 'dataset_experiment_info/weather_indian_cities/mumbai', 'dataset_experiment_info/weather_indian_cities/Rajasthan', 'dataset_experiment_info/weather_indian_cities/Bhubhneshwar', 'dataset_experiment_info/weather_indian_cities/delhi', 'dataset_experiment_info/weather_indian_cities/chennai', 'dataset_experiment_info/insects/abrupt_imbalanced', 'dataset_experiment_info/insects/out-of-control', 'dataset_experiment_info/insects/incremental_imbalanced', 'dataset_experiment_info/insects/incremental_reoccurring_balanced', 'dataset_experiment_info/insects/incremental_balanced', 'dataset_experiment_info/insects/incremental_abrupt_balanced', 'dataset_experiment_info/insects/gradual_imbalanced', 'dataset_experiment_info/insects/abrupt_balanced', 'dataset_experiment_info/insects/incremental_abrupt_imbalanced', 'dataset_experiment_info/insects/incremental_reoccurring_imbalanced', 'dataset_experiment_info/insects/gradual_balanced', 'dataset_experiment_info/italian_city_airquality', 'dataset_experiment_info/taxi_ride_duration', 'dataset_experiment_info/room_occupancy', 'dataset_experiment_info/rialto']
    # 'dataset_experiment_info/airlines', 'dataset_experiment_info/bitcoin', 'dataset_experiment_info/airbnb'

    # done=['dataset_experiment_info/house_rent', 'dataset_experiment_info/airbnb', 'dataset_experiment_info/allstate_claims_severity', 'dataset_experiment_info/bike_sharing_demand', 'dataset_experiment_info/rssi', 'dataset_experiment_info/noaa', 'dataset_experiment_info/KDDCUP99', 'dataset_experiment_info/electricity_prices', 'dataset_experiment_info/tetouan', 'dataset_experiment_info/beijing_multisite/wanliu', 'dataset_experiment_info/beijing_multisite/wanshouxingong', 'dataset_experiment_info/beijing_multisite/gucheng', 'dataset_experiment_info/beijing_multisite/huairou', 'dataset_experiment_info/beijing_multisite/nongzhanguan', 'dataset_experiment_info/beijing_multisite/changping', 'dataset_experiment_info/beijing_multisite/dingling', 'dataset_experiment_info/beijing_multisite/aotizhongxin', 'dataset_experiment_info/beijing_multisite/dongsi', 'dataset_experiment_info/beijing_multisite/shunyi', 'dataset_experiment_info/beijing_multisite/guanyuan', 'dataset_experiment_info/beijing_multisite/tiantan', 'dataset_experiment_info/weather_indian_cities/bangalore', 'dataset_experiment_info/weather_indian_cities/lucknow', 'dataset_experiment_info/weather_indian_cities/mumbai', 'dataset_experiment_info/weather_indian_cities/Rajasthan', 'dataset_experiment_info/weather_indian_cities/Bhubhneshwar', 'dataset_experiment_info/weather_indian_cities/delhi', 'dataset_experiment_info/weather_indian_cities/chennai', 'dataset_experiment_info/insects/abrupt_imbalanced', 'dataset_experiment_info/insects/out-of-control', 'dataset_experiment_info/insects/incremental_imbalanced', 'dataset_experiment_info/insects/incremental_reoccurring_balanced', 'dataset_experiment_info/insects/incremental_balanced', 'dataset_experiment_info/insects/incremental_abrupt_balanced', 'dataset_experiment_info/insects/gradual_imbalanced', 'dataset_experiment_info/insects/abrupt_balanced', 'dataset_experiment_info/insects/incremental_abrupt_imbalanced', 'dataset_experiment_info/insects/incremental_reoccurring_imbalanced', 'dataset_experiment_info/insects/gradual_balanced', 'dataset_experiment_info/italian_city_airquality', 'dataset_experiment_info/taxi_ride_duration', 'dataset_experiment_info/room_occupancy', 'dataset_experiment_info/rialto', 'dataset_experiment_info/traffic_volumn', 'dataset_experiment_info/news_popularity', 'dataset_experiment_info/beijingPM2.5', 'dataset_experiment_info/energy_prediction', 'dataset_experiment_info/household', 'dataset_experiment_info/election', 'dataset_experiment_info/covtype']
    done = ['dataset_experiment_info/airbnb']
    #dataset_prefix_list = ['dataset_experiment_info/household']
    #done=['dataset_experiment_info/airbnb', 'dataset_experiment_info/allstate_claims_severity', 'dataset_experiment_info/allstate_claims_severity', 'dataset_experiment_info/allstate_claims_severity', 'dataset_experiment_info/bike_sharing_demand', 'dataset_experiment_info/bike_sharing_demand', 'dataset_experiment_info/bike_sharing_demand', 'dataset_experiment_info/rssi', 'dataset_experiment_info/rssi', 'dataset_experiment_info/rssi', 'dataset_experiment_info/noaa', 'dataset_experiment_info/noaa', 'dataset_experiment_info/noaa', 'dataset_experiment_info/KDDCUP99', 'dataset_experiment_info/KDDCUP99', 'dataset_experiment_info/KDDCUP99', 'dataset_experiment_info/electricity_prices', 'dataset_experiment_info/electricity_prices', 'dataset_experiment_info/electricity_prices', 'dataset_experiment_info/tetouan', 'dataset_experiment_info/tetouan', 'dataset_experiment_info/tetouan', 'dataset_experiment_info/italian_city_airquality', 'dataset_experiment_info/italian_city_airquality', 'dataset_experiment_info/italian_city_airquality', 'dataset_experiment_info/powersupply', 'dataset_experiment_info/powersupply', 'dataset_experiment_info/powersupply', 'dataset_experiment_info/taxi_ride_duration', 'dataset_experiment_info/taxi_ride_duration', 'dataset_experiment_info/taxi_ride_duration', 'dataset_experiment_info/room_occupancy', 'dataset_experiment_info/room_occupancy', 'dataset_experiment_info/room_occupancy']

    run_pipeline(dataset_prefix_list, done, generated=False)

