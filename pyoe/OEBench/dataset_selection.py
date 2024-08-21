import pandas as pd
from sklearn.manifold import TSNE
import numpy as np

file = pd.read_excel('dataset_experiment_results.xlsx')
file.head(60)
file.to_csv('dataset_experiment_results.csv')

columns = file.columns

basic_info = file[['size', 'type', 'task', '#columns']]

basic_info_one_hot = pd.get_dummies(basic_info, columns=['task', 'type'])
basic_info_one_hot.head()

mean = np.mean(basic_info_one_hot, axis=0)
std = np.std(basic_info_one_hot, axis=0)
standarized_basic_info_one_hot = (basic_info_one_hot - mean) / std

# model = TSNE(n_components=4, init='pca', learning_rate='auto')
from sklearn.decomposition import PCA
model = PCA(n_components=3)

pca_basic_info = model.fit_transform(standarized_basic_info_one_hot)
pca_basic_info = pd.DataFrame(pca_basic_info)

missing_value_features = file[['ave_rows_with_missing_values_ratio_per_window',
       'max_rows_with_missing_values_ratio_per_window',
       'total_rows_with_missing_values_ratio', 'overall missing value ratio',
       'overall average null columns ratio',
       'average null columns ratio among all windows',
       'max average #null columns ratio among all windows',
       'average missing value ratio among all windows',
       'max missing value ration among all windows']]

pca_missing_value_features = model.fit_transform(missing_value_features)
pca_missing_value_features = pd.DataFrame(pca_missing_value_features)

outliers_features = file[['IForest_ave_anomaly_ratio', 'IForest_max_anomaly_ratio',
       'ECOD_ave_anomaly_ratio', 'ECOD_max_anomaly_ratio',
       'mean_ave_anomaly_ratio', 'max_ave_anomaly_ratio',
       'ECOD_overall_anomaly_ratio', 'IForest_overall_anomaly_ratio',
       'ave_overall_anomaly_ratio']]
pca_outliers_features = model.fit_transform(outliers_features)
pca_outliers_features = pd.DataFrame(pca_outliers_features)

data_drift_features = file[['hdddm_drift_percentage',
       'kdq_drift_percentage', 'ave_drift_percentage',
       'hdddm_warning_percentage', 'kdq_warning_percentage',
       'ave_warning_percentage', 'ks_ave_drift_percentage',
       'ks_max_drift_percentage', 'hdddm_ave_drift_percentage',
       'hdddm_max_drift_percentage', 'kdq_ave_drift_percentage',
       'kdq_max_drift_percentage', 'cbdb_ave_drift_percentage',
       'cbdb_max_drift_percentage', 'pca_ave_drift_percentage',
       'pca_max_drift_percentage', 'ave_drift_percentage.1',
       'max_drift_percentage', 'ks_ave_warning_percentage',
       'ks_max_warning_percentage', 'hdddm_ave_warning_percentage',
       'hdddm_max_warning_percentage', 'kdq_ave_warning_percentage',
       'kdq_max_warning_percentage', 'cbdb_ave_warning_percentage',
       'cbdb_max_warning_percentage', 'pca_ave_warning_percentage',
       'pca_max_warning_percentage', 'ave_warning_percentage.1',
       'max_warning_percentage']]
pca_data_drift_features = model.fit_transform(data_drift_features)
pca_data_drift_features = pd.DataFrame(pca_data_drift_features)

concept_drift_features = file[['perm', 'adwin', 'ddm', 'eddm',
       'concept_drift_ave', 'adwin_warning', 'ddm_warning', 'eddm_warning',
       'ave_concept_drift_warning']]
pca_concept_drift_features = model.fit_transform(concept_drift_features)
pca_concept_drift_features = pd.DataFrame(pca_concept_drift_features)

combined_pca_features = pd.concat([pca_basic_info, pca_missing_value_features, pca_concept_drift_features, pca_data_drift_features, pca_outliers_features], axis=1)

from sklearn.cluster import KMeans

k = 5  # Choose the desired number of clusters
kmeans = KMeans(n_clusters=k)
kmeans.fit(combined_pca_features)

print(kmeans.labels_)

combined_pca_features['cluster'] = kmeans.labels_

representative_datasets_indices = []

for i in range(k):
    cluster_data = combined_pca_features[combined_pca_features['cluster'] == i]
    cluster_center = kmeans.cluster_centers_[i]
    
    # Calculate the Euclidean distance between each data point and the cluster center
    distances = np.linalg.norm(cluster_data.iloc[:, :-1].values - cluster_center, axis=1)
    
    # Find the index of the dataset with the minimum distance to the cluster center
    min_distance_index = np.argmin(distances)
    
    # Get the representative dataset's index in the original dataframe
    representative_index = cluster_data.index[min_distance_index]
    
    # Store the representative dataset
    representative_datasets_indices.append(representative_index)
    
print(representative_datasets_indices)

representative_datasets = file['dataset'].iloc[representative_datasets_indices]
print(representative_datasets)
