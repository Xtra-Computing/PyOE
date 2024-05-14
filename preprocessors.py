import torch
from sklearn.impute import IterativeImputer, KNNImputer

def fill_missing_value(window_x, missing_fill):
    if missing_fill.startswith("knn"):
        num = eval(missing_fill[3:])
        imp = KNNImputer(n_neighbors=num, weights="uniform", keep_empty_features=True)
        filled = imp.fit_transform(window_x.numpy())
        return torch.tensor(filled)
    elif missing_fill == "regression":
        imp = IterativeImputer(keep_empty_features=True)
        filled = imp.fit_transform(window_x.numpy())
        return torch.tensor(filled)
    elif missing_fill == "avg":
        column_means = torch.mean(window_x, dim=0)
        column_means = torch.nan_to_num(column_means, nan=0.0)
        nan_mask = torch.isnan(window_x)
        filled = torch.where(nan_mask, column_means, window_x)
        return filled
    elif missing_fill == "zero": 
        filled = torch.nan_to_num(window_x, nan=0.0)
        return filled

class Preprocessor:
    def __init__(self, missing_fill="zero"):
        self.missing_fill = missing_fill
    
    def fill(self, x):
        if torch.isnan(x).any():
            return fill_missing_value(x, self.missing_fill)
        else:
            return x