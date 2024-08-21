import torch
from typing import Literal
from sklearn.experimental import enable_iterative_imputer
from sklearn.impute import IterativeImputer, KNNImputer


def fill_missing_value(
    window_x: torch.Tensor, missing_fill: Literal["knn", "regression", "avg", "zero"]
) -> torch.Tensor:
    """
    this function provides a few methods to fill missing values in a dataset

    Args:
        window_x (torch.Tensor): the input tensor with missing values.
        missing_fill (Literal["knn", "regression", "avg", "zero"]):
            the method to fill missing values.
    """
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
    else:
        raise ValueError(f"Filling method {fill_missing_value} is not supported.")


class Preprocessor:
    def __init__(
        self, missing_fill: Literal["knn", "regression", "avg", "zero"] = "zero"
    ):
        """
        Args:
            missing_fill (Literal["knn", "regression", "avg", "zero"]):
                the method to fill missing values.
        """
        self.missing_fill = missing_fill

    def fill(self, x: torch.Tensor) -> torch.Tensor:
        """
        This function fills missing values in a dataset

        Args:
            x (torch.Tensor): the input tensor with missing values.

        Returns:
            out (torch.Tensor): the input tensor with missing values filled.
        """
        if torch.isnan(x).any():
            return fill_missing_value(x, self.missing_fill)
        else:
            return x
