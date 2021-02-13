import pandas as pd
from typing import *

from moonlight.dataset import Dataset


class ProjectionDataset(Dataset):

    def __init__(self):
        super().__init__()
        self.primary_keys = ["playerid"]
        self.meta_cols = ["Name", "Team"]
        self.points_col = "fg_points"

        self.input_cols: Optional[List[str]] = None

    def build_dataset(self):
        self._read_csv()
        self.df = self.df[self.meta_cols + self.input_cols]
        self._compute_fg_points()
        return self.df[self.meta_cols + [self.points_col]].sort_values(self.points_col, ascending=False)

    def _compute_fg_points(self):
        raise NotImplementedError


class BatterProjectionDataset(ProjectionDataset):

    def __init__(self):
        super().__init__()
        self.csv_path = "https://www.dropbox.com/s/yvn6dh1cgmvtfo5/batters.csv?dl=1"
        self.input_cols = ["AB", "H", "2B", "3B", "HR", "BB", "HBP", "SB", "CS"]

    def _compute_fg_points(self):
        self.df[self.points_col] = -1*self.df["AB"] + 5.6*self.df["H"] + 2.9*self.df["2B"] + 5.7*self.df["3B"] + \
                                   9.4*self.df["HR"] + 3.0*self.df["BB"] + 3.0*self.df["HBP"] + 1.9*self.df["SB"] - \
                                   2.8*self.df["CS"]


class PitcherProjectionDataset(ProjectionDataset):

    def __init__(self):
        super().__init__()
        self.csv_path = "https://www.dropbox.com/s/l4pcsmwn6xy7pem/pitcher.csv?dl=1"
        self.input_cols = ["IP", "SO", "H", "BB", "HR"]

    def _compute_fg_points(self):
        self.df[self.points_col] = 7.4*self.df["IP"] + 2*self.df["SO"] - 2.6*self.df["H"] - 3*self.df["BB"] - \
                                   12.3*self.df["HR"]
