import pandas as pd
import numpy as np
from typing import *

from moonlight.dataset import Dataset


class ProjectionDataset(Dataset):

    def __init__(self):
        super().__init__()
        self.primary_keys = ["mlbamid"]
        self.meta_cols = ["firstname", "lastname", "age"]
        self.points_col = "fg_points"
        self.query_cols: Optional[List[str]] = None
        self.position_cols: Optional[List[str]] = None

        self.input_cols: Optional[List[str]] = None

    def build_dataset(self):
        self._read_csv()
        self.df = self.df[self.meta_cols + self.query_cols]
        self._derive_positions()
        self._compute_fg_points()
        # return self.df[self.meta_cols + [self.points_col]].sort_values(self.points_col, ascending=False)
        return self.df.sort_values(self.points_col, ascending=False)[self.meta_cols + [self.points_col] +
                                                                     self.position_cols]

    def _compute_fg_points(self):
        raise NotImplementedError

    def _derive_positions(self):
        raise NotImplementedError


class BatterProjectionDataset(ProjectionDataset):

    def __init__(self, season: int = 2022):
        super().__init__()
        self.csv_path = f"~/Dropbox (Personal)/baseball/steamer_hitters_{season}.csv"
        self.g_proj_cols = ["gC", "g1B", "g2B", "g3B", "gSS", "gLF", "gCF", "gRF", "gDH"]
        self.query_cols = ["AB", "H", "2B", "3B", "HR", "BB", "HBP", "SB", "CS"] + self.g_proj_cols
        self.position_cols = ["pC", "p1B", "p2B", "p3B", "pSS", "pOF", "pDH"]

        self.eligible_thres = 10

    def _compute_fg_points(self):
        self.df[self.points_col] = -1*self.df["AB"] + 5.6*self.df["H"] + 2.9*self.df["2B"] + 5.7*self.df["3B"] + \
                                   9.4*self.df["HR"] + 3.0*self.df["BB"] + 3.0*self.df["HBP"] + 1.9*self.df["SB"] - \
                                   2.8*self.df["CS"]

    def _derive_positions(self):
        self.df["gOF"] = self.df["gLF"] + self.df["gCF"] + self.df["gRF"]
        self.g_proj_cols += ["gOF"]
        self.df[self.g_proj_cols] = (self.df[self.g_proj_cols] >= self.eligible_thres) * 1
        self.df = self.df.rename(columns={x: f"p{x[1:]}" for x in self.g_proj_cols})
        self.df[self.position_cols] = self.df[self.position_cols].div(self.df[self.position_cols].sum(axis=1), axis=0)


class PitcherProjectionDataset(ProjectionDataset):

    def __init__(self, season: int = 2022):
        super().__init__()
        self.csv_path = f"~/Dropbox (Personal)/baseball/steamer_pitchers_{season}.csv"
        self.query_cols = ["IP", "K", "H", "BB", "HR", "SV", "HLD"] + ["start_percent"]
        self.position_cols = ["pSP", "pRP"]
        self.sp_threshold = .5

    def _compute_fg_points(self):
        self.df[self.points_col] = 7.4*self.df["IP"] + 2*self.df["K"] - 2.6*self.df["H"] - 3*self.df["BB"] - \
                                   12.3*self.df["HR"] + 5*self.df["SV"] + 4*self.df["HLD"]

    def _derive_positions(self):
        self.df["pRP"] = np.where(self.df["start_percent"] < .5, 1, 0)
        self.df["pSP"] = np.where(self.df["start_percent"] >= .5, 1, 0)


class PlayerProjectionDataset(Dataset):

    def __init__(self):
        super().__init__()
        self.ppd = PitcherProjectionDataset()
        self.bpd = BatterProjectionDataset()
        self.position_cols = self.ppd.position_cols + self.bpd.position_cols
        self.points_col = self.ppd.points_col

    def build_dataset(self):
        pitchers = self.ppd.build_dataset()
        batters = self.bpd.build_dataset()
        self.df = pd.concat([pitchers, batters])
        self.df[self.position_cols] = self.df[self.position_cols].fillna(0)
        self.df = self.df.loc[self.df[self.position_cols].sum(axis=1) > 0]
        return self.df.sort_values(self.points_col, ascending=False)