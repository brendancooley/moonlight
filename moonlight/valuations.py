import pandas as pd
import numpy as np
from sklearn.linear_model import LogisticRegression, LinearRegression
import scipy.optimize as opt
from typing import *
import datetime

from moonlight.salary_scraper import SalaryDataset
from moonlight.steamer_dataset import PlayerProjectionDataset


class ValuationModel:

    def __init__(self):
        self.projection_ds = PlayerProjectionDataset()
        self.position_cols = self.projection_ds.position_cols
        self.n_team_cols: Optional[List[str]] = None
        self.points_col = self.projection_ds.points_col
        self.data = SalaryDataset().load_from_csv()

        # TODO fit in outer loop
        self.threshold_tol: float = .25  # probably want this lower pre-auction and higher later

        self.tm: Optional[ThresholdModel] = None
        self.threshold_df: Optional[pd.DataFrame] = None
        self.slope: Optional[float] = None
        self.inflation_fct = 1.

        self.today = datetime.datetime.today().strftime('%m-%d-%Y')
        self.output_path = f"~/Dropbox (Princeton)/Public/ottoneu/valuations_{self.today}.csv"

    def build_dataset(self):
        self.solve_thresholds()
        self.compute_thresholds()
        self.solve_slope()
        self.compute_value()
        self.prepare_output()
        self.data.to_csv(self.output_path)

    def solve_thresholds(self):
        self._standardize_data()
        self._fit_threshold_model()
        self._prepare_threshold_product()
        self.threshold_df["threshold"] = opt.fsolve(self._eval_threshold_prob, x0=np.repeat(250, len(self.threshold_df)))
        self.threshold_df = self.threshold_df[self.position_cols + self.n_team_cols + ["threshold"]]

    def compute_thresholds(self):
        self._standardize_data()
        for col in self.n_team_cols:
            thresholds = self.threshold_df.loc[self.threshold_df[col] == 1]
            self.data.loc[self.data[col] == 1, "threshold"] = np.matmul(self.data.loc[self.data[col] == 1, self.position_cols], thresholds["threshold"])

    def solve_slope(self):
        self.data["PAR"] = np.where(self.data[self.points_col] > self.data["threshold"],
                                    self.data[self.points_col] - self.data["threshold"], 0)
        slope_data = self.data.loc[(self.data["PAR"] > 0) & (self.data["Salary"] > 0)]
        sm = SlopeModel(x=np.array(slope_data["PAR"]), y=np.array(slope_data["Salary"]))
        sm.fit()
        self.slope = sm.model.coef_[0] * self.inflation_fct

    def compute_value(self):
        self.data["Value"] = self.data["PAR"] * self.slope

    def prepare_output(self):
        self.data[self.position_cols] = (self.data[self.position_cols] > 0)
        self.data = self.data.drop(columns=self.n_team_cols + ["format", "n_teams"])

    def _standardize_data(self):
        self.data = self.data.loc[self.data["n_teams"].isin([12, 14, 16])]
        self.data = self.data[~self.data.index.duplicated(keep='first')]

    def _prepare_threshold_model_data(self):
        # TODO investigate duplicates
        y = self.data["Salary"] > 0
        n_team_dummies = pd.get_dummies(self.data["n_teams"])
        n_team_dummies = n_team_dummies.rename(columns={x: f"n_teams_{x}" for x in n_team_dummies.columns})
        self.n_team_cols = n_team_dummies.columns.tolist()
        self.data = self.data.merge(n_team_dummies, how='left',
                                    left_index=True, right_index=True)
        x = self.data[[self.points_col] + self.position_cols + self.n_team_cols]
        return x, np.array(y)

    def _prepare_threshold_product(self):
        idx = pd.MultiIndex.from_product([self.position_cols, self.n_team_cols])
        threshold_input = pd.DataFrame(data=pd.get_dummies(idx.to_frame().reset_index(drop=True),
                                                           prefix="", prefix_sep=""))
        self.threshold_df = threshold_input.set_index(idx)

    def _fit_threshold_model(self):
        x, y = self._prepare_threshold_model_data()
        tm = ThresholdModel(x=x, y=y)
        tm.fit()
        self.tm = tm

    def _eval_threshold_prob(self, x):
        threshold_input = self.threshold_df.copy()
        threshold_input[self.points_col] = x
        threshold_input = threshold_input[[self.points_col] + self.position_cols + self.n_team_cols]
        return self.tm.model.predict_proba(threshold_input)[:, 1] - self.threshold_tol


class ThresholdModel:

    def __init__(self, x: pd.DataFrame, y: np.ndarray):
        self.x = x
        self.y = y

        self.model = LogisticRegression(penalty='none')

    def fit(self):
        self.model.fit(X=self.x, y=self.y)


class SlopeModel:

    def __init__(self, x: np.ndarray, y: np.ndarray):
        self.x = x.reshape(-1, 1)
        self.y = y

        self.model = LinearRegression(fit_intercept=False)

    def fit(self):
        self.model.fit(X=self.x, y=self.y)
