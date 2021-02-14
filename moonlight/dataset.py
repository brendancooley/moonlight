import pandas as pd
from typing import *


class Dataset:

    def __init__(self):
        self.csv_path: Optional[str] = None
        self.primary_keys: Optional[List[str]] = None
        self.col_renamer: Optional[Dict] = None
        self.df: Optional[pd.DataFrame] = None

    def _read_csv(self, set_index: bool = True):
        if self.col_renamer is None:
            self.col_renamer = {}
        self.df = pd.read_csv(self.csv_path).rename(columns=self.col_renamer)
        if set_index:
            self.df = self.df.set_index(self.primary_keys)
