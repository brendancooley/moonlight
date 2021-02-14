import requests
import pandas as pd
from bs4 import BeautifulSoup
from typing import *

from moonlight.dataset import Dataset
from moonlight.steamer_dataset import PlayerProjectionDataset


class LeagueSalaryDataset(Dataset):

    def __init__(self, league_id: int = 1203):
        super().__init__()
        self.league_id = league_id
        self.primary_keys = ["mlbamid", "teamname"]
        self.col_renamer = {"FG MajorLeagueID": "fangraphsid", "Team Name": "teamname"}
        self.input_cols = ["fangraphsid", "teamname", "Salary"]
        self.csv_path = f"https://ottoneu.fangraphs.com/{self.league_id}/rosterexport?csv=1"
        self.settings_url = f"https://ottoneu.fangraphs.com/{self.league_id}/settings"

        self.projections: pd.DataFrame = PlayerProjectionDataset().build_dataset()

    def build_dataset(self):
        self._read_csv(set_index=False)
        self.df = self.df[self.input_cols]
        self._remove_restricted_list()
        self._strip_salary()
        self._map_ids()
        self._merge_projections()
        self._add_n_teams()
        self._add_format()
        self.df = self.df.set_index(self.primary_keys)
        return self.df

    def _remove_restricted_list(self):
        self.df = self.df.loc[self.df["teamname"] != "Restricted List"]

    def _strip_salary(self):
        self.df["Salary"] = self.df["Salary"].str[1:].astype(int)

    def _add_n_teams(self):
        self.df["n_teams"] = len(self.df["teamname"].unique()) - 1

    def _add_format(self):
        soup = BeautifulSoup(requests.get(self.settings_url).content, features="lxml")
        self.df["format"] = soup.find_all("a", attrs={'href': '/scoringoptions'})[0].text

    def _map_ids(self):
        id_map = PlayerIDDataset().get_mlbamid_map("fangraphsid")
        self.df = self.df.loc[self.df["fangraphsid"].notna()]
        self.df["fangraphsid"] = self.df["fangraphsid"].astype(int).astype(str)
        self.df["mlbamid"] = self.df["fangraphsid"].map(id_map)
        self.df = self.df.loc[self.df["mlbamid"].notna()]
        self.df["mlbamid"] = self.df["mlbamid"].astype(int)
        self.df = self.df.drop(columns=["fangraphsid"])

    def _merge_projections(self):
        self.df = self.df.set_index("mlbamid")
        self.df = self.projections.merge(self.df, how='left', left_index=True, right_index=True)
        self.df = self.df.sort_values("fg_points", ascending=False)
        self.df = self.df.reset_index()
        self.df["Salary"] = self.df["Salary"].fillna(0)
        self.df["teamname"] = self.df["teamname"].fillna("free_agent")


class PlayerIDDataset(Dataset):

    def __init__(self):
        super().__init__()
        self.primary_keys = ["mlbamid"]
        self.col_renamer = {"MLBID": "mlbamid", "IDFANGRAPHS": "fangraphsid"}
        self.csv_path = "~/Dropbox (Princeton)/baseball/sfbb_playerids.csv"

    def get_mlbamid_map(self, key: str):
        self._read_csv()
        self.df = self.df.loc[self.df.index.notna()]
        self.df.index = self.df.index.astype(int)
        self.df[key] = self.df[key].astype(str)
        return {v: int(k) for k, v in self.df[key].to_dict().items()}