import requests
import pandas as pd
import logging
import datetime

from moonlight.dataset import Dataset
from moonlight.salary_dataset import LeagueSalaryDataset


class SalaryDataset(Dataset):

    def __init__(self):
        super().__init__()
        self.min_league_id = 1
        self.max_league_id = 1500
        self.today = datetime.datetime.today().strftime('%m-%d-%Y')

        self.positions = ["SP", "RP", "C", "1B", "2B", "3B", "SS", "OF", "Util"]
        self.cols = ["Salary", "format", "n_teams"] + self.positions
        self.primary_keys = ["playerid", "teamname"]
        self.formats = ["FanGraphs Points", "H2H FanGraphs Points"]
        self.league_ids = []
        self.output_path = f"~/Dropbox (Princeton)/Public/ottoneu/salaries_{self.today}.csv"

    def scrape(self):
        self._collect_ids()
        self.df = pd.concat([self._get_league_data(league_id) for league_id in self.league_ids])
        self.df.to_csv(self.output_path)

    def load_from_csv(self):
        self.df = pd.read_csv(self.output_path).set_index(self.primary_keys)[self.cols]
        self._filter_format()
        self._fill_positions()
        return self.df

    def _collect_ids(self):
        for league_id in range(self.min_league_id, self.max_league_id+1):
            request = requests.get(f"https://ottoneu.fangraphs.com/{league_id}/rosterexport?csv=1")
            if len(request.history) == 0:
                logging.info(f"Found roster export path for league id {league_id}")
                self.league_ids.append(league_id)

    @staticmethod
    def _get_league_data(league_id: int):
        lsd = LeagueSalaryDataset(league_id=league_id)
        logging.info(f"Building salary dataset for league {league_id}")
        lsd.build_dataset()
        return lsd.df

    def _filter_format(self):
        self.df = self.df.loc[self.df["format"].isin(self.formats)]

    def _fill_positions(self):
        self.df[self.positions] = self.df[self.positions].fillna(0)
        # print((self.df[self.positions].sum(axis=1) == 0).sum())
        # self.df = self.df.loc[self.df[self.positions].sum(axis=1) != 0]