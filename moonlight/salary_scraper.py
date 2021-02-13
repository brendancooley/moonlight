import requests
import pandas as pd

from moonlight.dataset import Dataset
from moonlight.salary_dataset import LeagueSalaryDataset


class SalaryDataset(Dataset):

    def __init__(self):
        super().__init__()
        self.min_league_id = 1
        self.max_league_id = 2000

        self.league_ids = []

    def scrape(self):
        self._collect_ids()
        self.df = pd.concat([self._get_league_data(league_id) for league_id in self.league_ids])
        # TODO save output

    def _collect_ids(self):
        for league_id in range(self.min_league_id, self.max_league_id+1):
            request = requests.get(f"https://ottoneu.fangraphs.com/{league_id}/rosterexport?csv=1")
            if len(request.history) == 0:
                self.league_ids.append(league_id)

    def _get_league_data(self, league_id: int):
        lsd = LeagueSalaryDataset(league_id=league_id)
        lsd.build_dataset()
        return lsd.df


