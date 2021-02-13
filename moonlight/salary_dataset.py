import requests
from bs4 import BeautifulSoup

from moonlight.dataset import Dataset


class LeagueSalaryDataset(Dataset):

    def __init__(self, league_id: int = 1203):
        super().__init__()
        self.league_id = league_id
        self.primary_keys = ["playerid", "teamname"]
        self.col_renamer = {"FG MajorLeagueID": "playerid", "Team Name": "teamname"}
        self.input_cols = ["Salary", "Position(s)"]
        self.csv_path = f"https://ottoneu.fangraphs.com/{self.league_id}/rosterexport?csv=1"
        self.settings_url = f"https://ottoneu.fangraphs.com/{self.league_id}/settings"

    def build_dataset(self):
        self._read_csv()
        self.df = self.df[self.input_cols]
        self._remove_restricted_list()
        self._strip_salary()
        self._position_indicators()
        self._add_n_teams()
        self._add_format()
        return self.df

    def _remove_restricted_list(self):
        self.df = self.df.loc[self.df.index.get_level_values("teamname") != "Restricted List"]

    def _strip_salary(self):
        self.df["Salary"] = self.df["Salary"].str[1:].astype(int)

    def _position_indicators(self):
        self.df["Position(s)"] = self.df["Position(s)"].str.split("/")
        self.df = self.df.explode("Position(s)")
        self.df["Indicator"] = 1
        self.df = self.df.pivot_table(index=["playerid", "teamname", "Salary"], columns="Position(s)",
                                      values="Indicator")
        # if "Util" in self.df.columns:
        #     self.df = self.df.drop(columns=["Util"])
        self.df = self.df.fillna(0)
        self.df = self.df.div(self.df.sum(axis=1), axis=0)
        self.df = self.df.reset_index()
        self.df.columns.name = ""
        self.df = self.df.reset_index(drop=True)
        self.df = self.df.set_index(self.primary_keys)

    def _add_n_teams(self):
        self.df["n_teams"] = len(self.df.index.get_level_values("teamname").unique())

    def _add_format(self):
        soup = BeautifulSoup(requests.get(self.settings_url).content, features="lxml")
        self.df["format"] = soup.find_all("a", attrs={'href': '/scoringoptions'})[0].text
