from moonlight.dataset import Dataset


class LeagueSalaryDataset(Dataset):

    def __init__(self, league_id: int = 1203):
        super().__init__()
        self.league_id = league_id
        self.primary_keys = ["playerid", "teamname"]
        self.col_renamer = {"FG MajorLeagueID": "playerid", "Team Name": "teamname"}
        self.input_cols = ["Salary", "Position(s)"]
        self.csv_path = f"https://ottoneu.fangraphs.com/{self.league_id}/rosterexport?csv=1"

    def build_dataset(self):
        self._read_csv()
        self.df = self.df[self.input_cols]
        self._strip_salary()
        self._position_indicators()
        self._add_n_teams()
        return self.df

    def _strip_salary(self):
        self.df["Salary"] = self.df["Salary"].str[1:].astype(int)

    def _position_indicators(self):
        self.df["Position(s)"] = self.df["Position(s)"].str.split("/")
        self.df = self.df.explode("Position(s)")
        self.df["Indicator"] = 1
        self.df = self.df.pivot_table(index=["playerid", "teamname", "Salary"], columns="Position(s)",
                                      values="Indicator")
        self.df = self.df.drop(columns=["Util"])
        self.df = self.df.fillna(0)
        self.df = self.df.div(self.df.sum(axis=1), axis=0)
        self.df = self.df.reset_index()
        self.df.columns.name = ""
        self.df = self.df.reset_index(drop=True)
        self.df = self.df.set_index("playerid")

    def _add_n_teams(self):
        self.df["n_teams"] = len(self.df.index.get_level_values("teamname").unique())
