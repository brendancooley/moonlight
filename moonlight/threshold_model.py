import pandas as pd

from moonlight.salary_scraper import SalaryDataset
from moonlight.salary_dataset import LeagueSalaryDataset

# for tobit model: https://github.com/jamesdj/tobit/pulls
# github sourcer: https://github.com/ellisonbg/antipackage


class ThresholdModel:

    def __init__(self):
        self.sd = SalaryDataset()
        self.salaries = self.sd.load_from_csv()
