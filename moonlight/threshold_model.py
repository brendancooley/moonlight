import pandas as pd

from moonlight.salary_scraper import SalaryDataset
from moonlight.salary_dataset import LeagueSalaryDataset


class ThresholdModel:

    def __init__(self):
        self.sd = SalaryDataset()
        self.data = self.sd.load_from_csv()
