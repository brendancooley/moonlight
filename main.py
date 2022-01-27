import logging
import sys

from moonlight.salary_scraper import SalaryDataset
from moonlight.valuations import ValuationModel

logging.getLogger().setLevel(logging.INFO)

scrape = True
model = True

if __name__ == "__main__":

    if scrape:
        sd = SalaryDataset()
        sd.scrape()

    if model:
        vm = ValuationModel()
        vm.build_dataset()