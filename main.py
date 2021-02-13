import logging
import sys

from moonlight.salary_scraper import SalaryDataset

logging.getLogger().setLevel(logging.INFO)

if __name__ == "__main__":

    sd = SalaryDataset()
    sd.scrape()