
import glob
import subprocess
import pendulum
from dataclasses import dataclass
from zipfile import ZipFile
from os.path import exists, curdir, join, abspath, splitext, basename, dirname
from typing import List

RUN_TIMESTAMP = pendulum.today().format("YYYY-MM-DD")
APP_DIRECTORY = curdir | p(abspath) | p(join, "data")


@dataclass
class Kaggle:
    name: str
    dataset: str
    source_filename: str
    destination_dir: str

    def cli_download_command(self) -> List[str]:
        return [
            "kaggle",
            "datasets",
            "download",
            "-d",
            self.dataset,
            "--file",
            self.source_filename,
            "--path",
            self.destination_dir,
            "--unzip",
        ]

    def get_path(self):
        return join(self.destination_dir, self.source_filename.split("/")[-1])


def download_data():
    directory = join(APP_DIRECTORY, RUN_TIMESTAMP)

    if not exists(directory):
        os.makedirs(directory)

    datasets = [
        Kaggle(
            name="Corona",
            dataset="imdevskp/corona-virus-report",
            source_filename="covid_19_clean_complete.csv",
            destination_dir=directory,
        ),
        Kaggle(
            name="Worldbank",
            dataset="theworldbank/world-development-indicators",
            source_filename="wdi-csv-zip-57-mb-/WDIData.csv",
            destination_dir=directory,
        ),
    ]

    # Worldbank downloads a zip file...
    unzip_files(directory)

    for kaggle_source in datasets:
        subprocess.run(kaggle_source.cli_download_command())

    return {d.name: d.get_path() for d in datasets}


def unzip_files(source_dir):
    files = glob.glob(join(source_dir, "*.zip"))
    for f in files:
        filepath = os.path.splitext(f)[0]  # unzipped path

        if not exists(filepath):
            file = filepath | p(basename)
            path = filepath | p(dirname)

            ZipFile(f).extract(member=file, path=path)


def format_columns(df):
    return (
        df.pipe(lambda df: df.rename(columns={c: c.split("/")[0] for c in df.columns}))
        .pipe(
            lambda df: df.rename(
                columns={c: c.split(" ") | p("_".join) for c in df.columns}
            )
        )
        .pipe(lambda df: df.rename(columns={c: c.upper() for c in df.columns}))
    )

def reshape_worldbank(df):
    indicators = (
        "Air transport, passengers carried",
        "Cause of death, by communicable diseases and maternal, prenatal and nutrition conditions (% of total)",
        "Cause of death, by non-communicable diseases (% of total)",
        "Current health expenditure per capita, PPP (current international $)",
        "Death rate, crude (per 1,000 people)",
        "Diabetes prevalence (% of population ages 20 to 79)",
        "GDP per capita, PPP (current international $)",
        "Hospital beds (per 1,000 people)",
        "Incidence of tuberculosis (per 100,000 people)",
        "International migrant stock, total",
        "International tourism, number of arrivals",
        "International tourism, number of departures",
        "Labor force participation rate, total (% of total population ages 15+) (modeled ILO estimate)",
        "Life expectancy at birth, total (years)",
        "Mortality from CVD, cancer, diabetes or CRD between exact ages 30 and 70 (%)",
        "Mortality rate attributed to household and ambient air pollution, age-standardized (per 100,000 population)",
        "Mortality rate attributed to unsafe water, unsafe sanitation and lack of hygiene (per 100,000 population)",
        "Mortality rate, adult, female (per 1,000 female adults)",
        "Mortality rate, adult, male (per 1,000 male adults)",
        "Number of people spending more than 10% of household consumption or income on out-of-pocket health care expenditure",
        "Number of people spending more than 25% of household consumption or income on out-of-pocket health care expenditure",
        "Nurses and midwives (per 1,000 people)",
        "Out-of-pocket expenditure (% of current health expenditure)",
        "PM2.5 air pollution, population exposed to levels exceeding WHO guideline value (% of total)",
        "People using at least basic sanitation services (% of population)",
        "People using safely managed sanitation services (% of population)",
        "People with basic handwashing facilities including soap and water (% of population)",
        "Physicians (per 1,000 people)",
        "Population ages 15-64 (% of total)",
        "Population ages 65 and above (% of total)",
        "Population density (people per sq. km of land area)",
        "Population in the largest city (% of urban population)",
        "Population in urban agglomerations of more than 1 million (% of total population)",
        "Population, total",
        "Poverty headcount ratio at $3.20 a day (2011 PPP) (% of population)",
        "Prevalence of HIV, total (% of population ages 15-49)",
        "Smoking prevalence, females (% of adults)",
        "Smoking prevalence, males (% of adults)",
        "Survival to age 65, female (% of cohort)",
        "Survival to age 65, male (% of cohort)",
        "Trade (% of GDP)",
        "Tuberculosis case detection rate (%, all forms)",
        "Tuberculosis treatment success rate (% of new cases)",
        "Urban population (% of total)",
    )

    return (
        df
        .pipe(lambda df: df[df["INDICATOR_NAME"].isin(indicators)])
        .drop(["INDICATOR_CODE", "COUNTRY_CODE", "UNNAMED:_63"], axis=1)
        .melt(id_vars=["COUNTRY_NAME", "INDICATOR_NAME"], var_name="Year")
        .replace({"United States": "US"})
    )
