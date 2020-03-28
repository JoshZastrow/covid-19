
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


d = download_data()
