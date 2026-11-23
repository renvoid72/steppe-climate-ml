"""Рабочая версия CRU и общие годы расчётов."""

CRU_VERSION = '4.10'
ANALYSIS_YEAR = 2024
VALIDATION_START_YEAR = 1990
VALIDATION_END_YEAR = 2020
VALIDATION_PERIOD = f'{VALIDATION_START_YEAR}-{VALIDATION_END_YEAR}'
ALIGNED_FILENAME = 'cru_aligned_data.csv'


def cru_data_dir(base_dir):
    return base_dir / 'cru ceda' / CRU_VERSION


def cru_csv_path(base_dir):
    return cru_data_dir(base_dir) / ALIGNED_FILENAME


def results_dir(base_dir):
    return base_dir / 'output011_modules'


def forecast_filename(dataset_name, variable, year=ANALYSIS_YEAR):
    return f'{dataset_name}_{variable}_{year}.csv'


def validation_filename(dataset_name, variable, start_year=VALIDATION_START_YEAR, end_year=VALIDATION_END_YEAR):
    return f'Validation_{dataset_name}_{variable}_{start_year}-{end_year}.csv'
