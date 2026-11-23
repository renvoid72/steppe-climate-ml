"""Подготовка региональных данных CRU TS из официальных архивов."""

import argparse
from dataclasses import asdict, dataclass
import gzip
import json
from pathlib import Path
from shutil import copyfileobj
from tempfile import TemporaryDirectory
from urllib.request import urlopen

import numpy as np
import pandas as pd
import xarray as xr

from kursach011_modules.cru_loader import CruDataLoader
from kursach011_modules.project_settings import CRU_VERSION, cru_csv_path, cru_data_dir


ARCHIVE_ROOT = f'https://crudata.uea.ac.uk/cru/data/hrg/cru_ts_{CRU_VERSION}/cruts.2604091129.v{CRU_VERSION}'
VARIABLE_UNITS = {'dtr': 'degrees Celsius', 'pre': 'mm/month'}
SOURCE_URLS = {
    variable: f'{ARCHIVE_ROOT}/{variable}/cru_ts{CRU_VERSION}.1901.2025.{variable}.dat.nc.gz'
    for variable in VARIABLE_UNITS
}


@dataclass(frozen=True)
class Region:
    south: float
    north: float
    west: float
    east: float
    first_month: str
    last_month: str


REGION = Region(41.25, 59.75, 28.25, 124.75, '1901-01', '2025-12')


def write_json(path, data):
    with TemporaryDirectory(dir=path.parent) as directory:
        staging = Path(directory) / path.name
        staging.write_text(json.dumps(data, ensure_ascii=False, indent=2) + '\n', encoding='utf-8')
        staging.replace(path)


def download_archive(url, destination):
    print(f'Скачивание: {url}', flush=True)
    with urlopen(url, timeout=60) as response, destination.open('wb') as output:
        expected_size = response.headers.get('Content-Length')
        copyfileobj(response, output, length=1024 * 1024)
        if expected_size is not None and output.tell() != int(expected_size):
            raise OSError(f'Скачан неполный архив: {url}')


def validate_region(dataset, variable, region):
    expected_axes = {
        'lat': np.arange(region.south, region.north + 0.25, 0.5),
        'lon': np.arange(region.west, region.east + 0.25, 0.5),
        'time': pd.period_range(region.first_month, region.last_month, freq='M'),
    }
    for axis, expected in expected_axes.items():
        actual = dataset[axis].to_index()
        if axis == 'time':
            actual = actual.to_period('M')
        if not np.array_equal(actual, expected):
            raise ValueError(f'{variable}: ось {axis} выходит за ожидаемые параметры среза.')
    for name in (variable, 'stn'):
        if name not in dataset.data_vars or dataset[name].dims != ('time', 'lat', 'lon'):
            raise ValueError(f'{variable}: отсутствует трёхмерная переменная {name}.')
    if dataset[variable].attrs.get('units') != VARIABLE_UNITS[variable]:
        raise ValueError(f'{variable}: неожиданные единицы {dataset[variable].attrs.get("units")}.')


def prepare_netcdf(output_dir, variable, url, region):
    regional = output_dir / f'{variable}.nc'
    if regional.is_file():
        with xr.open_dataset(regional) as dataset:
            validate_region(dataset, variable, region)
        print(f'Региональный NetCDF из кэша: {regional.name}', flush=True)
        return

    archive = output_dir / 'archives' / url.rsplit('/', 1)[-1]
    archive.parent.mkdir(exist_ok=True)
    with TemporaryDirectory(dir=output_dir) as directory:
        staging = Path(directory)
        incoming = staging / archive.name
        cached = archive.is_file()
        if cached:
            print(f'Архив из кэша: {archive.name}', flush=True)
        else:
            download_archive(url, incoming)

        print(f'Подготовка регионального NetCDF: {regional.name}', flush=True)
        expanded = staging / 'global.nc'
        with gzip.open(archive if cached else incoming, 'rb') as source, expanded.open('wb') as target:
            copyfileobj(source, target, length=1024 * 1024)

        with xr.open_dataset(expanded, decode_times=False, mask_and_scale=False) as dataset:
            months = xr.decode_cf(dataset[['time']]).time.to_index().to_period('M')
            start = months.searchsorted(pd.Period(region.first_month, freq='M'))
            stop = months.searchsorted(pd.Period(region.last_month, freq='M'), side='right')
            subset = dataset[[variable, 'stn']].sel(
                lat=slice(region.south, region.north),
                lon=slice(region.west, region.east),
            ).isel(time=slice(start, stop))
            validate_region(xr.decode_cf(subset, mask_and_scale=False), variable, region)
            encoding = {
                name: {'zlib': True, 'complevel': 4}
                for name in (variable, 'stn')
            }
            for axis in ('lat', 'lon', 'time'):
                if '_FillValue' not in subset[axis].attrs:
                    encoding[axis] = {'_FillValue': None}
            subset.to_netcdf(staging / regional.name, encoding=encoding)

        if not cached:
            incoming.replace(archive)
        (staging / regional.name).replace(regional)


def prepare_cru(base_dir, region=REGION):
    output_dir = cru_data_dir(base_dir)
    loader = CruDataLoader(base_dir, cru_dir=output_dir)
    loader.load_coordinates()

    output_dir.mkdir(parents=True, exist_ok=True)
    for variable, url in SOURCE_URLS.items():
        prepare_netcdf(output_dir, variable, url, region)
    loader.extract_data()
    loader.save_data()

    grid = loader.station_grid
    cell_count = len(grid[['GridLatitude', 'GridLongitude']].drop_duplicates())
    print(f'Станций в области: {len(grid)}, уникальных ячеек: {cell_count}, строк CSV: {len(loader.final_df)}.')
    print(f'Результаты: {output_dir}')
    write_json(output_dir / 'source.json', {
        'version': CRU_VERSION,
        'source_urls': SOURCE_URLS,
        'method': 'HTTPS .dat.nc.gz; local subset',
        'region_cell_centers_and_period': asdict(region),
        'grid_step_degrees': 0.5,
        'command': 'python code/prepare_cru.py',
    })


def ensure_cru_csv(base_dir):
    if not cru_csv_path(base_dir).is_file():
        prepare_cru(base_dir)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.parse_args()
    try:
        prepare_cru(Path(__file__).resolve().parents[1])
    except (OSError, ValueError) as error:
        parser.exit(1, f'Ошибка подготовки CRU: {error}\n')
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
