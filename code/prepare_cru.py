"""Подготовка региональных данных CRU TS 4.06 из официальных архивов."""

import argparse
from dataclasses import asdict, dataclass
import gzip
import json
from pathlib import Path
from shutil import copyfileobj
from tempfile import TemporaryDirectory
from urllib.request import urlopen

from netCDF4 import num2date
import numpy as np
import pandas as pd
import xarray as xr

from kursach011_modules.cru_loader import ALIGNED_FILENAME, LEGACY_NETCDF_FILENAMES, CruDataLoader


VERSION = '4.06'
ARCHIVE_ROOT = f'https://crudata.uea.ac.uk/cru/data/hrg/cru_ts_{VERSION}/cruts.2205201912.v{VERSION}'
VARIABLE_UNITS = {'dtr': 'degrees Celsius', 'pre': 'mm/month'}
SOURCE_URLS = {
    variable: f'{ARCHIVE_ROOT}/{variable}/cru_ts{VERSION}.1901.2021.{variable}.dat.nc.gz'
    for variable in VARIABLE_UNITS
}
CSV_KEY = ['WMO', 'Year', 'Month']


@dataclass(frozen=True)
class Region:
    south: float
    north: float
    west: float
    east: float
    first_month: str
    last_month: str


REGION = Region(41.25, 59.75, 28.25, 124.75, '1901-01', '2020-12')


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


def numeric_differences(reference, result):
    reference = np.asarray(reference)
    result = np.asarray(result)
    if reference.shape != result.shape:
        return {
            'reference_shape': list(reference.shape),
            'result_shape': list(result.shape),
            'missing_mask_differences': None,
            'value_differences': None,
            'max_abs_difference': None,
            'matches': False,
        }
    reference_missing = pd.isna(reference)
    result_missing = pd.isna(result)
    both_present = ~(reference_missing | result_missing)
    different = both_present & (reference != result)
    missing_count = int(np.count_nonzero(reference_missing != result_missing))
    different_count = int(np.count_nonzero(different))
    max_abs = None
    if different_count:
        delta = reference[different].astype('float64') - result[different].astype('float64')
        max_abs = float(np.abs(delta).max())
    elif both_present.any():
        max_abs = 0.0
    return {
        'missing_mask_differences': missing_count,
        'value_differences': different_count,
        'max_abs_difference': max_abs,
        'matches': missing_count == different_count == 0,
    }


def compare_netcdf(reference_path, result_path, variable):
    with xr.open_dataset(reference_path) as reference, xr.open_dataset(result_path) as result:
        coordinates = {
            name: np.array_equal(reference[name].values, result[name].values)
            for name in ('lat', 'lon', 'time')
        }
        units = {
            name: reference[name].attrs.get('units') == result[name].attrs.get('units')
            for name in ('lat', 'lon', variable, 'stn')
        }
        reference_units = reference.time.encoding['units']
        result_units = result.time.encoding['units']
        reference_calendar = reference.time.encoding.get('calendar', 'standard')
        result_calendar = result.time.encoding.get('calendar', 'standard')
        # Origin and one unit of elapsed time must agree, even if the date strings differ.
        units['time'] = reference_calendar == result_calendar and np.array_equal(
            num2date([0, 1], reference_units, reference_calendar),
            num2date([0, 1], result_units, result_calendar),
        )
        variables = {
            name: numeric_differences(reference[name].values, result[name].values)
            for name in (variable, 'stn')
        }
    return {
        'coordinates_match': coordinates,
        'units_match': units,
        'time_units': {'reference': reference_units, 'result': result_units},
        'variables': variables,
        'matches': all(coordinates.values()) and all(units.values())
        and all(stats['matches'] for stats in variables.values()),
    }


def compare_csv(reference_path, result_path):
    reference = pd.read_csv(reference_path, float_precision='round_trip').set_index(CSV_KEY)
    result = pd.read_csv(result_path, float_precision='round_trip').set_index(CSV_KEY)
    reference_duplicates = int(reference.index.duplicated().sum())
    result_duplicates = int(result.index.duplicated().sum())
    reference = reference.loc[~reference.index.duplicated()]
    result = result.loc[~result.index.duplicated()]
    common = reference.index.intersection(result.index)
    variables = {
        name: numeric_differences(reference.loc[common, name].values, result.loc[common, name].values)
        for name in ('DTR', 'Precip')
    }
    only_reference = len(reference.index.difference(result.index))
    only_result = len(result.index.difference(reference.index))
    return {
        'reference_unique_keys': len(reference),
        'result_unique_keys': len(result),
        'common_keys': len(common),
        'only_in_reference': only_reference,
        'only_in_result': only_result,
        'reference_duplicate_keys': reference_duplicates,
        'result_duplicate_keys': result_duplicates,
        'variables': variables,
        'matches': reference_duplicates == result_duplicates == only_reference == only_result == 0
        and all(stats['matches'] for stats in variables.values()),
    }


def prepare_cru(base_dir, verify=False, region=REGION):
    cru_dir = base_dir / 'cru ceda'
    output_dir = cru_dir / f'{VERSION}-reproduced'
    reference_dir = cru_dir / VERSION
    loader = CruDataLoader(base_dir, cru_dir=output_dir)
    loader.load_coordinates()
    if verify:
        for filename in (*LEGACY_NETCDF_FILENAMES.values(), ALIGNED_FILENAME):
            if not (reference_dir / filename).is_file():
                raise FileNotFoundError(f'Эталонный файл не найден: {reference_dir / filename}')

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
        'version': VERSION,
        'source_urls': SOURCE_URLS,
        'method': 'HTTPS .dat.nc.gz; local subset',
        'region_cell_centers_and_period': asdict(region),
        'grid_step_degrees': 0.5,
        'command': 'python code/prepare_cru.py',
        'verification_command': 'python code/prepare_cru.py --verify',
    })

    if not verify:
        return True
    nc_report = {
        variable: compare_netcdf(reference_dir / filename, output_dir / f'{variable}.nc', variable)
        for variable, filename in LEGACY_NETCDF_FILENAMES.items()
    }
    csv_report = compare_csv(reference_dir / ALIGNED_FILENAME, loader.out_file)
    report = {
        'netcdf': nc_report,
        'station_csv': csv_report,
        'station_grid': {
            'station_count': len(grid),
            'unique_grid_cells': cell_count,
            '37054_and_37058': grid.loc[
                grid['WMO'].isin([37054, 37058]), ['WMO', 'GridLatitude', 'GridLongitude']
            ].to_dict('records'),
        },
        'matches': csv_report['matches'] and all(check['matches'] for check in nc_report.values()),
    }
    report_path = output_dir / 'verification.json'
    write_json(report_path, report)
    print(json.dumps(report, ensure_ascii=False, indent=2))
    print(f'Отчёт сверки: {report_path}')
    return report['matches']


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--verify', action='store_true', help='Сверить результаты с файлами в cru ceda/4.06.')
    args = parser.parse_args()
    try:
        matches = prepare_cru(Path(__file__).resolve().parents[1], verify=args.verify)
    except (OSError, ValueError) as error:
        parser.exit(1, f'Ошибка подготовки CRU: {error}\n')
    return 0 if matches else 1


if __name__ == '__main__':
    raise SystemExit(main())
