# -*- coding: utf-8 -*-
"""Извлечение месячных рядов CRU TS для ближайших к станциям ячеек."""

from pathlib import Path
from tempfile import TemporaryDirectory

import numpy as np
import pandas as pd
import xarray as xr

if __package__:
    from .project_settings import ALIGNED_FILENAME, cru_data_dir
else:
    from project_settings import ALIGNED_FILENAME, cru_data_dir

class CruDataLoader:
    def __init__(self, base_dir, cru_dir=None):
        base_dir = Path(base_dir)
        self.coords_file = base_dir / 'meteo_ru' / 'stations_coords.txt'
        cru_dir = cru_data_dir(base_dir) if cru_dir is None else Path(cru_dir)
        self.mapping_file = cru_dir / 'station_grid.csv'
        self.cru_dtr_path = cru_dir / 'dtr.nc'
        self.cru_pre_path = cru_dir / 'pre.nc'
        self.out_file = cru_dir / ALIGNED_FILENAME

    def load_coordinates(self):
        if not self.coords_file.is_file():
            raise FileNotFoundError(
                f'Файл координат не найден: {self.coords_file}. Сначала запустите Meteo Loader.'
            )
        self.stations_df = pd.read_csv(self.coords_file, sep=r'\s+')
        if self.stations_df.empty or self.stations_df['WMO'].isna().any():
            raise ValueError('В файле координат должны быть станции с заполненным WMO.')
        if self.stations_df['WMO'].duplicated().any():
            raise ValueError('В файле координат есть повторяющиеся WMO.')

    def extract_data(self):
        stations = self.stations_df
        with xr.open_dataset(self.cru_dtr_path) as dtr_ds, xr.open_dataset(self.cru_pre_path) as pre_ds:
            for axis in ('lat', 'lon', 'time'):
                if not np.array_equal(dtr_ds[axis].values, pre_ds[axis].values):
                    raise ValueError(f'Координата {axis} различается между NetCDF DTR и осадков.')

            in_region = (
                stations['Lat'].between(float(dtr_ds.lat.min()), float(dtr_ds.lat.max()))
                & stations['Lon'].between(float(dtr_ds.lon.min()), float(dtr_ds.lon.max()))
            )
            if not in_region.all():
                outside = ', '.join(stations.loc[~in_region, 'WMO'].astype(str))
                raise ValueError(f'Отсутствуют координаты или станция вне региона CRU: {outside}')

            selector = {
                'lat': xr.DataArray(stations['Lat'].to_numpy(), dims='station'),
                'lon': xr.DataArray(stations['Lon'].to_numpy(), dims='station'),
            }
            dtr = dtr_ds.get('dtr', dtr_ds[next(iter(dtr_ds.data_vars))])
            pre = pre_ds.get('pre', pre_ds[next(iter(pre_ds.data_vars))])
            dtr = dtr.load().sel(selector, method='nearest').transpose('station', 'time')
            pre = pre.load().sel(selector, method='nearest').transpose('station', 'time')

        self.station_grid = stations[['WMO', 'Lat', 'Lon']].rename(
            columns={'Lat': 'Latitude', 'Lon': 'Longitude'}
        ).assign(GridLatitude=dtr.lat.values, GridLongitude=dtr.lon.values)
        self.station_grid = self.station_grid.sort_values('WMO')

        dates = pd.DatetimeIndex(dtr.time.values)
        self.final_df = pd.DataFrame({
            'Date': np.tile(dates, len(stations)),
            'DTR': dtr.values.ravel(),
            'Precip': pre.values.ravel(),
            'WMO': np.repeat(stations['WMO'].to_numpy(), len(dates)),
            'Latitude': np.repeat(stations['Lat'].to_numpy(), len(dates)),
            'Longitude': np.repeat(stations['Lon'].to_numpy(), len(dates)),
            'Year': np.tile(dates.year, len(stations)),
            'Month': np.tile(dates.month, len(stations)),
        }).dropna(subset=['DTR', 'Precip']).sort_values(['WMO', 'Year', 'Month'])

    def save_data(self):
        if self.final_df.empty:
            raise ValueError('Для указанных станций нет месяцев с DTR и осадками.')
        with TemporaryDirectory(dir=self.out_file.parent) as staging:
            staging = Path(staging)
            csv_path = staging / self.out_file.name
            self.final_df.to_csv(csv_path, index=False)
            grid_path = staging / self.mapping_file.name
            self.station_grid.to_csv(grid_path, index=False)
            grid_path.replace(self.mapping_file)
            csv_path.replace(self.out_file)

    def run(self):
        self.load_coordinates()
        self.extract_data()
        self.save_data()
        cells = self.station_grid[['GridLatitude', 'GridLongitude']].drop_duplicates()
        print(f'CRU: {len(self.station_grid)} станций, {len(cells)} ячеек, {len(self.final_df)} строк.')
        print(f'Станционный CSV: {self.out_file}')


if __name__ == '__main__':
    CruDataLoader(Path(__file__).resolve().parents[2]).run()
