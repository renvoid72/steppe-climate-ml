import gzip
import json
from contextlib import redirect_stdout
from io import StringIO
from pathlib import Path
from shutil import copyfile, copyfileobj
from tempfile import TemporaryDirectory
from unittest import TestCase
from unittest.mock import patch

import numpy as np
import pandas as pd
import xarray as xr

from netCDF4 import Dataset

from prepare_cru import Region, numeric_differences, prepare_cru
from kursach011_modules.cru_loader import LEGACY_NETCDF_FILENAMES


class CruPreparationTest(TestCase):
    def test_download_slice_verify_and_reuse(self):
        self.enterContext(redirect_stdout(StringIO()))
        base_dir = Path(self.enterContext(TemporaryDirectory()))
        reference_dir = base_dir / 'cru ceda' / '4.06'
        reference_dir.mkdir(parents=True)
        output_dir = base_dir / 'cru ceda' / '4.06-reproduced'
        coords_dir = base_dir / 'meteo_ru'
        coords_dir.mkdir()
        (coords_dir / 'stations_coords.txt').write_text(
            'WMO Name Lat Lon\n37054 "Mineral Water" 44.15 43.15\n37058 Georgievsk 44.29 43.37\n',
            encoding='utf-8',
        )

        dates = pd.to_datetime(['2000-01-16', '2000-02-15', '2000-03-16'])
        for variable, units, offset in (('dtr', 'degrees Celsius', 0), ('pre', 'mm/month', 50)):
            data = np.arange(48, dtype='float32').reshape(3, 4, 4) + offset
            counts = np.full(data.shape, 2, dtype='int32')
            data[1, 1, 0] = -9999
            counts[1, 1, 0] = -999
            dimensions = ('time', 'lat', 'lon')
            grid = xr.Dataset(
                {
                    variable: (dimensions, data, {'units': units, '_FillValue': np.float32(-9999)}),
                    'stn': (dimensions, counts, {'_FillValue': np.int32(-999)}),
                },
                coords={
                    'time': dates,
                    'lat': [43.75, 44.25, 44.75, 45.25],
                    'lon': [42.75, 43.25, 43.75, 44.25],
                },
            )
            netcdf = base_dir / f'{variable}.nc'
            grid.to_netcdf(netcdf)
            with netcdf.open('rb') as source, gzip.open(base_dir / f'{variable}.gz', 'wb') as target:
                copyfileobj(source, target)
            reference = reference_dir / LEGACY_NETCDF_FILENAMES[variable]
            grid.isel(time=slice(2), lat=slice(3), lon=slice(3)).to_netcdf(reference)
            with Dataset(reference, 'r+') as dataset:
                dataset['time'].units = 'days since 2000-1-16'

        (reference_dir / 'cru_aligned_data.csv').write_text(
            'WMO,Year,Month,DTR,Precip\n'
            '37054,2000,1,5,55\n37054,2000,2,21,71\n'
            '37058,2000,1,5,55\n37058,2000,2,21,71\n',
            encoding='utf-8',
        )

        def fetch_fixture(url, destination):
            variable = url.split('/')[-2]
            copyfile(base_dir / f'{variable}.gz', destination)

        region = Region(43.75, 44.75, 42.75, 43.75, '2000-01', '2000-02')
        with patch('prepare_cru.download_archive', side_effect=fetch_fixture) as download:
            self.assertTrue(prepare_cru(base_dir, verify=True, region=region))
            self.assertEqual(download.call_count, 2)
            expected_csv = (output_dir / 'cru_aligned_data.csv').read_bytes()
            self.assertTrue(prepare_cru(base_dir, verify=True, region=region))
            self.assertEqual(download.call_count, 2)
            (output_dir / 'pre.nc').unlink()
            self.assertTrue(prepare_cru(base_dir, verify=True, region=region))
            self.assertEqual(download.call_count, 2)
            self.assertEqual((output_dir / 'cru_aligned_data.csv').read_bytes(), expected_csv)

        mapping = pd.read_csv(output_dir / 'station_grid.csv')
        self.assertEqual(mapping['WMO'].tolist(), [37054, 37058])
        self.assertEqual(mapping['GridLatitude'].tolist(), [44.25, 44.25])
        self.assertEqual(mapping['GridLongitude'].tolist(), [43.25, 43.25])
        report = json.loads((output_dir / 'verification.json').read_text(encoding='utf-8'))
        self.assertEqual(report['station_csv']['common_keys'], 4)
        self.assertEqual(report['station_grid']['unique_grid_cells'], 1)
        dtr_report = report['netcdf']['dtr']
        self.assertNotEqual(dtr_report['time_units']['reference'], dtr_report['time_units']['result'])
        self.assertTrue(dtr_report['units_match']['time'])

    def test_missing_masks_and_different_values_are_counted_separately(self):
        differences = numeric_differences([np.nan, 2, 5, 7], [1, 3, np.nan, 4])
        self.assertFalse(differences['matches'])
        self.assertEqual(differences['missing_mask_differences'], 2)
        self.assertEqual(differences['value_differences'], 2)
        self.assertEqual(differences['max_abs_difference'], 3)
