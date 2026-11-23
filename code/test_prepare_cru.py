import gzip
import json
import os
from contextlib import redirect_stdout
from io import StringIO
from pathlib import Path
from shutil import copyfile, copyfileobj
from tempfile import TemporaryDirectory
from unittest import TestCase, skipUnless
from unittest.mock import patch

import numpy as np
import pandas as pd
import xarray as xr

from kursach011_modules.project_settings import cru_csv_path
from prepare_cru import Region, ensure_cru_csv, prepare_cru


class CruPreparationTest(TestCase):
    def test_download_slice_and_reuse(self):
        self.enterContext(redirect_stdout(StringIO()))
        base_dir = Path(self.enterContext(TemporaryDirectory()))
        csv_path = cru_csv_path(base_dir)
        output_dir = csv_path.parent
        coords_dir = base_dir / 'meteo_ru'
        coords_dir.mkdir()
        (coords_dir / 'stations_coords.txt').write_text(
            'WMO Name Lat Lon\n37054 "Mineral Water" 44.15 43.15\n37058 Georgievsk 44.29 43.37\n',
            encoding='utf-8',
        )

        dates = pd.to_datetime(['2000-01-16', '2000-02-15', '2000-03-16'])
        expected_grids = {}
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
            expected_grids[variable] = xr.decode_cf(grid.isel(time=slice(2), lat=slice(3), lon=slice(3)))

        def fetch_fixture(url, destination):
            variable = url.split('/')[-2]
            self.assertTrue(url.endswith(f'/cru_ts4.10.1901.2025.{variable}.dat.nc.gz'))
            copyfile(base_dir / f'{variable}.gz', destination)

        region = Region(43.75, 44.75, 42.75, 43.75, '2000-01', '2000-02')
        with patch('prepare_cru.download_archive', side_effect=fetch_fixture) as download:
            prepare_cru(base_dir, region=region)
            self.assertEqual(download.call_count, 2)
            csv_bytes = csv_path.read_bytes()

            prepare_cru(base_dir, region=region)
            self.assertEqual(download.call_count, 2)
            self.assertEqual(csv_path.read_bytes(), csv_bytes)

            (output_dir / 'pre.nc').unlink()
            prepare_cru(base_dir, region=region)
            self.assertEqual(download.call_count, 2)
            self.assertEqual(csv_path.read_bytes(), csv_bytes)

        self.assertEqual(output_dir.name, '4.10')
        source = json.loads((output_dir / 'source.json').read_text(encoding='utf-8'))
        self.assertEqual(source['version'], '4.10')
        for variable, expected in expected_grids.items():
            with self.subTest(variable=variable), xr.open_dataset(output_dir / f'{variable}.nc') as subset:
                xr.testing.assert_equal(subset, expected)

        monthly = pd.read_csv(csv_path)
        self.assertEqual(
            list(monthly[['WMO', 'Year', 'Month', 'DTR', 'Precip']].itertuples(index=False, name=None)),
            [(37054, 2000, 1, 5, 55), (37054, 2000, 2, 21, 71),
             (37058, 2000, 1, 5, 55), (37058, 2000, 2, 21, 71)],
        )
        mapping = pd.read_csv(output_dir / 'station_grid.csv')
        self.assertEqual(mapping['WMO'].tolist(), [37054, 37058])
        self.assertEqual(mapping['GridLatitude'].tolist(), [44.25, 44.25])
        self.assertEqual(mapping['GridLongitude'].tolist(), [43.25, 43.25])

    def test_ensure_cru_csv_prepares_only_when_active_file_is_missing(self):
        base_dir = Path(self.enterContext(TemporaryDirectory()))
        with patch('prepare_cru.prepare_cru') as prepare:
            ensure_cru_csv(base_dir)
            prepare.assert_called_once_with(base_dir)
            path = cru_csv_path(base_dir)
            path.parent.mkdir(parents=True)
            path.touch()
            ensure_cru_csv(base_dir)
            prepare.assert_called_once_with(base_dir)


@skipUnless(os.environ.get('STEPPE_CHECK_LOCAL_CRU') == '1', 'Требует локальные архивы и STEPPE_CHECK_LOCAL_CRU=1')
class CruLocalComparisonTest(TestCase):
    project = Path(__file__).resolve().parents[1]
    reference_dir = project / 'cru ceda' / '4.06'
    result_dir = cru_csv_path(project).parent
    reference_files = {'dtr': 'temp dtr full rus stepp.nc', 'pre': 'precip full rus stepp.nc'}

    def check_values(self, label, reference, result):
        valid = ~np.isnan(reference)
        np.testing.assert_array_equal(valid, ~np.isnan(result))
        a, b = reference[valid], result[valid]
        self.assertGreater(a.size, 0)
        self.assertTrue(np.isfinite(a).all() and np.isfinite(b).all())
        diff = np.abs(b.astype('float64') - a.astype('float64'))
        print(
            f'{label}: {np.count_nonzero(diff):,} различий из {diff.size:,}. '
            f'Средняя {diff.mean():.3f}, 95-й перцентиль {np.quantile(diff, 0.95):.3f}, '
            f'максимум {diff.max():.3f}.',
            flush=True,
        )

    def test_compare_regional_netcdf(self):
        for variable, filename in self.reference_files.items():
            with (
                self.subTest(variable=variable),
                xr.open_dataset(self.reference_dir / filename) as reference,
                xr.open_dataset(self.result_dir / f'{variable}.nc') as result,
            ):
                common = result.sel(time=slice('1901-01', '2020-12'))
                self.assertEqual(reference[variable].shape, (1440, 38, 194))
                for axis in ('time', 'lat', 'lon'):
                    np.testing.assert_array_equal(reference[axis].values, common[axis].values)
                for name in (variable, 'stn'):
                    self.assertEqual(reference[name].attrs.get('units'), common[name].attrs.get('units'))
                    self.check_values(f'NetCDF {variable}/{name}', reference[name].values, common[name].values)

    def test_compare_station_csv(self):
        keys = ['WMO', 'Year', 'Month']
        old_csv = pd.read_csv(self.reference_dir / 'cru_aligned_data.csv', float_precision='round_trip')
        new_csv = pd.read_csv(cru_csv_path(self.project), float_precision='round_trip')
        reference = old_csv.set_index(keys, verify_integrity=True).sort_index()
        result = new_csv.loc[new_csv['Year'].le(2020)].set_index(keys, verify_integrity=True).sort_index()
        self.assertEqual(len(reference), 228960)
        pd.testing.assert_index_equal(reference.index, result.index)
        columns = ['Date', 'Latitude', 'Longitude']
        pd.testing.assert_frame_equal(reference[columns], result[columns])

        for variable in ('DTR', 'Precip'):
            self.check_values(f'CSV {variable}', reference[variable].to_numpy(), result[variable].to_numpy())

        stations = old_csv[['WMO', 'Latitude', 'Longitude']].drop_duplicates().sort_values('WMO')
        with xr.open_dataset(self.reference_dir / self.reference_files['dtr']) as reference_grid:
            lat = xr.DataArray(stations['Latitude'].to_numpy(), dims='station')
            lon = xr.DataArray(stations['Longitude'].to_numpy(), dims='station')
            cells = reference_grid[['lat', 'lon']].sel(lat=lat, lon=lon, method='nearest')
            expected = stations.assign(GridLatitude=cells.lat.values, GridLongitude=cells.lon.values)
        mapping = pd.read_csv(self.result_dir / 'station_grid.csv').sort_values('WMO')
        pd.testing.assert_frame_equal(
            expected.reset_index(drop=True), mapping.reset_index(drop=True), check_dtype=False
        )
