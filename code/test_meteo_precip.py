import unittest
from contextlib import redirect_stdout
from io import StringIO
from pathlib import Path
from tempfile import TemporaryDirectory
from unittest.mock import patch

import pandas as pd

from kursach011_eof import load_meteo_ru_monthly
from kursach011_modules.meteo_loader import SteppeClimateLoader


class MeteoPrecipitationTest(unittest.TestCase):
    def setUp(self):
        self.base_dir = Path(self.enterContext(TemporaryDirectory()))
        self.loader = SteppeClimateLoader(self.base_dir)
        self.enterContext(redirect_stdout(StringIO()))

    def test_daily_export_distinguishes_zero_from_missing_precipitation(self):
        dates = pd.date_range('2000-01-01', periods=3650, name='time')
        daily = pd.DataFrame(
            {'tmin': -2.0, 'tavg': 0.0, 'tmax': 2.0, 'prcp': 0.0},
            index=dates,
        ).astype({'prcp': 'Float64'})
        daily.loc['2000-01-02', 'prcp'] = pd.NA
        self.loader.target_stations = pd.DataFrame(
            {'name': ['Station'], 'latitude': [52.0], 'longitude': [49.0]},
            index=['27995'],
        )

        with patch('kursach011_modules.meteo_loader.Daily') as daily_source:
            daily_source.return_value.fetch.return_value = daily
            self.loader.step_3_download_data()

        self.loader.step_4_save()
        lines = self.loader.file_data.read_text(encoding='utf-8').splitlines()
        self.assertEqual(len(lines), len(daily))
        self.assertEqual(lines[0].split()[-1], '0.0')
        self.assertEqual(lines[1].split()[-1], '-999.9')

    def test_monthly_totals_keep_missing_precipitation(self):
        self.loader.file_data.write_text(
            '27995 2000 1 1 0 -2.0 0.0 2.0 0.0\n'
            '27995 2000 1 2 0 -2.0 0.0 2.0 0.0\n'
            '27995 2000 2 1 0 -2.0 0.0 2.0 -999.9\n'
            '27995 2000 3 1 0 -2.0 0.0 2.0 0.0\n'
            '27995 2000 3 2 0 -2.0 0.0 2.0 -999.9\n'
            '27995 2000 4 1 0 -2.0 0.0 2.0 1.5\n'
            '27995 2000 4 2 0 -2.0 0.0 2.0 2.5\n',
            encoding='utf-8',
        )
        self.loader.file_coords.write_text(
            'WMO Name Lat Lon\n27995 Station 52.0 49.0\n', encoding='utf-8'
        )

        with patch('kursach011_eof.PROJECT_DIR', self.base_dir):
            monthly = load_meteo_ru_monthly().set_index('Month')

        self.assertEqual(monthly.loc[1, 'Precip'], 0.0)
        self.assertTrue(pd.isna(monthly.loc[2, 'Precip']))
        self.assertTrue(pd.isna(monthly.loc[3, 'Precip']))
        self.assertEqual(monthly.loc[4, 'Precip'], 4.0)
        self.assertEqual(monthly['DTR'].tolist(), [4.0] * 4)
