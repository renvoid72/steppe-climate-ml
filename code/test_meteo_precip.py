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

    def export_daily(self, daily):
        self.loader.target_stations = pd.DataFrame(
            {'name': ['Station'], 'latitude': [52.0], 'longitude': [49.0]},
            index=['27995'],
        )
        with patch('kursach011_modules.meteo_loader.Daily') as daily_source:
            daily_source.return_value.fetch.return_value = daily
            self.loader.step_3_download_data()
        self.loader.step_4_save()

    def test_daily_export_distinguishes_zero_from_missing_precipitation(self):
        dates = pd.date_range('2000-01-01', periods=3650, name='time')
        daily = pd.DataFrame(
            {'tmin': -2.0, 'tavg': 0.0, 'tmax': 2.0, 'prcp': 0.0},
            index=dates,
        ).astype({'prcp': 'Float64'})
        daily.loc['2000-01-02', 'prcp'] = pd.NA
        self.export_daily(daily)
        lines = self.loader.file_data.read_text(encoding='utf-8').splitlines()
        self.assertEqual(len(lines), len(daily))
        self.assertEqual(lines[0].split()[-1], '0.0')
        self.assertEqual(lines[1].split()[-1], '-999.9')

    def test_daily_export_keeps_available_indicators_without_tavg(self):
        dates = pd.date_range('2000-01-01', periods=3650, name='time')
        daily = pd.DataFrame(
            index=dates, columns=['tmin', 'tavg', 'tmax', 'prcp'], dtype='Float64'
        )
        daily.loc['2000-01-01', 'prcp'] = 0.0
        daily.loc['2000-02-01', ['tmin', 'tmax']] = [-3.0, 2.0]
        daily.loc['2000-03-01', ['tmin', 'tmax', 'prcp']] = [-4.0, 4.0, 2.5]
        daily.loc['2000-04-01', 'tmin'] = 1.0
        daily.loc['2000-05-01', 'tavg'] = 0.0

        self.export_daily(daily)

        self.assertEqual(
            self.loader.file_data.read_text(encoding='utf-8').splitlines(),
            [
                '27995 2000 1 1 0 -999.9 -999.9 -999.9 0.0',
                '27995 2000 2 1 0 -3.0 -999.9 2.0 -999.9',
                '27995 2000 3 1 0 -4.0 -999.9 4.0 2.5',
                '27995 2000 5 1 0 -999.9 0.0 -999.9 -999.9',
            ],
        )
    def test_monthly_precipitation_requires_all_calendar_days(self):
        dates = pd.date_range('2000-01-01', periods=3651, name='time')
        daily = pd.DataFrame(
            {'tmin': -2.0, 'tavg': 0.0, 'tmax': 2.0, 'prcp': 0.0},
            index=dates,
        ).drop(pd.Timestamp('2000-02-29'))
        daily.loc['2000-03-02', 'prcp'] = None
        daily.loc['2000-04-01':'2000-04-02', 'prcp'] = [1.5, 2.5]
        daily.loc['2000-05', 'prcp'] = None
        daily.loc['2004-02-29', 'prcp'] = 3.0
        self.export_daily(daily)

        with patch('kursach011_eof.PROJECT_DIR', self.base_dir):
            monthly = load_meteo_ru_monthly().set_index(['Year', 'Month'])

        self.assertEqual(monthly.loc[(2000, 1), 'Precip'], 0.0)
        self.assertTrue(pd.isna(monthly.loc[(2000, 2), 'Precip']))
        self.assertEqual(monthly.loc[(2000, 2), 'DTR'], 4.0)
        self.assertTrue(pd.isna(monthly.loc[(2000, 3), 'Precip']))
        self.assertEqual(monthly.loc[(2000, 4), 'Precip'], 4.0)
        self.assertTrue(pd.isna(monthly.loc[(2000, 5), 'Precip']))
        self.assertEqual(monthly.loc[(2004, 2), 'Precip'], 3.0)

    def test_monthly_dtr_limits_total_and_consecutive_gaps(self):
        dates = pd.date_range('2000-01-01', periods=3654, name='time')
        daily = pd.DataFrame(
            {'tmin': -2.0, 'tavg': 0.0, 'tmax': 2.0, 'prcp': 0.0},
            index=dates,
        )
        daily.loc['2000-01-01':'2000-01-03', 'tmin'] = None
        daily.loc[['2000-01-15', '2000-01-31'], 'tmax'] = None
        daily.loc['2000-01-15', 'prcp'] = None
        daily.loc[[f'2000-02-{day:02}' for day in (2, 6, 10, 14, 18, 22)], 'tmin'] = None
        daily.loc['2000-03-28':'2000-03-31', 'tmin'] = None
        daily = daily.drop(pd.date_range('2000-04-10', '2000-04-13'))
        self.export_daily(daily)

        with patch('kursach011_eof.PROJECT_DIR', self.base_dir):
            monthly = load_meteo_ru_monthly().set_index(['Year', 'Month']).loc[2000]

        self.assertEqual(monthly.loc[1, 'DTR'], 4.0)
        self.assertTrue(pd.isna(monthly.loc[1, 'Precip']))
        self.assertTrue(monthly.loc[[2, 3, 4], 'DTR'].isna().all())
        self.assertEqual(monthly.loc[2, 'Precip'], 0.0)
        self.assertEqual(monthly.loc[3, 'Precip'], 0.0)
