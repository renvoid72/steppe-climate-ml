# -*- coding: utf-8 -*-
"""
Регрессионные тесты календарных лагов, скользящих средних и окон LSTM.

На данных двух станций проверяются пропуск месяца, границы окон и
сохранение строки прогноза. Тесты запускаются без обучения моделей.
"""

import unittest
from importlib.util import find_spec
from tempfile import TemporaryDirectory

import numpy as np
import pandas as pd
from pandas.testing import assert_frame_equal

from kursach011_modules.forecast_engine import BaseForecaster
from kursach011_modules.tuning_lab import create_time_features


class MonthlyFeaturesTest(unittest.TestCase):
    def setUp(self):
        self.source = pd.DataFrame(
            [
                ('A', '2023-03-01', 10),
                ('A', '2023-11-01', 90),
                ('A', '2024-01-01', 12),
                ('A', '2024-03-01', 120),
                ('B', '2024-03-01', 900),
                ('B', '2024-04-01', 1000),
                ('B', '2024-05-01', None),
            ],
            columns=['WMO', 'Date', 'DTR'],
        ).sample(frac=1, random_state=42)
        self.source['Date'] = pd.to_datetime(self.source['Date'])
        self.source['Precip'] = self.source['DTR'] * 2

        output_dir = self.enterContext(TemporaryDirectory())
        self.forecaster = BaseForecaster(output_dir, output_dir, 'test')
        self.engine_features = self.forecaster.create_features(self.source)
        self.features = self.engine_features.set_index(['WMO', 'Date'])

    def test_lags_match_calendar_months(self):
        march = self.features.loc[('A', pd.Timestamp('2024-03-01'))]
        self.assertTrue(pd.isna(march['DTR_lag_1']))
        self.assertEqual(march['DTR_lag_2'], 12)
        self.assertTrue(pd.isna(march['DTR_lag_3']))
        self.assertEqual(march['DTR_lag_12'], 10)
        self.assertEqual(march['Precip_lag_12'], 20)

    def test_rolling_means_use_only_past_calendar_months_at_the_same_station(self):
        march_a = self.features.loc[('A', pd.Timestamp('2024-03-01'))]
        self.assertEqual(march_a['DTR_roll_mean_3'], 12)
        self.assertEqual(march_a['DTR_roll_mean_6'], 51)
        self.assertAlmostEqual(march_a['DTR_roll_mean_12'], (10 + 90 + 12) / 3)

        march_b = self.features.loc[('B', pd.Timestamp('2024-03-01'))]
        weather_features = self.features.filter(regex=r'_(lag|roll_mean)_').columns
        self.assertTrue(march_b[weather_features].isna().all())

        april_b = self.features.loc[('B', pd.Timestamp('2024-04-01'))]
        self.assertEqual(april_b['DTR_lag_1'], 900)
        for window in (3, 6, 12):
            self.assertEqual(april_b[f'DTR_roll_mean_{window}'], 900)

    def test_both_entry_points_preserve_the_forecast_row(self):
        tuning_features = create_time_features(self.source)
        assert_frame_equal(self.engine_features, tuning_features)
        assert_frame_equal(
            tuning_features[self.source.columns], self.source.sort_values(['WMO', 'Date'])
        )

        future = self.features.loc[('B', pd.Timestamp('2024-05-01'))]
        self.assertTrue(future[['DTR', 'Precip', 'DTR_lag_12']].isna().all())
        self.assertEqual(future['DTR_lag_1'], 1000)
        for window in (3, 6, 12):
            self.assertEqual(future[f'DTR_roll_mean_{window}'], 950)

    def test_optional_snow_depth_uses_the_same_calendar(self):
        source = self.source.assign(snow_depth=self.source['DTR'] / 10)
        source.loc[source['Date'] == pd.Timestamp('2024-01-01'), 'snow_depth'] = None
        features = create_time_features(source).set_index(['WMO', 'Date'])
        march = features.loc[('A', pd.Timestamp('2024-03-01'))]

        self.assertTrue(pd.isna(march['snow_depth_lag_2']))
        self.assertEqual(march['snow_depth_lag_12'], 1)
        self.assertTrue(pd.isna(march['snow_depth_roll_mean_3']))
        self.assertEqual(march['snow_depth_roll_mean_6'], 9)


@unittest.skipUnless(find_spec('torch') is not None, 'Для тестов LSTM требуется PyTorch')
class LSTMCalendarWindowsTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        from kursach011_lstm import LSTMFORECASTENGINE

        cls.engine = LSTMFORECASTENGINE(lookback_window=12)

    def setUp(self):
        dates = pd.date_range('2023-01-01', '2025-04-01', freq='MS')
        self.source = pd.DataFrame(
            [
                (wmo, date, dtr, dtr * 2, latitude, 60.0)
                for wmo, offset, latitude in [('A', 10, 45.0), ('B', 100, 50.0)]
                for dtr, date in enumerate(dates, start=offset)
                if not (wmo == 'A' and date == pd.Timestamp('2024-02-01'))
            ],
            columns=['WMO', 'Date', 'DTR', 'Precip', 'Latitude', 'Longitude'],
        ).sample(frac=1, random_state=42)

    def test_lstm_skips_the_gap_and_resumes_after_twelve_months(self):
        _, _, dates, stations, _, _ = self.engine._build_sequences(self.source, 'DTR')
        pd.testing.assert_index_equal(
            pd.DatetimeIndex(dates[stations == 'A']),
            pd.DatetimeIndex(['2024-01-01', '2025-03-01', '2025-04-01']),
        )
        pd.testing.assert_index_equal(
            pd.DatetimeIndex(dates[stations == 'B']),
            pd.date_range('2024-01-01', '2025-04-01', freq='MS'),
        )

    def test_lstm_window_uses_its_station_and_excludes_the_target_month(self):
        target_date = pd.Timestamp('2025-03-01')
        history_dates = pd.date_range('2024-03-01', '2025-02-01', freq='MS')
        observed = self.source.set_index(['WMO', 'Date'])

        for target in ('DTR', 'Precip'):
            windows, values, dates, stations, _, _ = self.engine._build_sequences(self.source, target)
            for station in ('A', 'B'):
                with self.subTest(target=target, station=station):
                    is_target = (dates == target_date) & (stations == station)
                    np.testing.assert_array_equal(
                        values[is_target], [observed.loc[(station, target_date), target]]
                    )

                    history = observed.loc[station].loc[history_dates]
                    expected = history.assign(Precip=np.log1p(history['Precip']))
                    columns = ['DTR', 'Precip'] if target == 'DTR' else ['Precip', 'DTR']
                    np.testing.assert_allclose(
                        windows[is_target, :, :2], [expected[columns].to_numpy()]
                    )

    def test_lstm_keeps_tensor_and_date_types_when_all_windows_cross_a_gap(self):
        sparse = self.source.loc[
            (self.source['WMO'] == 'A') & self.source['Date'].between('2023-03-01', '2025-02-01')
        ]
        windows, values, dates, stations, latitudes, longitudes = self.engine._build_sequences(
            sparse, 'DTR'
        )
        self.assertEqual(windows.shape, (0, 12, 6))
        self.assertEqual(windows.dtype, np.float32)
        self.assertTrue(np.issubdtype(dates.dtype, np.datetime64))
        self.assertTrue(all(array.shape == (0,) for array in (values, dates, stations, latitudes, longitudes)))


if __name__ == '__main__':
    unittest.main()
