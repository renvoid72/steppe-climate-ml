# -*- coding: utf-8 -*-
"""
Регрессионные тесты календарных лагов и скользящих средних.

На данных двух станций проверяются пропуск месяца, границы окон и
сохранение строки прогноза. Тесты запускаются без обучения моделей.
"""

import unittest
from tempfile import TemporaryDirectory

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


if __name__ == '__main__':
    unittest.main()
