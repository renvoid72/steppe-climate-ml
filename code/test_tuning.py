from pathlib import Path
from tempfile import TemporaryDirectory
from unittest import TestCase
from unittest.mock import patch

import numpy as np
import pandas as pd

from kursach011_forecast_report import ForecastReporter
from kursach011_modules.annual_forecast import ANNUAL_FORECAST_MODE
from kursach011_modules.forecast_engine import LongTermValidator, StandardForecaster
from kursach011_modules.model_parameters import (
    BEST_PARAMS, MODEL_PARAMS_FILENAME, MONTHLY_FORECAST_MODE, read_model_parameters, save_model_parameters,
)
from kursach011_modules.tuning_lab import TUNING_YEARS, parameter_candidates, prepare_tuning_splits, select_parameters


def monthly_data(first_year, last_year):
    dates = pd.date_range(f'{first_year}-01-01', f'{last_year}-12-01', freq='MS')
    return pd.DataFrame({
        'Date': dates, 'WMO': 'A', 'Latitude': 45.0, 'Longitude': 60.0,
        'Year': dates.year, 'Month': dates.month,
        'DTR': dates.year * 100. + dates.month, 'Precip': dates.month - 1.,
    })


class TuningTest(TestCase):
    def test_monthly_tuning_uses_only_history_before_each_month_and_excludes_2024(self):
        data = monthly_data(2020, 2024)
        years = TUNING_YEARS[MONTHLY_FORECAST_MODE]
        dates_seen = []
        for train, test, targets, facts, dates in prepare_tuning_splits(data, MONTHLY_FORECAST_MODE, years)('DTR'):
            date = pd.Timestamp(dates.item())
            dates_seen.append(date)
            self.assertTrue((train['Date'] < date).all())
            self.assertTrue(test['DTR'].isna().all())
            self.assertIsNone(targets)
            self.assertEqual(facts.item(), date.year * 100. + date.month)
        self.assertEqual(dates_seen, list(pd.date_range('2022-01-01', '2023-12-01', freq='MS')))

    def test_annual_tuning_uses_only_complete_previous_years(self):
        data = monthly_data(1977, 1990)
        complete_months = data['Date'] != '1988-06-01'
        data = pd.concat([data, data.loc[complete_months].assign(WMO='B')], ignore_index=True)
        years = TUNING_YEARS[ANNUAL_FORECAST_MODE]
        splits = list(prepare_tuning_splits(data, ANNUAL_FORECAST_MODE, years)('DTR'))
        self.assertEqual(len(splits), 10)
        for year, (train, test, targets, facts, dates) in zip(years, splits, strict=True):
            self.assertTrue((train['Date'].dt.year < year).all())
            self.assertTrue(test['DTR'].isna().all())
            self.assertEqual(targets.shape, (len(train), 12))
            self.assertTrue(np.isfinite(targets).all())
            self.assertEqual(len(facts.ravel()), len(dates))
            np.testing.assert_array_equal(pd.DatetimeIndex(dates).month, np.tile(np.arange(1, 13), len(test)))
        last_train = splits[-1][0]
        stations_and_years = set(zip(last_train['Date'].dt.year, last_train['WMO']))
        self.assertNotIn((1988, 'B'), stations_and_years)
        self.assertIn((1988, 'A'), stations_and_years)

    def test_candidates_include_baseline_and_share_dates_and_known_facts(self):
        target, model = 'DTR', 'K-Neighbors'
        active = {'n_neighbors': 22, 'weights': 'distance'}
        self.assertEqual(parameter_candidates(model, active)[0], active)
        train = pd.DataFrame({'Feature': [0., 2.], target: [1., 9.]})
        test = pd.DataFrame({'Feature': [0., 2.]})
        dates = pd.DatetimeIndex(['2022-01-01'] * 2).to_numpy()
        candidates = [{'n_neighbors': 2}, {'n_neighbors': 1}]
        splits = [
            (train.iloc[:1], test, None, np.array([30., 30.]), dates),
            (train, test, None, np.array([1., np.nan]), dates + np.timedelta64(31, 'D')),
        ]
        result = select_parameters(splits, target, model, candidates)
        self.assertEqual(result['parameters'], candidates[1])
        self.assertEqual(result['score'], 0.)
        self.assertEqual(result['n_months'], 1)
        self.assertEqual(result['n_station_months'], 1)


class ModelParametersTest(TestCase):
    def setUp(self):
        self.output = Path(self.enterContext(TemporaryDirectory()))
        self.path = self.output / MODEL_PARAMS_FILENAME
        self.dataset, self.target, self.model = 'CRU_CEDA', 'DTR', 'K-Neighbors'
        self.key = (self.dataset, self.target, self.model)

    def save_neighbors(self, mode, count):
        save_model_parameters(self.path, {(mode, *self.key): {'parameters': {'n_neighbors': count}}})

    def test_loads_by_mode_preserves_other_mode_and_falls_back_without_file(self):
        default = BEST_PARAMS[self.dataset][self.target][self.model]
        initial = StandardForecaster(self.output, self.output)
        self.assertEqual(initial.model_parameters(self.model, self.target, self.dataset), default)

        self.save_neighbors(MONTHLY_FORECAST_MODE, 21)
        annual = LongTermValidator(self.output, self.output)
        self.assertEqual(annual.model_parameters(self.model, self.target, self.dataset), default)

        self.save_neighbors(ANNUAL_FORECAST_MODE, 42)
        reporter = ForecastReporter(self.output, self.output)
        features = pd.DataFrame({'Feature': [0., 1.], self.target: [1., 2.]})
        with patch('kursach011_forecast_report.build_regressor') as build:
            build.return_value.predict.return_value = np.array([1.])
            reporter._timed_fit_predict(features, features.iloc[:1], self.model, self.target, self.dataset)
            reporter._timed_fit_predict(
                features, features.iloc[:1], self.model, self.target, self.dataset, mode=ANNUAL_FORECAST_MODE,
            )
        self.assertEqual([call.args[2]['n_neighbors'] for call in build.call_args_list], [21, 42])
        self.path.write_text('{}', encoding='utf-8')
        with self.assertRaisesRegex(ValueError, 'не удалось прочитать параметры'):
            read_model_parameters(self.path)

    def test_only_changed_parameters_invalidate_the_matching_cache(self):
        standard = StandardForecaster(self.output, self.output)
        annual = LongTermValidator(self.output, self.output)
        filename = 'predictions.csv'
        rows = monthly_data(2024, 2024).iloc[:1].assign(
            Forecast=1., Fact=1., Model=self.model, Target_Var=self.target,
        )
        for engine in (standard, annual):
            engine._save_predictions([rows], filename, self.dataset)
            self.assertIsNotNone(engine._load_cached_predictions(filename, self.dataset))
        self.save_neighbors(ANNUAL_FORECAST_MODE, 42)

        standard = StandardForecaster(self.output, self.output)
        annual = LongTermValidator(self.output, self.output)
        self.assertIsNotNone(standard._load_cached_predictions(filename, self.dataset))
        self.assertIsNone(annual._load_cached_predictions(filename, self.dataset))
        annual._save_predictions([rows], filename, self.dataset)
        self.assertIsNotNone(annual._load_cached_predictions(filename, self.dataset))
        self.path.unlink()
        self.assertIsNone(LongTermValidator(self.output, self.output)._load_cached_predictions(filename, self.dataset))
