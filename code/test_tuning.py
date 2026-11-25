from contextlib import redirect_stdout
from io import StringIO
import json
from pathlib import Path
import sys
from tempfile import TemporaryDirectory
from unittest import TestCase
from unittest.mock import patch
import warnings

import numpy as np
import pandas as pd
from sklearn.exceptions import ConvergenceWarning

from kursach011_forecast_report import ForecastReporter
from kursach011_modules import mlp_check, tuning_lab
from kursach011_modules.annual_forecast import ANNUAL_FORECAST_MODE
from kursach011_modules.forecast_engine import LongTermValidator, StandardForecaster
from kursach011_modules.mlp_check import MLP, RF, compare_releases, summarize_seeds
from kursach011_modules.mlp_diagnostics import MlpDiagnostics
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


class MlpDiagnosticsTest(TestCase):
    def setUp(self):
        self.base_dir = Path(self.enterContext(TemporaryDirectory()))
        self.output = self.base_dir / 'output011_modules'
        self.diagnostics = self.base_dir / 'diagnostics' / 'mlp'
        self.source = monthly_data(1985, 1989)
        self.dataset = 'CRU_CEDA'
        self.params = {'hidden_layer_sizes': (2,), 'max_iter': 1}
        self.enterContext(redirect_stdout(StringIO()))

    def test_diagnostics_written_by_forecasters_report_and_tuning(self):
        updates = {
            (mode, self.dataset, target, MLP): {'parameters': self.params}
            for mode in (MONTHLY_FORECAST_MODE, ANNUAL_FORECAST_MODE) for target in ('DTR', 'Precip')
        }
        save_model_parameters(self.output / MODEL_PARAMS_FILENAME, updates)
        monthly = StandardForecaster(self.base_dir, self.output)
        annual = LongTermValidator(self.base_dir, self.output)
        reporter = ForecastReporter(self.base_dir, self.output)
        for engine in (monthly, annual, reporter):
            engine.model_classes = {MLP: engine.model_classes[MLP]}
            self.enterContext(patch.object(engine, 'load_data', return_value=self.source))

        active = {target: {MLP: {'parameters': self.params}} for target in ('DTR', 'Precip')}
        with warnings.catch_warnings():
            warnings.simplefilter('ignore', ConvergenceWarning)
            monthly.run(1989, self.dataset)
            annual.run(1989, self.dataset, end_year=1989)
            reporter.run_standard_forecast_report(1989, self.dataset)
            reporter.run_validation_report(1989, self.dataset, end_year=1989)
            with (
                patch.object(tuning_lab, 'BASE_DIR', self.base_dir),
                patch.object(tuning_lab, 'BEST_PARAMS', {self.dataset: active}),
                patch.dict(tuning_lab.TUNING_YEARS, {ANNUAL_FORECAST_MODE: (1989,)}),
                patch.object(tuning_lab, 'load_forecast_data', return_value=self.source),
                patch.object(tuning_lab, 'parameter_candidates', return_value=[self.params]),
                patch.object(sys, 'argv', ['tuning_lab.py', '--mode', ANNUAL_FORECAST_MODE]),
            ):
                tuning_lab.main()

        files = (
            ('standard_forecast/CRU_CEDA_DTR_1989.json', MONTHLY_FORECAST_MODE, 12),
            ('validation_results/Validation_CRU_CEDA_DTR_1989-1989.json', ANNUAL_FORECAST_MODE, 1),
            ('forecast_report/CRU_CEDA_DTR_1989.json', MONTHLY_FORECAST_MODE, 12),
            ('forecast_report/Validation_CRU_CEDA_DTR_1989-1989.json', ANNUAL_FORECAST_MODE, 1),
            ('tuning/direct_annual_CRU_CEDA_DTR_1989-1989.json', ANNUAL_FORECAST_MODE, 1),
        )
        for name, mode, count in files:
            path = self.diagnostics / name
            with self.subTest(path=path):
                record = json.loads(path.read_text(encoding='utf-8'))
                self.assertEqual(record['summary'], {
                    'fits': count, 'reached_limit': count, 'with_convergence_warning': count,
                })
                self.assertEqual(len(record['fits']), count)
                first = record['fits'][0]
                self.assertEqual(first['release'], '1989-01-01')
                self.assertEqual(first['mode'], mode)
                self.assertEqual(first['dataset'], self.dataset)
                self.assertEqual(first['target'], 'DTR')
                self.assertEqual(first['seed'], 42)
                self.assertEqual(first['n_iter_'], 1)
                self.assertEqual(first['max_iter'], 1)
                self.assertTrue(first['convergence_warnings'])
                self.assertEqual(len(first['loss_curve_']), 1)

        self.assertFalse((self.output / 'mlp_diagnostics').exists())
        self.assertTrue((self.output / 'standard_forecast/csv_data/CRU_CEDA_DTR_1989.csv').is_file())
        self.assertTrue((self.output / 'validation_results/csv_data/Validation_CRU_CEDA_DTR_1989-1989.csv').is_file())
        records = read_model_parameters(self.output / MODEL_PARAMS_FILENAME)
        self.assertEqual(records[ANNUAL_FORECAST_MODE][self.dataset]['DTR'][MLP]['selection'], 'tuning')
        self.assertFalse((self.diagnostics / MODEL_PARAMS_FILENAME).exists())

        cache_file = self.diagnostics / files[0][0]
        original = cache_file.read_bytes()
        with patch.object(monthly, '_fit_predict_model', side_effect=AssertionError('cache must be reused')):
            monthly.run(1989, self.dataset)
        self.assertEqual(cache_file.read_bytes(), original)

    def test_short_check_default_and_explicit_output_directories(self):
        config = self.output / MODEL_PARAMS_FILENAME
        save_model_parameters(config, {
            (MONTHLY_FORECAST_MODE, self.dataset, 'DTR', MLP): {'parameters': self.params},
        })
        original_config = config.read_bytes()
        custom = self.base_dir / 'manual_check'
        names = {'comparison.csv', 'seed_summary.csv', 'mlp_fits.json'}
        locations = (
            ([], self.diagnostics / 'short_check' / MONTHLY_FORECAST_MODE),
            (['--output-dir', str(custom)], custom),
        )
        with (
            patch.object(mlp_check, 'BASE_DIR', self.base_dir),
            patch.object(mlp_check, 'load_forecast_data', return_value=monthly_data(2020, 2023)),
            patch.object(mlp_check, 'EPOCH_LIMITS', (1, 2)),
        ):
            for options, directory in locations:
                with patch.object(sys, 'argv', ['mlp_check.py', '--mode', MONTHLY_FORECAST_MODE, *options]):
                    mlp_check.main()
                self.assertEqual({path.name for path in directory.iterdir()}, names)
                comparison = pd.read_csv(directory / 'comparison.csv')
                self.assertEqual(set(comparison['mode']), {MONTHLY_FORECAST_MODE})

        self.assertEqual({path.name for path in self.output.iterdir()}, {MODEL_PARAMS_FILENAME})
        self.assertEqual(config.read_bytes(), original_config)

    def test_short_check_uses_common_pairs_and_summarizes_only_active_seed_runs(self):
        self.source.loc[self.source['Date'].eq('1989-06-01'), 'Precip'] = np.nan
        diagnostics = MlpDiagnostics()
        with patch('kursach011_modules.mlp_check.EPOCH_LIMITS', (1, 2)):
            rows = compare_releases(
                self.source, mode=ANNUAL_FORECAST_MODE, dataset=self.dataset, target='Precip',
                releases=('1989-01-01',), active_params={
                    MLP: {'parameters': self.params},
                    RF: {'parameters': {'n_estimators': 2}},
                }, diagnostics=diagnostics,
            )

        comparison = pd.DataFrame(rows)
        self.assertEqual(len(comparison), 8)
        self.assertEqual(len(diagnostics.fits), 4)
        self.assertEqual(set(comparison['n_months']), {11})
        self.assertEqual(set(comparison['n_station_months']), {11})
        cap_one = comparison.loc[comparison['variant'].eq('epochs_1')].iloc[0]
        active = comparison.loc[comparison['variant'].eq('active') & comparison['model'].eq(MLP)]
        self.assertEqual(cap_one['regional_mae'], active.iloc[0]['regional_mae'])
        summary = summarize_seeds(comparison)
        self.assertEqual(len(summary), 4)
        self.assertEqual(set(summary['num_seeds']), {3})
        values = active['regional_mae'].to_numpy()
        regional = summary.loc[summary['model'].eq(MLP) & summary['metric'].eq('regional_mae')].iloc[0]
        self.assertAlmostEqual(regional['mean'], np.mean(values))
        self.assertAlmostEqual(regional['std'], np.std(values, ddof=1))
        self.assertAlmostEqual(regional['min'], min(values))
        self.assertAlmostEqual(regional['max'], max(values))


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
