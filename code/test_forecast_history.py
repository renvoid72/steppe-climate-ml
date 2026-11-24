# -*- coding: utf-8 -*-
import unittest
from contextlib import redirect_stdout
from io import BytesIO, TextIOWrapper
from pathlib import Path
from tempfile import TemporaryDirectory
from unittest.mock import patch

import numpy as np
import pandas as pd

from kursach011_modules.annual_forecast import ANNUAL_FORECAST_MODE, prepare_annual_features
from kursach011_modules.forecast_engine import (
    CRU_SOURCE_VERSION, LongTermValidator, StandardForecaster, read_prediction_csv, read_validation_csv,
)
from kursach011_modules.monthly_features import create_forecast_features, prepare_forecast_period, select_model_rows
from kursach011_modules.project_settings import validation_filename


class ForecastHistoryTest(unittest.TestCase):
    def setUp(self):
        dates = pd.date_range('2022-01-01', '2024-01-01', freq='MS')
        january = dates[-1]
        self.source = pd.DataFrame(
            [("A", date, 10.0, 0.0) for date in dates if date != pd.Timestamp('2023-06-01')]
            + [("B", date, 900.0, 90.0) for date in dates if date != pd.Timestamp('2023-12-01')]
            + [("C", date, 20.0, 1.0) for date in dates[-4:]]
            + [("D", january, 900.0, 90.0)],
            columns=['WMO', 'Date', 'DTR', 'Precip'],
        )
        self.source['Year'] = self.source['Date'].dt.year
        self.source['Month'] = self.source['Date'].dt.month
        self.source['Latitude'] = 45.0
        self.source['Longitude'] = 60.0
        self.output_dir = self.enterContext(TemporaryDirectory())

    def use_knn(self, forecaster):
        forecaster.model_classes = {'K-Neighbors': forecaster.model_classes['K-Neighbors']}
        self.enterContext(patch.object(forecaster, 'load_data', return_value=self.source))

    def run_with_cp1251_stdout(self, run, *args, **kwargs):
        with BytesIO() as buffer, TextIOWrapper(buffer, encoding='cp1251') as stream, redirect_stdout(stream):
            run(*args, **kwargs)
            stream.flush()
            return buffer.getvalue().decode('cp1251')

    def test_future_rows_require_history_but_allow_unknown_targets_and_zero_lags(self):
        forecast_date = pd.Timestamp('2024-01-01')
        features = create_forecast_features(self.source, forecast_date)
        future = features.loc[features['Date'] == forecast_date].set_index('WMO')

        self.assertTrue(future[['DTR', 'Precip']].isna().all().all())
        self.assertTrue(pd.isna(future.loc['B', 'DTR_lag_1']))
        self.assertTrue(pd.isna(future.loc['C', 'DTR_lag_12']))
        self.assertEqual(select_model_rows(future).index.tolist(), ['A'])
        self.assertEqual(future.loc['A', 'Precip_lag_1'], 0.0)
        self.assertTrue(select_model_rows(future, 'DTR').empty)

    def test_period_matches_individual_cutoffs_with_new_stations_and_missing_months(self):
        dates = pd.date_range('2023-11-01', periods=5, freq='MS')
        history, candidates = prepare_forecast_period(self.source, dates)
        for date in dates:
            with self.subTest(date=date):
                expected = create_forecast_features(self.source, date)
                for actual, reference in (
                    (history.loc[history['Date'] < date], expected.loc[expected['Date'] < date]),
                    (candidates.loc[candidates['Date'] == date], expected.loc[expected['Date'] == date]),
                ):
                    pd.testing.assert_frame_equal(
                        actual.reset_index(drop=True), reference.reset_index(drop=True),
                        check_exact=True, check_dtype=False,
                    )

    def test_meteo_keeps_one_known_indicator_for_targets_and_lags(self):
        base_dir = Path(self.output_dir)
        meteo_dir = base_dir / 'meteo_ru'
        meteo_dir.mkdir()
        (meteo_dir / 'all_steppe_data.txt').touch()
        (meteo_dir / 'stations_coords.txt').touch()
        monthly = self.source.drop(columns='Date').copy()
        january = monthly['Year'].eq(2024) & monthly['Month'].eq(1) & monthly['WMO'].eq('A')
        monthly.loc[january, 'Precip'] = None
        forecaster = StandardForecaster(base_dir, base_dir)

        with patch('kursach011_modules.forecast_engine.load_monthly_meteo', return_value=monthly):
            data = forecaster.load_data('METEO_RU')

        features = forecaster.create_features(data)
        targets = features.loc[features['Year'] == 2024]
        self.assertEqual(select_model_rows(targets, 'DTR')['WMO'].tolist(), ['A'])
        self.assertTrue(select_model_rows(targets, 'Precip').empty)

        february = pd.Timestamp('2024-02-01')
        forecast = create_forecast_features(data, february)
        next_month = forecast.loc[forecast['Date'] == february].set_index('WMO').loc['A']
        self.assertEqual(next_month['DTR_lag_1'], 10.0)
        self.assertTrue(pd.isna(next_month['Precip_lag_1']))

    def test_meteo_forecast_cache_requires_calendar_coverage(self):
        forecaster = StandardForecaster(self.output_dir, self.output_dir)
        filename = 'METEO_RU_DTR_2024.csv'
        forecast = self.source.iloc[:1].assign(
            Forecast=10.0, Fact=10.0, Model='K-Neighbors', Target_Var='DTR',
            History_Policy='complete_features',
        )
        forecast.to_csv(forecaster.output_dir / filename, index=False)

        self.assertFalse(forecaster._is_file_valid(filename, 'METEO_RU'))
        forecaster._save_predictions([forecast], filename, 'METEO_RU')
        self.assertTrue(forecaster._is_file_valid(filename, 'METEO_RU'))

    def test_standard_forecast_replaces_legacy_cache_and_preserves_missing_facts(self):
        forecaster = StandardForecaster(self.output_dir, self.output_dir)
        self.use_knn(forecaster)
        filename = 'CRU_CEDA_Precip_2024.csv'
        output = forecaster.output_dir / filename
        self.source.loc[(self.source['WMO'] == 'B') & (self.source['Year'] == 2024)].assign(
            Forecast=90.0, Fact=90.0, Model='K-Neighbors', Target_Var='Precip'
        ).to_csv(output, index=False)

        output_text = self.run_with_cp1251_stdout(forecaster.run, 2024, 'CRU_CEDA')

        self.assertEqual(output_text.count(
            'кандидатов: 47, допущено: 2, исключено: 45, прогнозов: 2, пар с фактом: 1'
        ), 2)
        self.assertIn('R2=', output_text)
        results = pd.read_csv(output, parse_dates=['Date']).set_index('Date')
        january = results.loc['2024-01-01']
        self.assertEqual(january['WMO'], 'A')
        self.assertEqual(january['Fact'], 0.0)
        self.assertEqual(results.loc['2024-02-01', 'WMO'], 'A')
        self.assertTrue(pd.isna(results.loc['2024-02-01', 'Fact']))
        with patch.object(forecaster, 'load_data', side_effect=AssertionError('CSV cache was ignored')):
            forecaster.run(2024, 'CRU_CEDA')

    def test_validation_forecasts_full_year_and_trains_only_on_completed_years(self):
        validator = LongTermValidator(self.output_dir, self.output_dir)
        validator.model_classes = {'Random Forest': validator.model_classes['Random Forest']}
        dates = pd.date_range('2020-01-01', '2025-12-01', freq='MS')
        data = pd.DataFrame({
            'Date': dates, 'WMO': 'A', 'Latitude': 45.0, 'Longitude': 60.0,
            'DTR': dates.year * 100. + dates.month, 'Precip': dates.month - 1.,
            'Year': dates.year, 'Month': dates.month,
        })
        data = pd.concat([data, data.loc[data['Date'] != '2024-06-01'].assign(WMO='B')])
        old_file = validator.output_dir / 'Validation_CRU_CEDA_DTR_2024-2025.csv'
        legacy = data.iloc[:1].assign(
            Forecast=10.0, Fact=10.0, Model='Random Forest', Target_Var='DTR',
            History_Policy='complete_features', Source_Version=CRU_SOURCE_VERSION,
        )
        legacy.to_csv(old_file, index=False)
        with self.assertRaisesRegex(ValueError, 'пересчитайте валидацию'):
            read_validation_csv(old_file, 'CRU_CEDA')

        changed = data.copy()
        changed.loc[changed['Year'] >= 2024, ['DTR', 'Precip']] = 999.
        before = prepare_annual_features(data, 2025).loc[lambda rows: rows['Date'].dt.year == 2024]
        after = prepare_annual_features(changed, 2025).loc[lambda rows: rows['Date'].dt.year == 2024]
        pd.testing.assert_frame_equal(before, after, check_exact=True)

        def predict(train, test, model_name, target_var, dataset_name, *, y_train):
            year = test['Date'].dt.year.unique().item()
            self.assertTrue((train['Date'].dt.year < year).all())
            months = np.tile(np.arange(1, 13), (len(train), 1))
            offset = train['Date'].dt.year.to_numpy()[:, None] * 100 if target_var == 'DTR' else -1
            np.testing.assert_array_equal(y_train, months + offset)
            years_and_stations = set(zip(train['Date'].dt.year, train['WMO']))
            self.assertEqual((2024, 'A') in years_and_stations, year == 2025)
            self.assertNotIn((2024, 'B'), years_and_stations)
            self.assertTrue(test[target_var].isna().all())
            return np.tile(np.arange(1.0, 13.0), (len(test), 1))

        with (
            patch.object(validator, 'load_data', return_value=data),
            patch.object(validator, '_fit_predict_model', side_effect=predict) as fit_predict,
        ):
            output = self.run_with_cp1251_stdout(validator.run, 2024, 'CRU_CEDA', end_year=2025)
        self.assertEqual(fit_predict.call_count, 4)
        self.assertIn('Месячных прогнозов: 48, пар с фактом: 47', output)
        results = pd.read_csv(
            validator.output_dir / validation_filename('CRU_CEDA', 'Precip', 2024, 2025),
            parse_dates=['Date'],
        )
        self.assertEqual(results.groupby(['WMO', results['Date'].dt.year]).size().tolist(), [12] * 4)
        self.assertTrue(results.loc[results['Fact'].isna(), 'Date'].eq('2024-06-01').item())
        np.testing.assert_array_equal(results['Forecast'], results['Date'].dt.month)
        replaced = read_validation_csv(old_file, 'CRU_CEDA')
        self.assertEqual(len(replaced), 48)
        self.assertTrue(validator._is_file_valid(old_file.name, 'CRU_CEDA'))
        mixed = pd.concat([replaced, legacy], ignore_index=True)
        mixed.to_csv(old_file, index=False)
        self.assertFalse(validator._is_file_valid(old_file.name, 'CRU_CEDA'))

    def test_standard_forecast_handles_a_year_with_no_history(self):
        forecaster = StandardForecaster(self.output_dir, self.output_dir)
        self.use_knn(forecaster)

        forecaster.run(2021, 'CRU_CEDA')

        results = pd.read_csv(forecaster.output_dir / 'CRU_CEDA_DTR_2021.csv')
        self.assertTrue(results.empty)
        self.assertIn('Forecast', results.columns)

    def test_cru_cache_requires_current_source_version(self):
        validator = LongTermValidator(self.output_dir, self.output_dir)
        filename = validation_filename('CRU_CEDA', 'DTR')
        output = validator.output_dir / filename
        forecast = self.source.iloc[:1].assign(
            Forecast=10.0, Fact=10.0, Model='K-Neighbors', Target_Var='DTR',
            History_Policy='complete_features',
        )
        old_version = forecast.assign(Source_Version='CRU TS 4.06')
        mixed = pd.concat([forecast.assign(Source_Version=CRU_SOURCE_VERSION), old_version], ignore_index=True)
        for stale in (forecast, old_version, mixed):
            stale.to_csv(output, index=False)
            self.assertFalse(validator._is_file_valid(filename, 'CRU_CEDA'))
            with self.assertRaisesRegex(ValueError, 'требуется результат расчёта'):
                read_prediction_csv(output, 'CRU_CEDA')

        validator._save_predictions([forecast], filename, 'CRU_CEDA')
        self.assertTrue(validator._is_file_valid(filename, 'CRU_CEDA'))
        result = read_prediction_csv(output, 'CRU_CEDA')
        self.assertEqual(result['Source_Version'].tolist(), [CRU_SOURCE_VERSION])

    def test_maps_read_only_requested_period_and_current_cru(self):
        from kursach011_modules.visualizer import StandardForecastVisualizer, ValidationVisualizer

        forecast = self.source.iloc[:1].assign(
            Forecast=10.0, Fact=10.0, Model='K-Neighbors', Target_Var='DTR',
            Source_Version=CRU_SOURCE_VERSION,
        )
        cases = (
            (StandardForecastVisualizer, '{dataset}_{variable}_{period}.csv', '2024', '2023', 2024),
            (ValidationVisualizer, 'Validation_{dataset}_{variable}_{period}.csv', '1990-2020', '1990-2024', 2020),
        )
        for visualize, pattern, period, other_period, year in cases:
            with self.subTest(visualize=visualize):
                visualizer = visualize.__new__(visualize)
                visualizer.csv_dir = Path(self.output_dir)
                records = forecast.assign(Date=f'{year}-01-01')
                if visualize is ValidationVisualizer:
                    records = records.assign(Forecast_Mode=ANNUAL_FORECAST_MODE)
                for dataset in ('METEO_RU', 'CRU_CEDA'):
                    name = pattern.format(dataset=dataset, variable='DTR', period=period)
                    records.to_csv(visualizer.csv_dir / name, index=False)
                old_period = pattern.format(dataset='METEO_RU', variable='Precip', period=other_period)
                records.to_csv(visualizer.csv_dir / old_period, index=False)
                old_cru = pattern.format(dataset='CRU_CEDA', variable='Precip', period=period)
                records.assign(Source_Version='CRU TS 4.06').to_csv(visualizer.csv_dir / old_cru, index=False)
                if visualize is ValidationVisualizer:
                    legacy = pattern.format(dataset='METEO_RU', variable='Precip', period=period)
                    records.drop(columns='Forecast_Mode').to_csv(visualizer.csv_dir / legacy, index=False)

                with (
                    patch.object(visualizer, '_plot_full_graph') as graph,
                    patch.object(visualizer, '_calculate_metrics'),
                    patch.object(visualizer, '_save_map_netcdf_and_png') as save_map,
                ):
                    visualizer.run()

                self.assertEqual([call.args[1:3] for call in graph.call_args_list], [
                    ('METEO_RU', 'DTR'), ('CRU_CEDA', 'DTR'),
                ])
                self.assertEqual({call.args[3] for call in save_map.call_args_list}, {'METEO_RU', 'CRU_CEDA'})
