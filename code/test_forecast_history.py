# -*- coding: utf-8 -*-
import unittest
from contextlib import redirect_stdout
from io import BytesIO, TextIOWrapper
from pathlib import Path
from tempfile import TemporaryDirectory
from unittest.mock import patch

import pandas as pd

from kursach011_modules.forecast_engine import (
    CRU_SOURCE_VERSION, LongTermValidator, StandardForecaster, read_prediction_csv,
)
from kursach011_modules.monthly_features import create_forecast_features, select_model_rows


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

    def test_validation_evaluates_only_eligible_rows(self):
        validator = LongTermValidator(self.output_dir, self.output_dir)
        self.use_knn(validator)

        output_text = self.run_with_cp1251_stdout(validator.run, 2024, 'CRU_CEDA', end_year=2024)

        self.assertEqual(output_text.count(
            'кандидатов: 4, допущено: 1, исключено: 3, прогнозов: 1, пар с фактом: 1'
        ), 2)
        self.assertIn('R2=', output_text)
        results = pd.read_csv(validator.output_dir / 'Validation_CRU_CEDA_Precip_2024-2024.csv')
        self.assertEqual(results['WMO'].tolist(), ['A'])
        self.assertEqual(results['Fact'].tolist(), [0.0])

    def test_standard_forecast_handles_a_year_with_no_history(self):
        forecaster = StandardForecaster(self.output_dir, self.output_dir)
        self.use_knn(forecaster)

        forecaster.run(2021, 'CRU_CEDA')

        results = pd.read_csv(forecaster.output_dir / 'CRU_CEDA_DTR_2021.csv')
        self.assertTrue(results.empty)
        self.assertIn('Forecast', results.columns)

    def test_cru_cache_requires_current_source_version(self):
        validator = LongTermValidator(self.output_dir, self.output_dir)
        filename = 'Validation_CRU_CEDA_DTR_1990-2020.csv'
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
                for dataset in ('METEO_RU', 'CRU_CEDA'):
                    name = pattern.format(dataset=dataset, variable='DTR', period=period)
                    records.to_csv(visualizer.csv_dir / name, index=False)
                old_period = pattern.format(dataset='METEO_RU', variable='Precip', period=other_period)
                records.to_csv(visualizer.csv_dir / old_period, index=False)
                old_cru = pattern.format(dataset='CRU_CEDA', variable='Precip', period=period)
                records.assign(Source_Version='CRU TS 4.06').to_csv(visualizer.csv_dir / old_cru, index=False)

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
