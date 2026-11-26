# -*- coding: utf-8 -*-
import unittest
from contextlib import redirect_stdout
from io import BytesIO, StringIO, TextIOWrapper
from pathlib import Path
from tempfile import TemporaryDirectory
from unittest.mock import patch

import numpy as np
import pandas as pd

from kursach011_modules.annual_forecast import ANNUAL_FORECAST_MODE, prepare_annual_features
from kursach011_modules.forecast_engine import (
    CRU_SOURCE_VERSION, LongTermValidator, StandardForecaster, read_prediction_csv, read_validation_csv,
)
from kursach011_modules.forecast_metrics import (
    calculate_climatology_comparison, calculate_forecast_metrics, paired_predictions, station_month_climatology,
)
from kursach011_modules.monthly_features import create_forecast_features, prepare_forecast_period, select_model_rows
from kursach011_modules.project_settings import forecast_filename, validation_filename


class ForecastMetricsTest(unittest.TestCase):
    def test_regional_metrics_use_paired_stations_and_distinct_calendar_months(self):
        predictions = pd.DataFrame([
            ('2024-01-01', 20.0, 30.0),
            ('2024-01-01', 40.0, 30.0),
            ('2024-01-01', 0.0, 0.0),
            ('2024-01-01', np.nan, 300.0),
            ('2024-01-01', 200.0, np.nan),
            ('2024-02-01', 5.0, np.inf),
            ('2025-01-01', 10.0, 14.0),
        ], columns=['Date', 'Fact', 'Forecast'])

        regional, station = calculate_forecast_metrics(paired_predictions(predictions), 'Precip')

        self.assertEqual((regional['N'], station['N']), (2, 4))
        self.assertAlmostEqual(regional['MAE'], 2.0)
        self.assertAlmostEqual(station['MAE'], 6.0)
        self.assertAlmostEqual(regional['RMSE'], np.sqrt(8))
        self.assertAlmostEqual(station['RMSE'], np.sqrt(54))
        self.assertEqual(regional['Ед. ошибки'], 'мм')

    def test_empty_and_single_pair(self):
        predictions = pd.DataFrame({
            'Date': pd.to_datetime(['2024-01-01']),
            'Fact': [0.0],
            'Forecast': [2.0],
        })

        for metric in calculate_forecast_metrics(paired_predictions(predictions.iloc[:0]), 'DTR'):
            self.assertEqual(metric['N'], 0)
            self.assertTrue(all(np.isnan(metric[name]) for name in ['R2', 'MAE', 'RMSE']))
        for metric in calculate_forecast_metrics(paired_predictions(predictions), 'DTR'):
            self.assertEqual(metric['N'], 1)
            self.assertEqual(metric['MAE'], 2.0)
            self.assertTrue(np.isnan(metric['R2']))
            self.assertEqual(metric['Ед. ошибки'], '°C')

    def test_climatology_uses_only_prior_years_for_each_station_month_and_target(self):
        history = pd.DataFrame([
            ('A', '2018-01-01', 4., 0.),
            ('A', '2019-01-01', 6., 20.),
            ('A', '2019-06-01', 8., 9.),
            ('A', '2020-01-01', 12., 40.),
            ('A', '2020-06-01', 90., 900.),
            ('A', '2021-01-01', 99., 999.),
            ('B', '2019-01-01', np.nan, 60.),
            ('B', '2018-01-01', np.inf, np.nan),
            ('C', '2020-01-01', 1., 1.),
        ], columns=['WMO', 'Date', 'DTR', 'Precip'])
        history['Date'] = pd.to_datetime(history['Date'])
        climate = station_month_climatology(history, [2020, 2021])

        self.assertEqual(climate.loc[('A', pd.Timestamp('2020-01-01'))].tolist(), [5., 10.])
        self.assertEqual(climate.loc[('A', pd.Timestamp('2020-06-01'))].tolist(), [8., 9.])
        self.assertEqual(climate.loc[('A', pd.Timestamp('2021-01-01')), 'Precip'], 20.)
        self.assertTrue(np.isnan(climate.loc[('B', pd.Timestamp('2020-01-01')), 'DTR']))
        self.assertEqual(climate.loc[('B', pd.Timestamp('2020-01-01')), 'Precip'], 60.)
        self.assertNotIn(('C', pd.Timestamp('2020-01-01')), climate.index)

    def test_climatology_comparison_scores_the_same_finite_pairs_at_both_levels(self):
        history = pd.DataFrame([
            ('A', '2019-01-01', 0.), ('B', '2019-01-01', 14.), ('A', '2019-02-01', 10.),
            ('D', '2019-01-01', 50.), ('E', '2019-01-01', 55.), ('F', '2019-01-01', 8.),
        ], columns=['WMO', 'Date', 'Precip']).assign(DTR=np.nan)
        history['Date'] = pd.to_datetime(history['Date'])
        climate = station_month_climatology(history, [2020])
        predictions = pd.DataFrame([
            ('A', '2020-01-01', 10., 4.), ('B', '2020-01-01', 20., 24.),
            ('C', '2020-01-01', 999., 0.), ('D', '2020-01-01', np.nan, 0.),
            ('E', '2020-01-01', 90., np.nan), ('F', '2020-01-01', 90., np.inf),
            ('A', '2020-02-01', 14., 12.),
        ], columns=['WMO', 'Date', 'Fact', 'Forecast'])
        regional, station = calculate_climatology_comparison(predictions, climate, 'Precip')

        self.assertEqual((regional['N'], station['N']), (2, 3))
        self.assertEqual((regional['MAE'], regional['MAE климатологии']), (1.5, 6.))
        self.assertEqual((station['MAE'], station['MAE климатологии']), (4., 20. / 3))
        self.assertAlmostEqual(station['RMSE климатологии'], np.sqrt(152 / 3))
        self.assertAlmostEqual(regional['R2 климатологии'], -159.)
        self.assertAlmostEqual(regional['Снижение MAE, %'], 75.)
        self.assertAlmostEqual(station['Снижение MAE, %'], 40.)

        exact_climate = predictions.iloc[:1].assign(Fact=0.)
        for row in calculate_climatology_comparison(exact_climate, climate, 'Precip'):
            self.assertEqual(row['MAE климатологии'], 0.)
            self.assertTrue(np.isnan(row['Снижение MAE, %']))


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

        self.assertIsNone(forecaster._load_cached_predictions(filename, 'METEO_RU'))
        forecaster._save_predictions([forecast], filename, 'METEO_RU')
        self.assertIsNotNone(forecaster._load_cached_predictions(filename, 'METEO_RU'))

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
        self.assertRegex(output_text, r'Региональные месячные средние\s+1\s+месяцы\s+—\s+0\.0000')
        self.assertRegex(output_text, r'Станция — месяц\s+1\s+пары\s+—\s+0\.0000')
        self.assertEqual(output_text.count('MAE климатологии'), 2)
        results = pd.read_csv(output, parse_dates=['Date']).set_index('Date')
        january = results.loc['2024-01-01']
        self.assertEqual(january['WMO'], 'A')
        self.assertEqual(january['Fact'], 0.0)
        self.assertEqual(results.loc['2024-02-01', 'WMO'], 'A')
        self.assertTrue(pd.isna(results.loc['2024-02-01', 'Fact']))
        saved = output.read_bytes()
        with patch.object(forecaster, '_fit_predict_model', side_effect=AssertionError('CSV cache was ignored')):
            cached_output = self.run_with_cp1251_stdout(forecaster.run, 2024, 'CRU_CEDA')
        self.assertEqual(cached_output.count('[SKIP]'), 2)
        self.assertEqual(cached_output.count('MAE климатологии'), 2)
        self.assertRegex(cached_output, r'Региональные месячные средние\s+1\s+месяцы')
        self.assertRegex(cached_output, r'Станция — месяц\s+1\s+пары')
        self.assertEqual(output.read_bytes(), saved)

        with patch.object(forecaster, 'load_data', return_value=pd.DataFrame()):
            cached_only = self.run_with_cp1251_stdout(forecaster.run, 2024, 'CRU_CEDA')
        self.assertEqual(cached_only.count('[SKIP]'), 2)
        self.assertEqual(cached_only.count('Для сравнения с климатологией нужны исходные месячные ряды.'), 2)

    def test_standard_forecast_can_cache_december_separately_from_the_calendar_year(self):
        forecaster = StandardForecaster(self.output_dir, self.output_dir)
        self.use_knn(forecaster)
        for target in ('DTR', 'Precip'):
            forecaster.model_parameters('K-Neighbors', target, 'METEO_RU')['n_neighbors'] = 3
        self.run_with_cp1251_stdout(forecaster.run, 2024, 'METEO_RU')
        annual_csv = forecaster.output_dir / 'METEO_RU_Precip_2024.csv'
        annual_contents = annual_csv.read_bytes()

        with patch.object(forecaster, '_fit_predict_model', wraps=forecaster._fit_predict_model) as fit:
            self.run_with_cp1251_stdout(forecaster.run, 2023, 'METEO_RU', target_month=12)
        self.assertEqual(fit.call_count, 2)
        for call in fit.call_args_list:
            train, forecast = call.args[:2]
            self.assertTrue(train['Date'].lt('2023-12-01').all())
            self.assertTrue(forecast['Date'].eq('2023-12-01').all())

        december_csv = forecaster.output_dir / 'METEO_RU_Precip_2023-12.csv'
        december = pd.read_csv(december_csv)
        self.assertEqual(december['Date'].unique().tolist(), ['2023-12-01'])
        self.assertEqual(december['Fact'].dropna().tolist(), [0.0])
        self.assertEqual(annual_csv.read_bytes(), annual_contents)
        self.assertFalse((forecaster.output_dir / 'METEO_RU_Precip_2023.csv').exists())
        with patch.object(forecaster, '_fit_predict_model', side_effect=AssertionError('CSV cache was ignored')):
            results = self.run_with_cp1251_stdout(forecaster.run, 2023, 'METEO_RU', target_month=12)
        self.assertEqual(results.count('[SKIP]'), 2)

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
        self.assertRegex(output, r'Региональные месячные средние\s+24\s+месяцы')
        self.assertRegex(output, r'Станция — месяц\s+47\s+пары')
        self.assertEqual(output.count('MAE климатологии'), 2)
        results = pd.read_csv(
            validator.output_dir / validation_filename('CRU_CEDA', 'Precip', 2024, 2025),
            parse_dates=['Date'],
        )
        self.assertEqual(results.groupby(['WMO', results['Date'].dt.year]).size().tolist(), [12] * 4)
        self.assertTrue(results.loc[results['Fact'].isna(), 'Date'].eq('2024-06-01').item())
        np.testing.assert_array_equal(results['Forecast'], results['Date'].dt.month)
        replaced = read_validation_csv(old_file, 'CRU_CEDA')
        self.assertEqual(len(replaced), 48)
        self.assertIsNotNone(validator._load_cached_predictions(old_file.name, 'CRU_CEDA'))
        mixed = pd.concat([replaced, legacy], ignore_index=True)
        mixed.to_csv(old_file, index=False)
        self.assertIsNone(validator._load_cached_predictions(old_file.name, 'CRU_CEDA'))

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
            self.assertIsNone(validator._load_cached_predictions(filename, 'CRU_CEDA'))
            with self.assertRaisesRegex(ValueError, 'требуется результат расчёта'):
                read_prediction_csv(output, 'CRU_CEDA')

        validator._save_predictions([forecast], filename, 'CRU_CEDA')
        self.assertIsNotNone(validator._load_cached_predictions(filename, 'CRU_CEDA'))
        result = read_prediction_csv(output, 'CRU_CEDA')
        self.assertEqual(result['Source_Version'].tolist(), [CRU_SOURCE_VERSION])

    def test_maps_read_only_requested_period_and_current_cru(self):
        from kursach011_modules.visualizer import StandardForecastVisualizer, ValidationVisualizer

        forecast = self.source.iloc[:1].assign(
            Forecast=10.0, Fact=10.0, Model='K-Neighbors', Target_Var='DTR',
            Source_Version=CRU_SOURCE_VERSION,
        )
        cases = (
            (StandardForecastVisualizer, '{dataset}_{variable}_{period}.csv', '2024', '2023', '2024-04-01'),
            (ValidationVisualizer, 'Validation_{dataset}_{variable}_{period}.csv', '1990-2020', '1990-2024', '2020-01-01'),
        )
        for visualize, pattern, period, other_period, date in cases:
            with self.subTest(visualize=visualize):
                visualizer = visualize.__new__(visualize)
                visualizer.csv_dir = Path(self.output_dir)
                records = forecast.assign(Date=date)
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


class ForecastGraphTest(unittest.TestCase):
    def plot_from_csv(self, visualizer_class, filename, rows):
        from kursach011_modules.visualizer import plt

        results = pd.DataFrame(rows, columns=['Date', 'WMO', 'Model', 'Fact', 'Forecast'])
        results['Forecast_Mode'] = ANNUAL_FORECAST_MODE
        plotted = {}

        def capture_graph(*args, **kwargs):
            for line in plt.gca().lines:
                dates = pd.DatetimeIndex(line.get_xdata())
                plotted[line.get_label()] = pd.Series(line.get_ydata(), index=dates)

        with TemporaryDirectory() as output:
            visualizer = visualizer_class(output, output)
            visualizer.csv_dir.mkdir()
            results.to_csv(visualizer.csv_dir / filename, index=False)
            with (
                patch.object(visualizer, '_calculate_metrics'),
                patch.object(visualizer, '_save_map_netcdf_and_png'),
                patch.object(plt, 'savefig', side_effect=capture_graph),
            ):
                visualizer.run()

        return pd.DataFrame(plotted)

    def test_standard_graph_uses_paired_stations_and_keeps_monthly_gaps(self):
        from kursach011_modules.visualizer import RU_TITLES, StandardForecastVisualizer

        model = 'K-Neighbors'
        rows = [
            ('2024-01-01', 'A', model, 0., 4.),
            ('2024-01-01', 'B', model, 60., 40.),
            ('2024-01-01', 'C', model, np.nan, 100.),
            ('2024-02-01', 'A', model, 10., np.nan),
            ('2024-02-01', 'C', model, np.nan, 120.),
            ('2024-04-01', 'B', model, 12., 11.),
        ]
        plotted = self.plot_from_csv(StandardForecastVisualizer, forecast_filename('METEO_RU', 'Precip'), rows)

        expected = pd.DataFrame({
            RU_TITLES['Fact']: [30., np.nan, np.nan, 12.],
            RU_TITLES[model]: [22., np.nan, np.nan, 11.],
        }, index=pd.date_range('2024-01-01', periods=4, freq='MS'))
        pd.testing.assert_frame_equal(plotted, expected, check_freq=False)

    def test_validation_graph_uses_intersection_of_stations_for_all_models(self):
        from kursach011_modules.visualizer import RU_TITLES, ValidationVisualizer

        rf, knn = 'Random Forest', 'K-Neighbors'
        rows = [
            ('2019-01-01', 'A', rf, 60., 40.),
            ('2019-01-01', 'A', knn, 60., 45.),
            ('2019-01-01', 'B', rf, 90., 80.),
            ('2019-01-01', 'C', knn, 20., 10.),
            ('2019-02-01', 'A', rf, 15., 10.),
            ('2019-02-01', 'B', knn, 15., 10.),
            ('2019-03-01', 'B', rf, 20., 15.),
            ('2019-03-01', 'B', knn, 20., 18.),
            ('2019-04-01', 'B', rf, np.nan, 15.),
            ('2019-04-01', 'B', knn, np.nan, 18.),
        ]
        plotted = self.plot_from_csv(ValidationVisualizer, validation_filename('METEO_RU', 'Precip'), rows)

        expected = pd.DataFrame({
            RU_TITLES['Fact']: [60., np.nan, 20., np.nan],
            RU_TITLES[rf]: [40., np.nan, 15., np.nan],
            RU_TITLES[knn]: [45., np.nan, 18., np.nan],
        }, index=pd.date_range('2019-01-01', periods=4, freq='MS'))
        pd.testing.assert_frame_equal(plotted, expected, check_freq=False)


class ForecastMapTest(unittest.TestCase):
    def test_seasonal_pearson_skips_failed_masks_and_keeps_sparse_station_markers(self):
        import geopandas as gpd
        from matplotlib.collections import PathCollection, QuadMesh
        from matplotlib.figure import Figure
        from rioxarray.raster_array import RasterArray
        from shapely.geometry import box

        from kursach011_modules.visualizer import BaseVisualizer

        metrics = pd.DataFrame({
            'Latitude': [50., 51.], 'Longitude': [40., 41.], 'N_Pairs': [3, 2], 'Correlation': [0.5, -1.],
        })

        with TemporaryDirectory() as output:
            visualizer = BaseVisualizer(output, output, 'maps_test')
            rendered = {}

            def capture_map(figure, path, **kwargs):
                ax = figure.axes[0]
                rendered[Path(path).name] = (ax.get_xlabel(), [type(item) for item in ax.collections])

            def save(name):
                visualizer._save_map_netcdf_and_png(
                    metrics, 'Correlation', name, 'METEO_RU', 'DTR', 'K-Neighbors', seasonal=True,
                )

            messages = StringIO()
            with patch.object(Figure, 'savefig', capture_map), redirect_stdout(messages):
                save('missing_mask')
                visualizer.mask = gpd.GeoSeries([box(39, 49, 42, 52)], crs='EPSG:4326')
                with patch.object(RasterArray, 'clip', side_effect=ValueError('invalid geometry')):
                    save('failed_clip')
                save('sparse_stations')

            self.assertIn('missing_mask', messages.getvalue())
            self.assertIn('не найден слой стран', messages.getvalue())
            self.assertIn('failed_clip', messages.getvalue())
            self.assertIn('invalid geometry', messages.getvalue())
            self.assertEqual(set(rendered), {'sparse_stations.png', 'sparse_stations.tif'})
            self.assertEqual(list(visualizer.out_maps_nc.glob('*.nc')), [visualizer.out_maps_nc / 'sparse_stations.nc'])
            for xlabel, collection_types in rendered.values():
                self.assertNotIn('0,05°', xlabel)
                self.assertIn(PathCollection, collection_types)
                self.assertNotIn(QuadMesh, collection_types)

    def test_winter_maps_and_graph_use_december_from_the_previous_year(self):
        from kursach011_modules.visualizer import StandardForecastVisualizer

        records = pd.DataFrame([
            ('2023-01-01', 5., 50.),
            ('2023-12-01', 10., 10.),
            ('2024-01-01', 20., 20.),
            ('2024-02-01', 30., 30.),
            ('2024-04-01', 40., 40.),
            ('2024-12-01', 50., 90.),
        ], columns=['Date', 'Fact', 'Forecast']).assign(
            WMO='A', Latitude=50., Longitude=40., Model='K-Neighbors', Source_Version=CRU_SOURCE_VERSION,
        )
        cases = (('2023-12', True), ('2023', True), (None, True), ('2023-12', False))
        for december_file, has_february in cases:
            with self.subTest(december_file=december_file, has_february=has_february), TemporaryDirectory() as output:
                visualizer = StandardForecastVisualizer.__new__(StandardForecastVisualizer)
                visualizer.csv_dir = Path(output)
                current_year = records.loc[records.Date.str.startswith('2024')]
                if not has_february:
                    current_year = current_year.loc[current_year.Date.ne('2024-02-01')]
                current_year.to_csv(visualizer.csv_dir / 'CRU_CEDA_Precip_2024.csv', index=False)
                if december_file is not None:
                    records.loc[records.Date.str.startswith(december_file)].to_csv(
                        visualizer.csv_dir / f'CRU_CEDA_Precip_{december_file}.csv', index=False,
                    )

                messages = StringIO()
                with (
                    patch.object(visualizer, '_plot_full_graph') as graph,
                    patch.object(visualizer, '_save_map_netcdf_and_png') as save_map,
                    redirect_stdout(messages),
                ):
                    visualizer.run(2024)

                annual_dates = graph.call_args_list[0].args[0].Date.dt.strftime('%Y-%m-%d').tolist()
                self.assertEqual(annual_dates, current_year.Date.tolist())
                maps = {call.args[2]: call.args for call in save_map.call_args_list}
                self.assertIn('CRU_CEDA_Precip_K-Neighbors_2024_Spring_Accuracy', maps)
                winter_maps = {name for name in maps if '_Winter_' in name}
                if december_file is None or not has_february:
                    self.assertFalse(winter_maps)
                    self.assertEqual(graph.call_count, 1)
                    self.assertIn('зима пропущена', messages.getvalue())
                    continue

                winter_prefix = 'CRU_CEDA_Precip_K-Neighbors_2023-2024_Winter'
                self.assertEqual(winter_maps, {f'{winter_prefix}_Accuracy', f'{winter_prefix}_Correlation'})
                winter_accuracy = maps[f'{winter_prefix}_Accuracy']
                self.assertEqual(winter_accuracy[0].N_Pairs.tolist(), [3])
                self.assertEqual(winter_accuracy[0].Accuracy.tolist(), [100.])
                period = 'Зима 2023/24 (декабрь 2023 — февраль 2024)'
                self.assertIn(period, winter_accuracy[6])
                self.assertEqual(graph.call_count, 2)
                winter_graph = graph.call_args_list[1].args
                self.assertEqual(winter_graph[0].Date.dt.strftime('%Y-%m').tolist(), ['2023-12', '2024-01', '2024-02'])
                self.assertEqual(winter_graph[3:], (period, 'Graph_CRU_CEDA_Precip_2023-2024_Winter'))

    def test_station_metrics_keep_existing_tolerances_and_score_colocated_stations_separately(self):
        from kursach011_modules.visualizer import BaseVisualizer

        records = pd.DataFrame({
            'WMO': list('ABCDE'), 'Latitude': 50, 'Longitude': 40, 'Date': '2024-01-01',
            'Fact': [0., 50., 50., 0., np.nan], 'Forecast': [3., 60., 60.01, 5.01, 0.],
        })
        visualizer = BaseVisualizer.__new__(BaseVisualizer)
        for target, accuracy in [('Precip', [100., 100., 0., 0., np.nan]), ('DTR', [100., 0., 0., 0., np.nan])]:
            with self.subTest(target=target):
                metrics = visualizer._calculate_metrics(records, target).set_index('WMO')
                self.assertEqual(metrics['N_Pairs'].tolist(), [1, 1, 1, 1, 0])
                np.testing.assert_allclose(metrics['Accuracy'], accuracy)
                self.assertTrue(metrics['Correlation'].isna().all())

    def test_maps_use_available_pairs_and_show_counts_in_both_modes(self):
        import xarray as xr
        import geopandas as gpd
        from matplotlib.collections import PathCollection, QuadMesh
        from matplotlib.figure import Figure
        from matplotlib.text import Annotation
        from PIL import Image
        from shapely.geometry import box

        from kursach011_modules.visualizer import StandardForecastVisualizer, ValidationVisualizer

        records = pd.DataFrame([
            (6, 'A', 50, 40, 0., 0.),
            (7, 'A', 50, 40, np.nan, 1.),
            (8, 'A', 50, 40, 5., np.nan),
            (6, 'B', 52, 40, 0., 1.),
            (7, 'B', 52, 40, 0., 2.),
            (8, 'B', 52, 40, 5., np.inf),
            (6, 'C', 51, 43.25, np.nan, 1.),
            (6, 'D', 50, 43, 0., 1.),
            (7, 'D', 50, 43, 5., 6.),
            (6, 'E', 52, 43, 5., 5.),
            (7, 'E', 52, 43, 6., 6.),
            (8, 'E', 52, 43, 7., 5.),
            (6, 'F', 51, 44, 1., 5.),
            (7, 'F', 51, 44, 5., 1.),
        ], columns=['Month', 'WMO', 'Latitude', 'Longitude', 'Fact', 'Forecast'])
        records = records.assign(Model='K-Neighbors', Forecast_Mode=ANNUAL_FORECAST_MODE)
        cases = (
            (StandardForecastVisualizer, forecast_filename('METEO_RU', 'Precip'), 2024,
             'METEO_RU_Precip_K-Neighbors_2024_Summer'),
            (ValidationVisualizer, validation_filename('METEO_RU', 'Precip'), 2020,
             'ValidMap_METEO_RU_Precip_K-Neighbors'),
        )
        expected_counts = {(40, 50): '1', (40, 52): '2', (43.25, 51): '0', (43, 50): '2', (43, 52): '3', (44, 51): '2'}
        original_savefig = Figure.savefig

        for visualizer_class, filename, year, prefix in cases:
            with self.subTest(mode=visualizer_class.__name__), TemporaryDirectory() as output:
                visualizer = visualizer_class(output, output)
                visualizer.mask = gpd.GeoSeries([box(39, 49, 43.5, 53)], crs='EPSG:4326')
                visualizer.csv_dir.mkdir()
                dates = pd.to_datetime(records[['Month']].assign(Year=year, Day=1))
                records.assign(Date=dates).to_csv(visualizer.csv_dir / filename, index=False)
                rendered = {}

                def capture_map(figure, path, **kwargs):
                    if Path(path).parent in (visualizer.out_maps_png, visualizer.out_maps_tiff):
                        ax = figure.axes[0]
                        rendered[Path(path).name] = (
                            {text.xy: text.get_text() for text in ax.texts if isinstance(text, Annotation)},
                            {
                                tuple(point) for collection in ax.collections if collection.get_label() == 'Нет оценки'
                                for point in collection.get_offsets()
                            },
                            ' '.join([ax.get_title(), *(text.get_text() for text in ax.texts)]),
                            ax.get_xlabel(),
                            any(isinstance(collection, QuadMesh) for collection in ax.collections),
                            {
                                tuple(point): value for collection in ax.collections
                                if isinstance(collection, PathCollection) and collection.get_array() is not None
                                for point, value in zip(collection.get_offsets(), collection.get_array())
                            },
                        )
                    return original_savefig(figure, path, **kwargs)

                with (
                    patch.object(Figure, 'savefig', capture_map),
                    patch.object(visualizer, '_save_map_netcdf_and_png', wraps=visualizer._save_map_netcdf_and_png)
                    as save_map,
                ):
                    visualizer.run()

                metrics = save_map.call_args_list[0].args[0].set_index('WMO').sort_index()
                self.assertEqual(metrics['N_Pairs'].tolist(), [1, 2, 0, 2, 3, 2])
                np.testing.assert_allclose(metrics['Accuracy'], [100., 100., np.nan, 100., 100., 100.])
                np.testing.assert_allclose(metrics['Correlation'], [np.nan, np.nan, np.nan, 1., 0., -1.], atol=1e-12)

                for metric in ('Correlation', 'Accuracy'):
                    stem = f'{prefix}_{metric}'
                    with Image.open(visualizer.out_maps_png / f'{stem}.png') as png:
                        self.assertEqual(png.format, 'PNG')
                    with Image.open(visualizer.out_maps_tiff / f'{stem}.tif') as tiff:
                        self.assertEqual(tiff.format, 'TIFF')
                    seasonal_pearson = metric == 'Correlation' and year == 2024
                    for ext in ('png', 'tif'):
                        counts, unscored, text, xlabel, has_fill, colored = rendered[f'{stem}.{ext}']
                        self.assertEqual(counts, expected_counts)
                        expected_unscored = {(43.25, 51)} if metric == 'Accuracy' else {(40, 50), (40, 52), (43.25, 51)}
                        self.assertEqual(unscored, expected_unscored)
                        self.assertEqual('До трёх пар' in text, seasonal_pearson)
                        self.assertTrue(has_fill)
                        self.assertIn('0,05°', xlabel)
                        if metric == 'Correlation':
                            self.assertEqual(colored.keys(), {(43, 50), (43, 52), (44, 51)})
                            np.testing.assert_allclose(list(colored.values()), [1., 0., -1.], atol=1e-12)
                    with xr.open_dataset(visualizer.out_maps_nc / f'{stem}.nc', decode_coords='all') as dataset:
                        grid = dataset[metric]
                        finite = grid.values[np.isfinite(grid.values)]
                        self.assertEqual(grid.attrs['interpolation'], 'cubic')
                        self.assertEqual(grid.attrs['grid_step_degrees'], 0.05)
                        self.assertTrue(grid.where(grid.longitude > 43.5, drop=True).isnull().all().item())
                        self.assertGreater(finite.size, 0)
                        self.assertTrue(grid.sel(latitude=51, longitude=43.25, method='nearest').notnull().item())
                        if metric == 'Correlation':
                            self.assertTrue(grid.sel(latitude=51, longitude=42, method='nearest').isnull().item())
                        if metric == 'Accuracy':
                            np.testing.assert_allclose(finite, 100.)
