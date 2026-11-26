"""Проверки сезонных расчётов на неполных входных данных."""

import calendar
from contextlib import redirect_stdout
from io import StringIO
from pathlib import Path
from tempfile import TemporaryDirectory
import unittest
from unittest.mock import patch

import numpy as np
import pandas as pd

import kursach011_analytics as analytics
import kursach011_cross_correlation as correlation
import kursach011_eof as eof
from kursach011_modules.annual_forecast import ANNUAL_FORECAST_MODE
from kursach011_modules.forecast_engine import mark_prediction_source
from kursach011_modules.project_settings import cru_csv_path, validation_filename


def write_validation(root, dataset, variable, rows):
    folder = root / 'validation_results' / 'csv_data'
    folder.mkdir(parents=True, exist_ok=True)
    rows = rows.assign(Target_Var=variable, Forecast_Mode=ANNUAL_FORECAST_MODE)
    mark_prediction_source(rows, dataset)
    rows.to_csv(folder / validation_filename(dataset, variable), index=False)


class SeasonalPostprocessingTest(unittest.TestCase):
    def setUp(self):
        self.root = Path(self.enterContext(TemporaryDirectory()))
        self.output = self.root / 'output'
        self.log = StringIO()
        self.enterContext(redirect_stdout(self.log))

    def test_correlation_pairs_are_independent_of_meteo_validation_and_cru_forecast_facts(self):
        meteo = self.root / 'meteo_ru'
        meteo.mkdir()
        (meteo / 'stations_coords.txt').write_text('WMO Name Lat Lon\n1 Station 52 48\n', encoding='utf-8')
        with (meteo / 'all_steppe_data.txt').open('w', encoding='utf-8') as daily:
            for year in (1989, 2019, 2020, 2021):
                for month in (6, 7, 8):
                    for day in range(1, calendar.monthrange(year, month)[1] + 1):
                        daily.write(f'1 {year} {month} {day} 0 0 4 {month} {month - 6}\n')

        cru_csv = cru_csv_path(self.root)
        cru_csv.parent.mkdir(parents=True)
        pd.DataFrame([
            (1, year, month, 52, 48, month, np.nan if month == 6 else 10 * month)
            for year in (1989, 2019, 2020, 2021) for month in (6, 7)
        ], columns=['WMO', 'Year', 'Month', 'Latitude', 'Longitude', 'DTR', 'Precip']).to_csv(cru_csv, index=False)

        predictions = pd.DataFrame([
            ('2020-06-01', 1, 'K-Neighbors', 0.0, np.nan),
            ('2020-07-01', 1, 'K-Neighbors', 12.0, 70.0),
            ('2020-08-01', 1, 'K-Neighbors', np.nan, 80.0),
            ('2020-06-01', 1, 'Random Forest', 1.0, np.nan),
            ('2021-07-01', 1, 'K-Neighbors', 12.0, 70.0),
        ], columns=['Date', 'WMO', 'Model', 'Forecast', 'Fact'])
        write_validation(self.output, 'CRU_CEDA', 'Precip', predictions)

        with patch.object(correlation.BaseVisualizer, '_load_geo_data'):
            analyzer = correlation.CrossCorrelationAnalyzer(self.root, self.output)
        observations, forecasts = analyzer.load_and_merge_data()
        self.assertEqual(len(observations['DTR']), 4)
        self.assertEqual(len(observations['Precip']), 2)
        self.assertTrue(forecasts['DTR'].empty)
        knn = forecasts['Precip'].loc[forecasts['Precip']['Model'] == 'K-Neighbors']
        self.assertEqual(knn['Date'].tolist(), pd.to_datetime(['2020-06-01', '2020-07-01']).tolist())
        self.assertEqual(knn['Fact_meteo'].tolist(), [0, 31])
        self.assertEqual(knn['Forecast_cru'].tolist(), [0, 12])

        with patch.object(analyzer, 'plot_scatter'), patch.object(analyzer, 'plot_correlation_map'):
            analyzer.run_analysis(observations, forecasts)
        summary = (analyzer.work_dir / 'correlation_summary.txt').read_text(encoding='utf-8')
        observations_summary, forecasts_summary = summary.split('БЛОК 2')
        self.assertIn('Число пар: 4', observations_summary)
        self.assertIn('Число пар: 2', forecasts_summary)
        self.assertIn('Число пар: 1', forecasts_summary)
        self.assertIn('Число пар: 0', forecasts_summary)

    def test_eof_skips_missing_season_and_small_matrix_but_processes_other_slices(self):
        rng = np.random.default_rng(42)
        rows = pd.DataFrame([
            (date, wmo, 52 + wmo / 10, 48 + wmo / 10, rng.normal())
            for date in pd.date_range('2017-01-01', '2020-12-01', freq='MS')
            for wmo in range(4)
        ], columns=['Date', 'WMO', 'Latitude', 'Longitude', 'Fact'])
        rows = rows.assign(Model='K-Neighbors')
        write_validation(self.output, 'METEO_RU', 'DTR', rows.loc[rows['Date'].dt.month != 2])
        write_validation(self.output, 'METEO_RU', 'Precip', rows.loc[rows['WMO'] < 2])
        eof_dir = self.output / 'eof'
        with (
            patch.multiple(eof, RESULTS_DIR=self.output, EOF_DIR=eof_dir, HAS_VISUALIZER=False,
                           VALIDATION_START_YEAR=2017, VALIDATION_END_YEAR=2020),
            patch.object(eof, 'export_historical_eof_series'),
            patch.object(eof.plt, 'savefig'),
        ):
            eof.main()

        self.assertIn('Нет наблюдений для календарных месяцев: 2', self.log.getvalue())
        self.assertIn('Недостаточно данных для 3 компонент', self.log.getvalue())
        summary = (eof_dir / 'variance_summary.txt').read_text(encoding='utf-8')
        winter, other_seasons = summary.split('СЕЗОН: Весна')
        self.assertNotIn('ЕОФ-1:', winter)
        self.assertIn('ЕОФ-1:', other_seasons)
        dtr = pd.read_excel(eof_dir / 'Meteo_RU_DTR_EOF_Variance_All.xlsx')
        precip = pd.read_excel(eof_dir / 'Meteo_RU_Precip_EOF_Variance_All.xlsx')
        self.assertIn('календарных месяцев: 2', dtr.loc[0, 'Статус'])
        self.assertEqual(precip['Компонента'].tolist(), ['ЕОФ-1', 'ЕОФ-2'])

        history = rows.assign(Year=rows['Date'].dt.year, DTR=rows['Fact'])
        absent_june = history['Date'] == '2018-06-01'
        history = history.loc[~absent_june].copy()
        four_gaps = history['Date'].dt.year.eq(2017) & history['Date'].dt.month.le(4)
        history.loc[history['WMO'].eq(3) & four_gaps, 'DTR'] = np.nan
        matrix = eof.eof_calendar_matrix(history, 'DTR', 2017, 2020)
        _, _, stations = eof.fit_eof(matrix, n_components=3)
        self.assertEqual(stations.tolist(), [0, 1, 2])
        pcs = eof.calculate_regional_pcs(history, 'DTR', start_year=2017, end_year=2020)
        self.assertEqual(len(pcs), 47)
        self.assertNotIn(pd.Timestamp('2018-06-01'), pcs.index)

    def test_atlas_saves_fact_and_available_model_only(self):
        rows = pd.DataFrame([
            (pd.Timestamp(2020, 6, 1), 1, 52, 48, 0.0, 0.0, 'K-Neighbors'),
            (pd.Timestamp(2020, 6, 1), 2, 53, 49, 5.0, 4.0, 'K-Neighbors'),
        ], columns=['Date', 'WMO', 'Latitude', 'Longitude', 'Fact', 'Forecast', 'Model'])
        pivoted = analytics.pivot_forecasts(rows)
        with (
            patch.object(analytics, 'BaseVisualizer') as visualizer,
            patch.object(analytics, 'plot_interpolated_map', wraps=analytics.plot_interpolated_map) as plot,
            patch.object(analytics.plt, 'savefig') as save,
        ):
            visualizer.return_value.world_map = None
            visualizer.return_value.clip_map.side_effect = lambda data, name: data
            analytics.generate_map_atlas({'meteo': {'pivoted': pivoted, 'var': 'Precip', 'dataset': 'METEO_RU'}})

        self.assertEqual(plot.call_count, 2)
        self.assertEqual(save.call_count, 4)
        self.assertEqual([call.args[1] for call in visualizer.return_value.clip_map.call_args_list], [
            'Метео_РФ_Осадки_Факт_Июнь_2020',
            'Метео_РФ_Осадки_Ближайшие_Соседи_Июнь_2020',
        ])
