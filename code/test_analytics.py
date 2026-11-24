"""Регрессионные проверки региональных Excel-отчётов."""

import unittest
from pathlib import Path
from tempfile import TemporaryDirectory
from unittest.mock import patch

import numpy as np
import pandas as pd
from openpyxl import load_workbook

import kursach011_analytics as analytics


class RegionalExcelTest(unittest.TestCase):
    def setUp(self):
        self.output = Path(self.enterContext(TemporaryDirectory()))
        self.enterContext(patch.object(analytics, 'ANALYTICS_DIR', self.output))

    def test_fact_only_report_leaves_incomplete_annual_values_blank(self):
        source = pd.DataFrame([
            (year, month, 4.0, float(month - 1), 'Западная Сибирь')
            for year in (2019, 2020, 2021)
            for month in range(1, 13)
        ], columns=['Year', 'Month', 'DTR', 'Precip', 'Region'])
        source.loc[(source['Year'] == 2020) & (source['Month'] > 9), ['DTR', 'Precip']] = np.nan
        source.loc[source['Year'] == 2021, ['DTR', 'Precip']] = np.nan

        analytics.generate_fact_only_excel_reports({'METEO_RU': source})

        for variable, complete_value in [('DTR', 4.0), ('Precip', 66.0)]:
            with self.subTest(variable=variable):
                path = self.output / f'Meteo_RU_{variable}_Fact_2019-2021.xlsx'
                with path.open('rb') as file:
                    workbook = load_workbook(file, read_only=True, data_only=True)
                    self.addCleanup(workbook.close)
                    annual = workbook['Западная Сибирь'].iter_rows(
                        min_col=8, max_col=10, max_row=4, values_only=True,
                    )
                    self.assertEqual(list(annual), [
                        ('Year', 'Fact', 'N_Months'),
                        (2019, complete_value, 12),
                        (2020, None, 9),
                        (2021, None, 0),
                    ])

    def test_comparative_report_uses_common_stations_and_shows_empty_months(self):
        records = []
        for month in (1, 2, 3, 5):
            for station, longitude, fact in [('A', 55.0, 0.0), ('B', 70.0, 100.0)]:
                for model in ('Random Forest', 'K-Neighbors', 'MLP Neural Network'):
                    if model == 'K-Neighbors' and (month == 2 or (month == 1 and station == 'B')):
                        continue
                    actual = np.nan if month == 5 else fact
                    records.append((pd.Timestamp(2020, month, 1), station, 45.0, longitude, actual, fact, model))
        source = pd.DataFrame(records, columns=['Date', 'WMO', 'Latitude', 'Longitude', 'Fact', 'Forecast', 'Model'])
        analytics.generate_excel_reports({
            'meteo_precip': {'raw': source, 'var': 'Precip', 'dataset': 'METEO_RU'},
        })

        path = self.output / f'Meteo_RU_Precip_{analytics.VALIDATION_PERIOD}.xlsx'
        with path.open('rb') as file:
            workbook = load_workbook(file, read_only=True, data_only=True)
            self.addCleanup(workbook.close)
            whole_region = list(workbook['Вся зона степей'].iter_rows(max_col=7, values_only=True))
            west_siberia = list(workbook['Западная Сибирь'].iter_rows(max_col=7, values_only=True))
            self.assertEqual(whole_region[0], (
                'Year', 'Month', 'Fact', 'Random Forest', 'K-Neighbors', 'MLP Neural Network', 'N_Stations',
            ))
            self.assertEqual(whole_region[1:], [
                (2020, 1, 0, 0, 0, 0, 1),
                (2020, 2, None, None, None, None, 0),
                (2020, 3, 50, 50, 50, 50, 2),
                (2020, 4, None, None, None, None, 0),
                (2020, 5, None, None, None, None, 0),
            ])
            self.assertEqual(west_siberia[1:], [
                (2020, 1, None, None, None, None, 0),
                (2020, 2, None, None, None, None, 0),
                (2020, 3, 100, 100, 100, 100, 1),
                (2020, 4, None, None, None, None, 0),
                (2020, 5, None, None, None, None, 0),
            ])
