"""Прямой прогноз на календарный год из одной точки выпуска."""

import numpy as np
import pandas as pd

from .monthly_features import prepare_forecast_period

FORECAST_MONTHS = tuple(range(1, 13))
ANNUAL_FORECAST_MODE = 'direct_annual'
STATION_COLUMNS = ['WMO', 'Latitude', 'Longitude']


def prepare_annual_features(monthly, final_year):
    """Формирует январские строки с признаками по декабрь предыдущего года."""
    first_year = monthly['Date'].dt.year.min()
    dates = pd.date_range(f'{first_year}-01-01', f'{final_year}-01-01', freq='YS')
    _, forecasts = prepare_forecast_period(monthly, dates)
    return forecasts


def annual_targets(monthly, anchors, target_var):
    """Сопоставляет каждой январской строке факты за январь–декабрь."""
    by_year = monthly.pivot(index=['WMO', 'Year'], columns='Month', values=target_var)
    keys = pd.MultiIndex.from_arrays([anchors['WMO'], anchors['Date'].dt.year])
    return by_year.reindex(index=keys, columns=FORECAST_MONTHS).to_numpy(dtype=float)


def annual_splits(anchors, targets, test_years):
    """Для обучения берёт только завершённые годы с полным вектором целей."""
    years = anchors['Date'].dt.year.to_numpy()
    complete_targets = np.isfinite(targets).all(axis=1)
    for year in test_years:
        yield (years < year) & complete_targets, years == year


def annual_prediction_rows(anchors, predictions, facts, model_name, target_var):
    """Разворачивает 12 выходов модели в месячные строки для CSV и отчёта."""
    repeats = len(FORECAST_MONTHS)
    row_indices = np.repeat(np.arange(len(anchors)), repeats)
    rows = anchors.iloc[row_indices][STATION_COLUMNS].reset_index(drop=True)
    dates = pd.to_datetime({
        'year': anchors['Date'].dt.year.repeat(repeats).to_numpy(),
        'month': np.tile(FORECAST_MONTHS, len(anchors)),
        'day': 1,
    })
    rows.insert(0, 'Date', dates)
    rows['Forecast'] = predictions.ravel()
    rows['Fact'] = facts.ravel()
    rows['Model'] = model_name
    rows['Target_Var'] = target_var
    return rows


def print_monthly_errors(paired):
    """Показывает MAE и покрытие по каждому горизонту годового прогноза."""
    abs_errors = (paired['Forecast'] - paired['Fact']).abs()
    scores = abs_errors.groupby(paired['Date'].dt.month).agg(['mean', 'size'])
    scores = scores.reindex(FORECAST_MONTHS)
    for month, (mae, count) in scores.iterrows():
        if pd.isna(count):
            print(f'    Месяц {month:02d}: нет пар с фактом')
        else:
            print(f'    Месяц {month:02d}: MAE={mae:.4f}, пар с фактом: {int(count)}')
