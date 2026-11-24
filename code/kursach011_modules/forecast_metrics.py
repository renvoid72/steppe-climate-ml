"""Сводная оценка прогноза по станциям и региональным месячным средним."""

from textwrap import indent

import numpy as np
import pandas as pd
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score


def regional_forecast_summary(predictions, models):
    """Усредняет факт и прогнозы по общим станциям каждого месяца."""
    station_keys = ['Date', 'WMO']
    paired = predictions.dropna(subset=['Fact', 'Forecast'])
    station_forecasts = paired.pivot_table(
        index=station_keys, columns='Model', values='Forecast', observed=True,
    )
    common_stations = station_forecasts.reindex(columns=models).dropna()
    station_facts = paired.groupby(station_keys, observed=True)['Fact'].first()
    by_month = common_stations.join(station_facts).groupby(level='Date')
    return by_month.mean().assign(N_Stations=by_month.size())


def paired_predictions(predictions):
    """Оставляет датированные пары с известными фактом и прогнозом."""
    pairs = predictions[['Date', 'Fact', 'Forecast']].dropna()
    finite = np.isfinite(pairs[['Fact', 'Forecast']].to_numpy(dtype=float)).all(axis=1)
    return pairs.loc[finite].assign(Date=lambda rows: pd.to_datetime(rows['Date'])).dropna(subset=['Date'])


def calculate_forecast_metrics(pairs, target_var):
    """Рассчитывает метрики по парам «станция — месяц» и по календарным месяцам."""
    months = pairs['Date'].dt.to_period('M')
    regional = pairs.groupby(months)[['Fact', 'Forecast']].mean()
    unit = {'DTR': '°C', 'Precip': 'мм'}[target_var]

    rows = []
    for level, count_unit, data in (
        ('Региональные месячные средние', 'месяцы', regional),
        ('Станция — месяц', 'пары', pairs),
    ):
        count = len(data)
        fact, forecast = data['Fact'], data['Forecast']
        rows.append({
            'Уровень оценки': level,
            'N': count,
            'Ед. подсчёта': count_unit,
            'R2': r2_score(fact, forecast) if count > 1 else np.nan,
            'MAE': mean_absolute_error(fact, forecast) if count else np.nan,
            'RMSE': np.sqrt(mean_squared_error(fact, forecast)) if count else np.nan,
            'Ед. ошибки': unit,
        })
    return rows


def format_forecast_metrics(rows, *, padding=''):
    table = pd.DataFrame(rows).to_string(index=False, float_format='{:.4f}'.format, na_rep='—')
    return indent(table, padding)
