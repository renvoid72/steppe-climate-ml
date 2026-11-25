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


def paired_predictions(predictions, *, value_columns=('Fact', 'Forecast')):
    """Оставляет датированные пары с известными фактом и прогнозом."""
    pairs = predictions[['Date', *value_columns]].dropna()
    finite = np.isfinite(pairs[list(value_columns)].to_numpy(dtype=float)).all(axis=1)
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


def station_month_climatology(monthly, forecast_years):
    """Считает среднее по станции и месяцу на истории до года прогноза."""
    targets = ['DTR', 'Precip']
    history = monthly[['WMO', *targets]].replace([np.inf, -np.inf], np.nan).assign(
        Year=monthly['Date'].dt.year,
        Month=monthly['Date'].dt.month,
    )
    normals = []
    for year in forecast_years:
        # В месячном режиме к дате выпуска также известны только предыдущие
        # годы для целевого календарного месяца.
        previous_years = history.loc[history['Year'] < year]
        means = previous_years.groupby(['WMO', 'Month'], as_index=False, observed=True)[targets].mean()
        normals.append(means.assign(Year=year))

    result = pd.concat(normals, ignore_index=True)
    result['Date'] = pd.to_datetime(result[['Year', 'Month']].assign(day=1))
    return result.set_index(['WMO', 'Date'])[targets]


def calculate_climatology_comparison(predictions, climatology, target_var):
    """Сравнивает модель и климатологию на общем наборе станций и месяцев."""
    if climatology is None:
        return calculate_forecast_metrics(paired_predictions(predictions), target_var)

    dated = predictions.assign(Date=pd.to_datetime(predictions['Date']))
    aligned = dated.join(climatology[target_var].rename('Climatology'), on=['WMO', 'Date'], validate='many_to_one')
    common = paired_predictions(aligned, value_columns=('Fact', 'Forecast', 'Climatology'))
    model_metrics = calculate_forecast_metrics(common, target_var)
    climate_metrics = calculate_forecast_metrics(common.assign(Forecast=common['Climatology']), target_var)
    for model, climate in zip(model_metrics, climate_metrics, strict=True):
        model.update({f'{name} климатологии': climate[name] for name in ('R2', 'MAE', 'RMSE')})
        climate_mae = climate['MAE']
        model['Снижение MAE, %'] = 100 * (1 - model['MAE'] / climate_mae) if climate_mae > 0 else np.nan
    return model_metrics


def format_forecast_metrics(rows, *, padding=''):
    table = pd.DataFrame(rows).to_string(index=False, float_format='{:.4f}'.format, na_rep='—')
    return indent(table, padding)
