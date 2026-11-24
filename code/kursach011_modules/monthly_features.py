# -*- coding: utf-8 -*-
"""
Расчёт календарных признаков для прогноза и подбора параметров моделей.

Для каждой станции рассчитываются лаги на 1, 2, 3 и 12 месяцев и средние
за 3, 6 и 12 предыдущих календарных месяцев. Пропущенные месяцы остаются
пустыми и не сдвигают даты наблюдений.
"""

import numpy as np
import pandas as pd


def model_feature_columns(df):
    """Выбирает колонки, которые получает модель вместо исходных значений месяца."""
    excluded = {
        'Date', 'WMO', 'Year', 'Month', 'Quality', 'Tmin', 'Tavg', 'Tmax',
        'DTR', 'Precip', 'snow_depth', 'Latitude', 'Longitude',
    }
    return [column for column in df.columns if column not in excluded]


def select_model_rows(df, target_var=None):
    """Оставляет строки с полными признаками и, если нужно, с известной целью."""
    required = model_feature_columns(df)
    if target_var is not None:
        required.append(target_var)
    return df.dropna(subset=required)


def create_forecast_features(df, forecast_date):
    """Рассчитывает признаки для истории и целевого месяца без его наблюдений."""
    history = df.loc[df['Date'] < forecast_date]
    future = history[['WMO', 'Latitude', 'Longitude']].drop_duplicates().copy()
    future['Date'] = forecast_date
    return create_time_features(pd.concat([history, future], ignore_index=True))


def prepare_forecast_period(df, forecast_dates):
    """Возвращает историю и строки прогноза с общим расчётом лагов для всех дат."""
    dates = pd.DataFrame({'Date': pd.DatetimeIndex(forecast_dates).unique()})
    history = df.loc[df['Date'] < dates['Date'].max()]
    station = ['WMO', 'Latitude', 'Longitude']
    first_seen = history.groupby(station, as_index=False, observed=True)['Date'].min()
    candidates = first_seen.rename(columns={'Date': 'FirstDate'}).merge(dates, how='cross')
    candidates = candidates.loc[candidates['FirstDate'] < candidates['Date'], station + ['Date']]

    # Добавляем отсутствующие месяцы для прогноза, сохраняя факты для последующих лагов.
    keys = ['WMO', 'Date']
    observed = pd.MultiIndex.from_frame(history[keys])
    missing = candidates.loc[~pd.MultiIndex.from_frame(candidates[keys]).isin(observed)]
    calendar = pd.concat([history, missing], ignore_index=True)
    features = create_time_features(calendar)

    # Для прогноза берём только календарь и вычисленные признаки, без фактов месяца.
    derived = [col for col in features if col not in calendar]
    future = candidates.merge(features[keys + derived], on=keys, validate='one_to_one', sort=True)
    return features.loc[features.index < len(history)], future.reindex(columns=features.columns)


def create_time_features(df):
    """Добавляет временные признаки к таблице с одной строкой на станцию и месяц."""
    df = df.sort_values(['WMO', 'Date']).copy()
    months = df['Date'].dt.to_period('M')
    month_angle = 2 * np.pi * months.dt.month / 12
    df['Month_sin'] = np.sin(month_angle)
    df['Month_cos'] = np.cos(month_angle)

    variables = [var for var in ('DTR', 'Precip', 'snow_depth') if var in df.columns]
    index_names = ['WMO', 'CalendarMonth']
    observed_months = pd.MultiIndex.from_arrays([df['WMO'], months], names=index_names)
    if not observed_months.is_unique:
        raise ValueError('Для каждой станции нужна одна строка на календарный месяц')

    monthly_values = df[variables].set_axis(observed_months)
    if not df.empty:
        # Временно достраиваем пропущенные месяцы отдельно для каждой станции.
        bounds = months.groupby(df['WMO'], sort=False, observed=True).agg(['min', 'max'])
        station_calendars = [
            pd.MultiIndex.from_product(
                [[wmo], pd.period_range(first_month, last_month, freq='M')], names=index_names
            )
            for wmo, first_month, last_month in bounds.itertuples(name=None)
        ]
        calendar = station_calendars[0].append(station_calendars[1:])
        monthly_values = monthly_values.reindex(calendar)

    stations = monthly_values.groupby(level='WMO', sort=False, observed=True)
    lags = {lag: stations.shift(lag) for lag in (1, 2, 3, 12)}

    # Сдвиг на месяц исключает из окна текущий месяц.
    history = lags[1].groupby(level='WMO', sort=False, observed=True)
    means = {
        window: history.rolling(window, min_periods=1).mean().droplevel(0)
        for window in (3, 6, 12)
    }

    # Возвращаем признаки только для строк исходной таблицы.
    lags = {lag: values.reindex(observed_months) for lag, values in lags.items()}
    means = {window: values.reindex(observed_months) for window, values in means.items()}
    for var in variables:
        for lag, values in lags.items():
            df[f'{var}_lag_{lag}'] = values[var].to_numpy()
        for window, values in means.items():
            df[f'{var}_roll_mean_{window}'] = values[var].to_numpy()

    return df
