import pandas as pd


MAX_DTR_MISSING_DAYS = 5
MAX_DTR_MISSING_RUN = 3


def load_monthly_meteo(data_path, coords_path):
    """Читает суточные данные METEO_RU и агрегирует их по станции и месяцу."""
    columns = ['WMO', 'Year', 'Month', 'Day', 'Quality', 'Tmin', 'Tavg', 'Tmax', 'Precip']
    daily = pd.read_csv(data_path, sep=r'\s+', names=columns, na_values=[-999.9])
    calendar_columns = ['Year', 'Month', 'Day']
    dates = pd.to_datetime(daily[calendar_columns])
    if dates.isna().any() or daily.duplicated(['WMO', *calendar_columns]).any():
        raise ValueError('В суточных данных METEO_RU нужны уникальные даты для каждой станции')
    daily['DaysInMonth'] = dates.dt.days_in_month
    daily['DTR'] = daily['Tmax'] - daily['Tmin']

    coords = pd.read_csv(coords_path, sep=r'\s+')
    daily['Latitude'] = daily['WMO'].map(dict(zip(coords['WMO'], coords['Lat'])))
    daily['Longitude'] = daily['WMO'].map(dict(zip(coords['WMO'], coords['Lon'])))

    month_keys = ['WMO', 'Year', 'Month', 'Latitude', 'Longitude']
    monthly = daily.groupby(month_keys).agg(
        DTR=('DTR', 'mean'),
        Precip=('Precip', 'sum'),
        DTRDays=('DTR', 'count'),
        PrecipDays=('Precip', 'count'),
        DaysInMonth=('DaysInMonth', 'first'),
    )
    days_in_month = monthly['DaysInMonth']
    monthly['Precip'] = monthly['Precip'].where(monthly['PrecipDays'].eq(days_in_month))

    available_dtr = daily.loc[daily['DTR'].notna(), [*month_keys, 'Day']].sort_values(
        [*month_keys, 'Day']
    )
    # Разрыв перед первым известным днём считается от начала календарного месяца.
    day_steps = available_dtr.groupby(month_keys, sort=False)['Day'].diff()
    available_dtr['GapBefore'] = day_steps.fillna(available_dtr['Day']) - 1
    gaps = available_dtr.groupby(month_keys, sort=False).agg(
        MaxGapBefore=('GapBefore', 'max'), LastDay=('Day', 'max')
    ).reindex(monthly.index)
    dtr_complete = (
        (days_in_month - monthly['DTRDays']).le(MAX_DTR_MISSING_DAYS)
        & gaps['MaxGapBefore'].le(MAX_DTR_MISSING_RUN)
        & (days_in_month - gaps['LastDay']).le(MAX_DTR_MISSING_RUN)
    )
    monthly['DTR'] = monthly['DTR'].where(dtr_complete)
    return monthly[['DTR', 'Precip']].reset_index()
