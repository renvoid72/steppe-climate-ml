import pandas as pd


def load_monthly_meteo(data_path, coords_path):
    """Читает суточные данные METEO_RU и агрегирует их по станции и месяцу."""
    columns = ['WMO', 'Year', 'Month', 'Day', 'Quality', 'Tmin', 'Tavg', 'Tmax', 'Precip']
    daily = pd.read_csv(data_path, sep=r'\s+', names=columns, na_values=[-999.9])
    daily['DTR'] = daily['Tmax'] - daily['Tmin']

    coords = pd.read_csv(coords_path, sep=r'\s+')
    daily['Latitude'] = daily['WMO'].map(dict(zip(coords['WMO'], coords['Lat'])))
    daily['Longitude'] = daily['WMO'].map(dict(zip(coords['WMO'], coords['Lon'])))

    grouped = daily.groupby(['WMO', 'Year', 'Month', 'Latitude', 'Longitude'])
    monthly = grouped.agg(DTR=('DTR', 'mean'), Precip=('Precip', 'sum'))
    # Сумма осадков известна, только если есть значение у каждой записанной строки.
    monthly['Precip'] = monthly['Precip'].where(grouped['Precip'].count().eq(grouped.size()))
    return monthly.reset_index()
