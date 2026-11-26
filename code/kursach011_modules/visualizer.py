# -*- coding: utf-8 -*-
import pandas as pd
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib import patheffects
from matplotlib.text import Text
from scipy.interpolate import griddata
from scipy.spatial import QhullError
import xarray as xr
from itertools import product
from pathlib import Path
import warnings

from .forecast_engine import read_prediction_csv, read_validation_csv
from .forecast_metrics import paired_predictions, regional_forecast_summary
from .project_settings import (
    ANALYSIS_YEAR, VALIDATION_START_YEAR, VALIDATION_END_YEAR, forecast_filename, validation_filename,
)

DATASETS = ('METEO_RU', 'CRU_CEDA')
TARGETS = ('DTR', 'Precip')

try:
    import geopandas as gpd
    from shapely.geometry import mapping
    import rioxarray
    HAS_GEO = True
except ImportError:
    HAS_GEO = False

warnings.filterwarnings('ignore')

# --- СЛОВАРЬ ПЕРЕВОДА ---
RU_TITLES = {
    'DTR': 'Суточная амплитуда температуры (DTR)',
    'Precip': 'Осадки',
    'METEO_RU': 'Данные Метеостанций РФ',
    'CRU_CEDA': 'Данные CRU CEDA',
    'Random Forest': 'Случайный Лес (RF)',
    'K-Neighbors': 'Метод Ближайших Соседей (KNN)',
    'MLP Neural Network': 'Нейросеть (MLP)',
    'Winter': 'Зима', 'Spring': 'Весна', 
    'Summer': 'Лето', 'Autumn': 'Осень',
    'Correlation': 'Коэффициент Корреляции',
    'Accuracy': 'Оправдываемость прогноза (%)',
    'Fact': 'Фактические данные'
}

MODEL_COLORS = {
    'Random Forest': '#1f77b4',      
    'K-Neighbors': '#2ca02c',        
    'MLP Neural Network': '#d62728'  
}

class BaseVisualizer:
    def __init__(self, base_dir, results_dir, subfolder_name):
        self.base_dir = Path(base_dir)
        self.results_dir = Path(results_dir)
        self.work_dir = self.results_dir / subfolder_name
        self.csv_dir = self.work_dir / "csv_data"
        
        # Папки выхода
        self.out_plots = self.work_dir / "plots"
        self.out_maps_nc = self.work_dir / "maps"
        self.out_maps_png = self.work_dir / "maps_png"
        self.out_maps_tiff = self.work_dir / "maps_tiff" # Карты TIFF (300 dpi)
        
        for p in [self.out_plots, self.out_maps_nc, self.out_maps_png, self.out_maps_tiff]:
            p.mkdir(parents=True, exist_ok=True)
            
        print(f"[{subfolder_name}] Инициализация визуализатора...")
        
        self.mask = None
        self.world_map = None
        self._load_geo_data()

    def _load_geo_data(self):
        if not HAS_GEO: return
        
        shapefile = self.base_dir / "maps_countries" / "ne_10m_admin_0_countries.shp"
        if not shapefile.exists(): 
            shapefile = self.base_dir / "maps" / "ne_10m_admin_0_countries.shp"
        
        eco_file = self.base_dir / "maps_ecoregions" / "Ecoregions2017.shp"
        if not eco_file.exists():
            eco_file = self.base_dir / "maps" / "Ecoregions2017.shp"

        if shapefile.exists():
            try:
                self.world_map = gpd.read_file(shapefile)
            except: pass

        if shapefile.exists() and eco_file.exists():
            try:
                countries = gpd.read_file(shapefile)
                russia = countries[countries['NAME'].isin(['Russia', 'Russian Federation'])]
                eco = gpd.read_file(eco_file)
                biome_col = 'BIOME_NUM' if 'BIOME_NUM' in eco.columns else 'BIOME'
                steppes = eco[eco[biome_col] == 8]
                
                if steppes.crs != russia.crs: steppes = steppes.to_crs(russia.crs)
                mask_gdf = gpd.overlay(steppes, russia, how='intersection')
                if not mask_gdf.empty:
                    self.mask = mask_gdf.geometry
            except: pass

    def _calculate_metrics(self, df, target_var):
        station_columns = ['WMO', 'Latitude', 'Longitude']
        located = np.isfinite(df[['Latitude', 'Longitude']]).all(axis=1)
        metrics = []
        for station, group in df.loc[located].groupby(station_columns):
            pairs = paired_predictions(group)
            count = len(pairs)
            fact, forecast = pairs['Fact'], pairs['Forecast']
            correlation = np.nan
            if count >= 2 and fact.nunique() > 1 and forecast.nunique() > 1:
                correlation = fact.corr(forecast)

            threshold = 3.0 if target_var == 'DTR' else np.maximum(5.0, 0.2 * fact)
            accuracy = (forecast.sub(fact).abs() <= threshold).mean() * 100
            metrics.append((*station, count, correlation, accuracy))

        return pd.DataFrame(metrics, columns=[*station_columns, 'N_Pairs', 'Correlation', 'Accuracy'])

    @staticmethod
    def _annotate_pair_counts(ax, metrics_df):
        ax.apply_aspect()
        renderer = ax.figure.canvas.get_renderer()
        pixels_per_point = renderer.points_to_pixels(1)
        station_x, station_y = ax.transData.transform(metrics_df[['Longitude', 'Latitude']]).T
        offsets = sorted(product(range(-48, 49, 4), repeat=2), key=lambda offset: offset[0] ** 2 + offset[1] ** 2)
        used_boxes = []

        for station in metrics_df.itertuples(index=False):
            label = ax.annotate(
                str(station.N_Pairs), (station.Longitude, station.Latitude),
                textcoords='offset points', xytext=(0, 0), ha='center', va='center', fontsize=7, zorder=5,
                path_effects=[patheffects.withStroke(linewidth=1.6, foreground='white')],
                arrowprops={'arrowstyle': '-', 'color': '#333333', 'linewidth': 0.5, 'shrinkB': 3},
            )
            label.update_positions(renderer)
            center_box = Text.get_window_extent(label, renderer).padded(1.5 * pixels_per_point)
            chosen = None
            # Выбираем ближайшее свободное место, учитывая маркеры и уже размещённые подписи.
            for dx, dy in offsets:
                box = center_box.translated(dx * pixels_per_point, dy * pixels_per_point)
                if any(box.overlaps(used) for used in used_boxes):
                    continue
                if chosen is None:
                    chosen = (dx, dy), box
                marker_box = box.padded(3 * pixels_per_point)
                covers_marker = np.any(
                    (marker_box.xmin < station_x) & (station_x < marker_box.xmax)
                    & (marker_box.ymin < station_y) & (station_y < marker_box.ymax)
                )
                if not covers_marker:
                    chosen = (dx, dy), box
                    break
            offset, box = chosen if chosen is not None else ((0, 0), center_box)
            label.set_position(offset)
            label.arrow_patch.set_visible(max(map(abs, offset)) > 8)
            used_boxes.append(box)

    def _save_map_netcdf_and_png(
        self, metrics_df, metric_name, filename_base, dataset, var, model, extra_info="", *, seasonal=False,
    ):
        if metrics_df.empty:
            return

        # 1. Сетка
        lat_min, lat_max = metrics_df['Latitude'].min() - 0.5, metrics_df['Latitude'].max() + 0.5
        lon_min, lon_max = metrics_df['Longitude'].min() - 0.5, metrics_df['Longitude'].max() + 0.5
        
        grid_lat, grid_lon = np.mgrid[lat_min:lat_max:0.05, lon_min:lon_max:0.05]
        has_estimate = np.isfinite(metrics_df[metric_name])
        scored = metrics_df.loc[has_estimate]
        unscored = metrics_df.loc[~has_estimate]
        # В сетку попадают только точки с рассчитанной метрикой.
        grid = np.full(grid_lat.shape, np.nan)
        if len(scored) >= 3:
            try:
                grid = griddata(
                    scored[['Latitude', 'Longitude']], scored[metric_name], (grid_lat, grid_lon), method='cubic',
                )
            except QhullError:
                # При совпадающих или коллинеарных точках оценка остаётся видна на маркерах.
                pass

        if metric_name == 'Correlation':
            cmap, vmin, vmax = 'coolwarm', -1, 1
        elif metric_name == 'Accuracy':
            cmap, vmin, vmax = 'RdYlGn', 0, 100
        else:
            cmap, vmin, vmax = 'viridis', None, None
        if vmin is not None:
            grid = np.clip(grid, vmin, vmax)
        
        fig = None
        try:
            # 3. Xarray
            da = xr.DataArray(
                grid,
                coords={'latitude': grid_lat[:,0], 'longitude': grid_lon[0,:]},
                dims=('latitude', 'longitude'),
                name=metric_name
            )
            da.rio.write_crs("epsg:4326", inplace=True)
            
            # Обрезка
            if self.mask is not None:
                try:
                    da = da.rio.clip(self.mask.apply(mapping), da.rio.crs, drop=False)
                except: pass
            
            da.to_netcdf(self.out_maps_nc / f"{filename_base}.nc")
            
            # --- 4. ОТРИСОВКА PNG (ФИНАЛЬНЫЙ СТИЛЬ) ---
            fig, ax = plt.subplots(figsize=(18, 6))
            
            # А. ФОН: Серый материк + Белые границы (ОЧЕНЬ ТОНКИЕ)
            # zorder=0 (самый низ)
            if self.world_map is not None:
                self.world_map.plot(ax=ax, 
                                  color='#e0e0e0',       # Светло-серый
                                  edgecolor='white',     # Белые границы
                                  linewidth=0.2,         # <=== ЕЩЕ ТОНЬШЕ
                                  zorder=0)
            
            # Г. ДАННЫЕ
            im = da.plot(ax=ax, cmap=cmap, vmin=vmin, vmax=vmax, add_colorbar=False, zorder=1, alpha=1.0)
            if not scored.empty:
                ax.scatter(
                    scored['Longitude'], scored['Latitude'], c=scored[metric_name], cmap=im.cmap, norm=im.norm,
                    edgecolors='#333333', linewidths=0.5, s=16, zorder=3,
                )
            if not unscored.empty:
                ax.scatter(
                    unscored['Longitude'], unscored['Latitude'], color='#595959', marker='X',
                    edgecolors='white', linewidths=0.5, s=26, label='Нет оценки', zorder=4,
                )
                ax.legend(loc='lower left')
            if not scored.empty and not np.isfinite(da.values).any():
                ax.text(
                    0.99, 0.02, 'Значения показаны в точках станций', transform=ax.transAxes,
                    ha='right', va='bottom', fontsize=9,
                )
            
            # Б. СЕТКА (GRID):
            # zorder=2 (ПОВЕРХ ДАННЫХ)
            # linestyle=':' (точки) - чтобы не перегружать
            ax.grid(True, linestyle=':', alpha=0.4, color='black', linewidth=0.8, zorder=2)

            # Д. ЛЕГЕНДА
            cbar = plt.colorbar(im, ax=ax, fraction=0.015, pad=0.02, shrink=0.7)
            cbar.set_label(RU_TITLES.get(metric_name, metric_name), fontsize=10)
            
            # Е. ЗАГОЛОВОК
            ru_model = RU_TITLES.get(model, model)
            ru_var = RU_TITLES.get(var, var)
            ru_dataset = RU_TITLES.get(dataset, dataset)
            ru_metric = RU_TITLES.get(metric_name, metric_name)
            
            title = f"{ru_metric}\n{ru_model} | {ru_var}\n{ru_dataset} {extra_info}"
            if seasonal and metric_name == 'Correlation':
                title += '\nДо трёх пар на станцию. При двух парах с изменяющимися рядами r = −1 или +1.'
            ax.set_title(title, fontsize=12, fontweight='bold', pad=10)
            ax.set_xlabel('Долгота\nЧисло у станции — количество пар «факт — прогноз»', fontsize=10)
            ax.set_ylabel("Широта", fontsize=10)
            
            # Ж. ЗУМ
            ax.set_xlim(lon_min - 10, lon_max + 15) 
            ax.set_ylim(lat_min - 8, lat_max + 12)
            
            fig.tight_layout(rect=(0, 0.05, 1, 0.95))
            self._annotate_pair_counts(ax, metrics_df)
            
            # 1. Сохраняем PNG (обычное качество)
            plt.savefig(self.out_maps_png / f"{filename_base}.png", dpi=200)
            
            # 2. Сохраняем TIFF (300 DPI для куратора)
            plt.savefig(
                self.out_maps_tiff / f"{filename_base}.tif", 
                dpi=300, 
                format='tiff', 
                pil_kwargs={'compression': 'tiff_lzw'}
            )
            
        except Exception as e:
            print(f"    Ошибка рисования {filename_base}: {e}")
        finally:
            if fig is not None:
                plt.close(fig)

    def _plot_full_graph(self, df, dataset, var, year_info, filename):
        models = df['Model'].dropna().unique()
        dates = pd.date_range(df['Date'].min(), df['Date'].max(), freq='MS')
        monthly = regional_forecast_summary(df, models).reindex(dates)

        plt.figure(figsize=(12, 6))

        plt.plot(dates, monthly['Fact'], label=RU_TITLES['Fact'], color='black', linewidth=3, marker='o')
        for model in models:
            plt.plot(
                dates, monthly[model], label=RU_TITLES.get(model, model),
                linestyle='--', linewidth=2, color=MODEL_COLORS.get(model),
            )
        
        ru_var = RU_TITLES.get(var, var)
        ru_dataset = RU_TITLES.get(dataset, dataset)
        
        plt.title(f"Прогноз: {ru_var}\n{ru_dataset} ({year_info})", fontsize=14)
        plt.ylabel(f"Значение ({'°C' if var=='DTR' else 'мм'})")
        plt.legend()
        plt.grid(True, alpha=0.5)
        plt.savefig(self.out_plots / f"{filename}.png", dpi=150)
        plt.close()

# Классы-наследники
class StandardForecastVisualizer(BaseVisualizer):
    def __init__(self, base_dir, results_dir):
        super().__init__(base_dir, results_dir, "standard_forecast")

    def _plot_season_maps(self, rows, dataset, var, season, period, title):
        for model, forecasts in rows.groupby('Model', sort=False):
            metrics = self._calculate_metrics(forecasts, var)
            for metric in ('Correlation', 'Accuracy'):
                filename = f"{dataset}_{var}_{model}_{period}_{season}_{metric}"
                self._save_map_netcdf_and_png(
                    metrics, metric, filename, dataset, var, model, title, seasonal=True,
                )

    def _plot_winter(self, year_rows, dataset, var, year):
        previous_year = year - 1
        december_file = self.csv_dir / forecast_filename(dataset, var, previous_year, month=12)
        if not december_file.is_file():
            december_file = self.csv_dir / forecast_filename(dataset, var, previous_year)
        if not december_file.is_file():
            print(f"  {dataset} / {var}: нет прогноза за декабрь {previous_year}, зима пропущена.")
            return

        previous_rows = read_prediction_csv(december_file, dataset)
        previous_rows['Date'] = pd.to_datetime(previous_rows['Date'])
        winter_months = pd.period_range(f'{previous_year}-12', periods=3, freq='M')
        december_rows = previous_rows.loc[previous_rows['Date'].dt.to_period('M').eq(winter_months[0])]
        jan_feb_rows = year_rows.loc[year_rows['Date'].dt.to_period('M').isin(winter_months[1:])]
        winter = pd.concat([december_rows, jan_feb_rows], ignore_index=True)

        months_by_model = winter['Date'].dt.to_period('M').groupby(winter['Model']).nunique()
        complete_models = months_by_model.index[months_by_model.eq(len(winter_months))]
        year_models = pd.Index(year_rows['Model'].dropna().unique())
        for model in year_models.difference(complete_models, sort=False):
            print(f"  {dataset} / {var} / {model}: неполный набор зимних месяцев, зима пропущена.")
        winter = winter.loc[winter['Model'].isin(complete_models)]
        if winter.empty:
            return

        period = f'{previous_year}-{year}'
        title = f'Зима {previous_year}/{year % 100:02d} (декабрь {previous_year} — февраль {year})'
        self._plot_full_graph(winter, dataset, var, title, f'Graph_{dataset}_{var}_{period}_Winter')
        self._plot_season_maps(winter, dataset, var, 'Winter', period, f'\n{title}')

    def run(self, year=ANALYSIS_YEAR):
        print("\n--- [Viz] Стандартный Прогноз (Сезонные карты) ---")
        seasons = {'Spring': [3, 4, 5], 'Summer': [6, 7, 8], 'Autumn': [9, 10, 11]}

        for dataset, var in product(DATASETS, TARGETS):
            f = self.csv_dir / forecast_filename(dataset, var, year)
            if not f.is_file():
                continue
            print(f"  Обработка: {f.name}...")
            try:
                df = read_prediction_csv(f, dataset)
                if 'Fact' not in df.columns: continue
                df['Date'] = pd.to_datetime(df['Date'])
                df = df.loc[df['Date'].dt.year.eq(year)]
                if df.empty:
                    continue

                self._plot_full_graph(df, dataset, var, year, f"Graph_{dataset}_{var}_{year}")

                for season, months in seasons.items():
                    season_rows = df.loc[df['Date'].dt.month.isin(months)]
                    title = f"({RU_TITLES[season]} {year})"
                    self._plot_season_maps(season_rows, dataset, var, season, year, title)
                self._plot_winter(df, dataset, var, year)
            except Exception as e: print(f"  Ошибка: {e}")

class ValidationVisualizer(BaseVisualizer):
    def __init__(self, base_dir, results_dir):
        super().__init__(base_dir, results_dir, "validation_results")

    def run(self, start_year=VALIDATION_START_YEAR, end_year=VALIDATION_END_YEAR):
        print("\n--- [Viz] Прямой прогноз на 12 месяцев (Карты за весь период оценки) ---")
        years_range = f'{start_year}-{end_year}'
        for dataset, var in product(DATASETS, TARGETS):
            f = self.csv_dir / validation_filename(dataset, var, start_year, end_year)
            if not f.is_file():
                continue
            print(f"  Обработка: {f.name}...")
            try:
                df = read_validation_csv(f, dataset)
                df['Date'] = pd.to_datetime(df['Date'])
                df = df.loc[df['Date'].dt.year.between(start_year, end_year)]
                if df.empty:
                    continue
                
                self._plot_full_graph(df, dataset, var, f"Прямой прогноз на 12 мес., {years_range}", f"ValidGraph_{dataset}_{var}")

                for model in df['Model'].unique():
                    df_mod = df[df['Model'] == model]
                    metrics = self._calculate_metrics(df_mod, var)
                    for met in ['Correlation', 'Accuracy']:
                        fname = f"ValidMap_{dataset}_{var}_{model}_{met}"
                        self._save_map_netcdf_and_png(metrics, met, fname, dataset, var, model, f"(Среднее за {years_range})")
            except Exception as e: print(f"  Ошибка: {e}")
