# -*- coding: utf-8 -*-
import pandas as pd
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from scipy.interpolate import griddata
import xarray as xr
from itertools import product
from pathlib import Path
import warnings
import os

from .forecast_engine import read_prediction_csv, read_validation_csv
from .forecast_metrics import regional_forecast_summary
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
        def compute(group):
            if len(group) < 2: corr = 0
            else: corr = group['Forecast'].corr(group['Fact'])
            
            abs_err = np.abs(group['Forecast'] - group['Fact'])
            if target_var == 'DTR': threshold = 3.0
            else: threshold = np.maximum(5.0, 0.2 * group['Fact'])
            
            hits = (abs_err <= threshold).sum()
            accuracy = (hits / len(group)) * 100.0
            return pd.Series({'Correlation': corr, 'Accuracy': accuracy})

        return df.groupby(['Latitude', 'Longitude']).apply(compute).reset_index().fillna(0)

    def _save_map_netcdf_and_png(self, metrics_df, metric_name, filename_base, dataset, var, model, extra_info=""):
        # 1. Сетка
        lat_min, lat_max = metrics_df['Latitude'].min() - 0.5, metrics_df['Latitude'].max() + 0.5
        lon_min, lon_max = metrics_df['Longitude'].min() - 0.5, metrics_df['Longitude'].max() + 0.5
        
        grid_lat, grid_lon = np.mgrid[lat_min:lat_max:0.05, lon_min:lon_max:0.05]
        points = metrics_df[['Latitude', 'Longitude']].values
        values = metrics_df[metric_name].values
        
        try:
            # 2. Интерполяция
            grid = griddata(points, values, (grid_lat, grid_lon), method='cubic')
            
            if metric_name == 'Accuracy': grid = np.clip(grid, 0, 100)
            elif metric_name == 'Correlation': grid = np.clip(grid, -1, 1)

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
            
            # В. ЦВЕТА
            if metric_name == 'Correlation':
                cmap, vmin, vmax = 'coolwarm', -1, 1
            elif metric_name == 'Accuracy':
                cmap, vmin, vmax = 'RdYlGn', 0, 100
            else: 
                cmap, vmin, vmax = 'viridis', None, None

            # Г. ДАННЫЕ
            # zorder=1 (поверх карты)
            da.plot(ax=ax, cmap=cmap, vmin=vmin, vmax=vmax, add_colorbar=False, zorder=1, alpha=1.0)
            
            # Б. СЕТКА (GRID):
            # zorder=2 (ПОВЕРХ ДАННЫХ)
            # linestyle=':' (точки) - чтобы не перегружать
            ax.grid(True, linestyle=':', alpha=0.4, color='black', linewidth=0.8, zorder=2)

            # Д. ЛЕГЕНДА
            im = ax.collections[-1] # Берем объект heatmap для легенды
            cbar = plt.colorbar(im, ax=ax, fraction=0.015, pad=0.02, shrink=0.7)
            cbar.set_label(RU_TITLES.get(metric_name, metric_name), fontsize=10)
            
            # Е. ЗАГОЛОВОК
            ru_model = RU_TITLES.get(model, model)
            ru_var = RU_TITLES.get(var, var)
            ru_dataset = RU_TITLES.get(dataset, dataset)
            ru_metric = RU_TITLES.get(metric_name, metric_name)
            
            title = f"{ru_metric}\n{ru_model} | {ru_var}\n{ru_dataset} {extra_info}"
            ax.set_title(title, fontsize=12, fontweight='bold', pad=10)
            ax.set_xlabel("Долгота", fontsize=10)
            ax.set_ylabel("Широта", fontsize=10)
            
            # Ж. ЗУМ
            ax.set_xlim(lon_min - 10, lon_max + 15) 
            ax.set_ylim(lat_min - 8, lat_max + 12)
            
            plt.tight_layout()
            
            # 1. Сохраняем PNG (обычное качество)
            plt.savefig(self.out_maps_png / f"{filename_base}.png", dpi=200)
            
            # 2. Сохраняем TIFF (300 DPI для куратора)
            plt.savefig(
                self.out_maps_tiff / f"{filename_base}.tif", 
                dpi=300, 
                format='tiff', 
                pil_kwargs={'compression': 'tiff_lzw'}
            )
            
            plt.close()
            
        except Exception as e:
            print(f"    Ошибка рисования {filename_base}: {e}")

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

    def run(self, year=ANALYSIS_YEAR):
        print("\n--- [Viz] Стандартный Прогноз (Сезонные карты) ---")
        seasons = {'Winter': [12, 1, 2], 'Spring': [3, 4, 5], 'Summer': [6, 7, 8], 'Autumn': [9, 10, 11]}

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
                
                for model in df['Model'].unique():
                    df_mod = df[df['Model'] == model]
                    for s_name, months in seasons.items():
                        df_seas = df_mod[df_mod['Date'].dt.month.isin(months)]
                        if df_seas.empty: continue
                        metrics = self._calculate_metrics(df_seas, var)
                        ru_season = RU_TITLES.get(s_name, s_name)
                        for met in ['Correlation', 'Accuracy']:
                            fname = f"{dataset}_{var}_{model}_{year}_{s_name}_{met}"
                            self._save_map_netcdf_and_png(metrics, met, fname, dataset, var, model, f"({ru_season} {year})")
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

                df = df.dropna(subset=['Forecast', 'Fact'])
                for model in df['Model'].unique():
                    df_mod = df[df['Model'] == model]
                    metrics = self._calculate_metrics(df_mod, var)
                    for met in ['Correlation', 'Accuracy']:
                        fname = f"ValidMap_{dataset}_{var}_{model}_{met}"
                        self._save_map_netcdf_and_png(metrics, met, fname, dataset, var, model, f"(Среднее за {years_range})")
            except Exception as e: print(f"  Ошибка: {e}")
