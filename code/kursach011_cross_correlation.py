# -*- coding: utf-8 -*-
"""
Скрипт кросс-корреляционного анализа (Rosgidromet vs CRU CEDA).
Выполняет сравнение наземных наблюдений Росгидрометра и глобального архива CRU CEDA,
а также кросс-корреляцию наблюдений Росгидрометра с прогностическими моделями, обученными на CRU.
Расчеты производятся по четырем сезонам года.
Задание: задание corr.md
"""

import sys
import os
from pathlib import Path
import numpy as np
import pandas as pd
import warnings
from scipy.interpolate import griddata
import xarray as xr
from shapely.geometry import mapping
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

warnings.filterwarnings('ignore')

# --- КОНСТАНТЫ ПУТЕЙ ---
PROJECT_DIR = Path(__file__).resolve().parent.parent
RESULTS_DIR = PROJECT_DIR / "output011_modules"
CORR_DIR = RESULTS_DIR / "cross_correlation_analysis"

# Добавляем путь к коду в sys.path, чтобы импортировать BaseVisualizer
sys.path.append(str(PROJECT_DIR / "code"))
try:
    from kursach011_modules.visualizer import BaseVisualizer
    HAS_VISUALIZER = True
except ImportError as e:
    HAS_VISUALIZER = False
    print(f"ВНИМАНИЕ: Не удалось импортировать BaseVisualizer: {e}")

# --- СЛОВАРИ ПЕРЕВОДА ДЛЯ ОФОРМЛЕНИЯ ---
RU_VARS = {
    'DTR': 'Суточная амплитуда температуры (DTR)',
    'Precip': 'Осадки'
}

RU_VARS_SHORT = {
    'DTR': 'DTR',
    'Precip': 'Осадки'
}

RU_VARS_UNITS = {
    'DTR': '°C',
    'Precip': 'мм'
}

RU_MODELS = {
    'Random Forest': 'Случайный Лес (RF)',
    'K-Neighbors': 'Метод Ближайших Соседей (KNN)',
    'MLP Neural Network': 'Нейросеть (MLP)'
}

SEASONS = {
    'Winter': [12, 1, 2],
    'Spring': [3, 4, 5],
    'Summer': [6, 7, 8],
    'Autumn': [9, 10, 11]
}

RU_SEASONS = {
    'Winter': 'Winter',
    'Spring': 'Spring',
    'Summer': 'Summer',
    'Autumn': 'Autumn'
}

RU_SEASONS_NAME = {
    'Winter': 'Зима',
    'Spring': 'Весна',
    'Summer': 'Лето',
    'Autumn': 'Осень'
}


class CrossCorrelationAnalyzer(BaseVisualizer):
    def __init__(self, base_dir, results_dir):
        """Инициализация путей, гео-маски и подложки через BaseVisualizer."""
        if not HAS_VISUALIZER:
            raise ImportError("Необходим класс BaseVisualizer из модуля visualizer.py")
        
        super().__init__(base_dir, results_dir, "cross_correlation_analysis")
        print(f"   -> Рабочая папка анализа: {self.work_dir}")

    def load_and_merge_data(self):
        """Загружает CSV долгосрочной валидации и объединяет их по ['Date', 'WMO', 'Model', 'Target_Var']."""
        print("\n2. Загрузка и объединение данных валидации за 1990-2020 гг...")
        
        csv_configs = {
            'DTR': {
                'meteo': self.results_dir / 'validation_results' / 'csv_data' / 'Validation_METEO_RU_DTR_1990-2020.csv',
                'cru': self.results_dir / 'validation_results' / 'csv_data' / 'Validation_CRU_CEDA_DTR_1990-2020.csv'
            },
            'Precip': {
                'meteo': self.results_dir / 'validation_results' / 'csv_data' / 'Validation_METEO_RU_Precip_1990-2020.csv',
                'cru': self.results_dir / 'validation_results' / 'csv_data' / 'Validation_CRU_CEDA_Precip_1990-2020.csv'
            }
        }
        
        merged_datasets = {}
        
        for param, paths in csv_configs.items():
            if not paths['meteo'].exists() or not paths['cru'].exists():
                raise FileNotFoundError(
                    f"Не найдены исходные файлы валидации для параметра {param}. "
                    "Убедитесь, что вы предварительно запустили расчет в kursach011.py!"
                )
            
            print(f"   Обработка параметра {param}...")
            df_meteo = pd.read_csv(paths['meteo'])
            df_cru = pd.read_csv(paths['cru'])
            
            # Обеспечиваем типы данных для корректного слияния
            df_meteo['WMO'] = df_meteo['WMO'].astype(int)
            df_cru['WMO'] = df_cru['WMO'].astype(int)
            df_meteo['Date'] = pd.to_datetime(df_meteo['Date'])
            df_cru['Date'] = pd.to_datetime(df_cru['Date'])
            
            # Внутреннее слияние по Date, WMO, Model и Target_Var
            merged_df = pd.merge(
                df_meteo,
                df_cru,
                on=['Date', 'WMO', 'Model', 'Target_Var'],
                suffixes=('_meteo', '_cru')
            )
            
            # Чистим пропуски
            merged_df = merged_df.dropna(subset=['Fact_meteo', 'Fact_cru', 'Forecast_cru'])
            
            print(f"      Наземные наблюдения (Meteo): {len(df_meteo)} строк")
            print(f"      Глобальный архив (CRU):     {len(df_cru)} строк")
            print(f"      Объединенных записей:        {len(merged_df)} строк")
            
            merged_datasets[param] = merged_df
            
        return merged_datasets

    def plot_correlation_map(self, lats, lons, values, title, filename_base, season_name):
        """Интерполирует значения корреляции станций на регулярную сетку и строит карту в сезонной папке."""
        # Определение границ сетки
        lat_min, lat_max = lats.min() - 0.5, lats.max() + 0.5
        lon_min, lon_max = lons.min() - 0.5, lons.max() + 0.5
        grid_lat, grid_lon = np.mgrid[lat_min:lat_max:0.05, lon_min:lon_max:0.05]
        points = np.column_stack((lats, lons))
        
        # Кубическая интерполяция
        grid = griddata(points, values, (grid_lat, grid_lon), method='cubic')
        grid = np.clip(grid, -1.0, 1.0)
        
        # Создание DataArray и обрезка по маске степей
        da = xr.DataArray(
            grid,
            coords={'latitude': grid_lat[:, 0], 'longitude': grid_lon[0, :]},
            dims=('latitude', 'longitude')
        )
        da.rio.write_crs("epsg:4326", inplace=True)
        if self.mask is not None:
            try:
                da = da.rio.clip(self.mask.apply(mapping), da.rio.crs, drop=False)
            except:
                pass
        
        # Отрисовка карты
        fig, ax = plt.subplots(figsize=(18, 6))
        
        if self.world_map is not None:
            self.world_map.plot(ax=ax, color='#e0e0e0', edgecolor='white', linewidth=0.2, zorder=0)
            
        # Устанавливаем границы шкалы: coolwarm центрируется на 0
        vmin = 0.0 if np.nanmin(grid) >= 0 else -1.0
        vmax = 1.0
        
        da.plot(ax=ax, cmap='coolwarm', vmin=vmin, vmax=vmax, add_colorbar=False, zorder=1, alpha=1.0)
        ax.grid(True, linestyle=':', alpha=0.4, color='black', linewidth=0.8, zorder=2)
        
        # Легенда
        im = ax.collections[-1]
        cbar = plt.colorbar(im, ax=ax, fraction=0.015, pad=0.02, shrink=0.7)
        cbar.set_label("Коэффициент линейной корреляции Пирсона", fontsize=10)
        
        ax.set_title(title, fontsize=12, fontweight='bold', pad=10, wrap=True)
        ax.set_xlabel("Долгота", fontsize=10)
        ax.set_ylabel("Широта", fontsize=10)
        
        # Ограничение видимой области по границам станций
        ax.set_xlim(lon_min - 10, lon_max + 15)
        ax.set_ylim(lat_min - 8, lat_max + 12)
        
        plt.tight_layout()
        
        # Создаем сезонные директории для карт
        png_season_dir = self.out_maps_png / season_name
        tiff_season_dir = self.out_maps_tiff / season_name
        png_season_dir.mkdir(parents=True, exist_ok=True)
        tiff_season_dir.mkdir(parents=True, exist_ok=True)
        
        png_path = png_season_dir / f"{filename_base}.png"
        tiff_path = tiff_season_dir / f"{filename_base}.tif"
        
        plt.savefig(png_path, dpi=200)
        plt.savefig(tiff_path, dpi=300, format='tiff', pil_kwargs={'compression': 'tiff_lzw'})
        plt.close()

    def plot_scatter(self, x_vals, y_vals, r_coeff, xlabel, ylabel, title, filename, season_name):
        """Строит диаграмму рассеяния с линией 1:1 и трендом регрессии в сезонной папке."""
        plt.figure(figsize=(8, 8))
        
        # Точки рассеяния
        plt.scatter(x_vals, y_vals, color='#1f77b4', alpha=0.25, s=6, label='Месячные значения наблюдений')
        
        # Линия идеального соответствия 1:1
        min_val = min(x_vals.min(), y_vals.min())
        max_val = max(x_vals.max(), y_vals.max())
        plt.plot([min_val, max_val], [min_val, max_val], color='black', linestyle='--', linewidth=1.5, label='Идеальное соответствие 1:1')
        
        # Линия линейного регрессионного тренда
        try:
            slope, intercept = np.polyfit(x_vals, y_vals, 1)
            x_trend = np.linspace(min_val, max_val, 100)
            y_trend = slope * x_trend + intercept
            plt.plot(x_trend, y_trend, color='red', linestyle='-', linewidth=2, label=f'Линейный тренд (y = {slope:.2f}x + {intercept:.2f})')
        except Exception as e:
            print(f"      Ошибка расчета тренда: {e}")
            
        plt.xlabel(xlabel, fontsize=11)
        plt.ylabel(ylabel, fontsize=11)
        plt.title(title, fontsize=10.5, fontweight='bold', pad=12, wrap=True)
        
        # Текстовый блок со значением коэффициента корреляции
        textstr = f"Коэффициент корреляции:\n$r = {r_coeff:.4f}$"
        props = dict(boxstyle='round', facecolor='wheat', alpha=0.85)
        plt.gca().text(0.05, 0.95, textstr, transform=plt.gca().transAxes, fontsize=11,
                    verticalalignment='top', bbox=props)
                    
        plt.legend(loc='lower right', fontsize=10)
        plt.grid(True, linestyle=':', alpha=0.5)
        plt.tight_layout()
        
        # Создаем сезонную директорию для графиков рассеяния
        plots_season_dir = self.out_plots / season_name
        plots_season_dir.mkdir(parents=True, exist_ok=True)
        
        plot_path = plots_season_dir / f"{filename}.png"
        plt.savefig(plot_path, dpi=200)
        plt.close()

    def run_analysis(self, merged_datasets):
        """Основной цикл сезонных расчетов пространственной и кросс-корреляции."""
        summary_data = {
            'block1': {},
            'block2': {}
        }
        
        # ====================================================================
        # БЛОК 1: Сравнение датасетов (Meteo_Fact vs CRU_Fact) по сезонам
        # ====================================================================
        print("\n3. Расчет Блока 1 (Meteo_Fact vs CRU_Fact) по сезонам...")
        for param, df in merged_datasets.items():
            summary_data['block1'][param] = {}
            print(f"   Параметр: {param}...")
            
            # Убираем дублирование месяцев из-за наличия строк по разным моделям
            df_unique = df.drop_duplicates(subset=['Date', 'WMO']).copy()
            df_unique['Month'] = df_unique['Date'].dt.month
            
            for season_key, months in SEASONS.items():
                print(f"      Сезон: {RU_SEASONS_NAME[season_key]}...")
                df_season = df_unique[df_unique['Month'].isin(months)].copy()
                
                if df_season.empty:
                    print(f"         ВНИМАНИЕ: Нет данных для сезона {season_key}!")
                    continue
                
                # Расчет корреляции по станциям
                station_corrs = []
                station_lats = []
                station_lons = []
                
                for wmo, group in df_season.groupby('WMO'):
                    if len(group) >= 12:  # Требуем репрезентативный объем данных для сезона
                        r = group['Fact_meteo'].corr(group['Fact_cru'])
                        if not np.isnan(r):
                            station_corrs.append(r)
                            station_lats.append(group['Latitude_meteo'].iloc[0])
                            station_lons.append(group['Longitude_meteo'].iloc[0])
                
                station_corrs = np.array(station_corrs)
                station_lats = np.array(station_lats)
                station_lons = np.array(station_lons)
                
                if len(station_corrs) == 0:
                    print(f"         ВНИМАНИЕ: Недостаточно станций для расчета корреляции в сезоне {season_key}!")
                    continue
                
                mean_station_r = np.mean(station_corrs)
                overall_r = df_season['Fact_meteo'].corr(df_season['Fact_cru'])
                
                summary_data['block1'][param][season_key] = {
                    'mean_station_r': mean_station_r,
                    'overall_r': overall_r
                }
                
                print(f"         Средняя корреляция по {len(station_corrs)} станциям: {mean_station_r:.4f}")
                print(f"         Общая корреляция (по всем точкам):           {overall_r:.4f}")
                
                # 1. Построение карты
                title_map = (
                    f"Пространственная корреляция фактических данных:\n"
                    f"Наблюдения Росгидрометра vs Архив CRU CEDA\n"
                    f"Параметр: {RU_VARS[param]} | Сезон: {RU_SEASONS_NAME[season_key]} | 1990-2020 гг."
                )
                file_map = f"Base_Correlation_Meteo_vs_CRU_{param}_{season_key}"
                self.plot_correlation_map(station_lats, station_lons, station_corrs, title_map, file_map, season_key)
                
                # 2. Построение Scatter Plot
                xlabel = f"Фактические значения Росгидромета, {RU_VARS_UNITS[param]}"
                ylabel = f"Фактические значения архива CRU CEDA, {RU_VARS_UNITS[param]}"
                title_scatter = (
                    f"Сопоставление фактических данных:\n"
                    f"Наблюдения Росгидрометра vs Архив CRU CEDA\n"
                    f"Параметр: {RU_VARS[param]} | Сезон: {RU_SEASONS_NAME[season_key]} | 1990-2020 гг."
                )
                file_scatter = f"Scatter_Meteo_vs_CRU_{param}_{season_key}"
                self.plot_scatter(df_season['Fact_meteo'].values, df_season['Fact_cru'].values, overall_r, 
                                  xlabel, ylabel, title_scatter, file_scatter, season_key)
            
        # ====================================================================
        # БЛОК 2: Кросс-корреляция (Meteo_Fact vs CRU_Forecast) по сезонам
        # ====================================================================
        print("\n4. Расчет Блока 2 (Meteo_Fact vs CRU_Forecast) по сезонам...")
        for param, df in merged_datasets.items():
            summary_data['block2'][param] = {}
            print(f"   Параметр: {param}...")
            
            df['Month'] = df['Date'].dt.month
            
            for model_name in RU_MODELS.keys():
                summary_data['block2'][param][model_name] = {}
                df_model = df[df['Model'] == model_name].copy()
                
                if df_model.empty:
                    print(f"      ВНИМАНИЕ: Данные для модели {model_name} отсутствуют!")
                    continue
                
                print(f"      Модель: {RU_MODELS[model_name]}...")
                
                for season_key, months in SEASONS.items():
                    print(f"         Сезон: {RU_SEASONS_NAME[season_key]}...")
                    df_season = df_model[df_model['Month'].isin(months)].copy()
                    
                    if df_season.empty:
                        print(f"            ВНИМАНИЕ: Нет данных для сезона {season_key}!")
                        continue
                    
                    # Расчет корреляции по станциям
                    station_corrs = []
                    station_lats = []
                    station_lons = []
                    
                    for wmo, group in df_season.groupby('WMO'):
                        if len(group) >= 12:
                            r = group['Fact_meteo'].corr(group['Forecast_cru'])
                            if not np.isnan(r):
                                station_corrs.append(r)
                                station_lats.append(group['Latitude_meteo'].iloc[0])
                                station_lons.append(group['Longitude_meteo'].iloc[0])
                    
                    station_corrs = np.array(station_corrs)
                    station_lats = np.array(station_lats)
                    station_lons = np.array(station_lons)
                    
                    if len(station_corrs) == 0:
                        print(f"            ВНИМАНИЕ: Недостаточно станций для расчета в сезоне {season_key}!")
                        continue
                    
                    mean_station_r = np.mean(station_corrs)
                    overall_r = df_season['Fact_meteo'].corr(df_season['Forecast_cru'])
                    
                    summary_data['block2'][param][model_name][season_key] = {
                        'mean_station_r': mean_station_r,
                        'overall_r': overall_r
                    }
                    
                    print(f"            Средняя корреляция по {len(station_corrs)} станциям: {mean_station_r:.4f}")
                    print(f"            Общая корреляция (по всем точкам):           {overall_r:.4f}")
                    
                    # Имя файла без пробелов
                    clean_model_name = model_name.replace(" ", "_")
                    
                    # 1. Построение карты
                    title_map = (
                        f"Кросс-корреляция: Наблюдения Росгидрометра vs Прогноз CRU ({RU_MODELS[model_name]})\n"
                        f"Параметр: {RU_VARS[param]} | Сезон: {RU_SEASONS_NAME[season_key]} | Период: 1990-2020 гг."
                    )
                    file_map = f"Cross_Correlation_Meteo_vs_CRU_{param}_{clean_model_name}_{season_key}"
                    self.plot_correlation_map(station_lats, station_lons, station_corrs, title_map, file_map, season_key)
                    
                    # 2. Построение Scatter Plot
                    xlabel = f"Фактические значения Росгидромета, {RU_VARS_UNITS[param]}"
                    ylabel = f"Прогноз модели на основе CRU ({RU_MODELS[model_name]}), {RU_VARS_UNITS[param]}"
                    title_scatter = (
                        f"Кросс-сопоставление:\n"
                        f"Наблюдения Росгидрометра vs Прогноз CRU ({RU_MODELS[model_name]})\n"
                        f"Параметр: {RU_VARS[param]} | Сезон: {RU_SEASONS_NAME[season_key]} | 1990-2020 гг."
                    )
                    file_scatter = f"Scatter_Cross_Meteo_vs_CRU_{param}_{clean_model_name}_{season_key}"
                    self.plot_scatter(df_season['Fact_meteo'].values, df_season['Forecast_cru'].values, overall_r,
                                      xlabel, ylabel, title_scatter, file_scatter, season_key)
                
        # ====================================================================
        # БЛОК 3: Сохранение сводного отчета
        # ====================================================================
        print("\n5. Сохранение сводного текстового отчета...")
        summary_path = self.work_dir / "correlation_summary.txt"
        
        with open(summary_path, 'w', encoding='utf-8') as f:
            f.write("========================================================================\n")
            f.write("СВОДНЫЙ ОТЧЕТ ПО СЕЗОННОМУ КРОСС-КОРРЕЛЯЦИОННОМУ АНАЛИЗУ (1990-2020 гг.)\n")
            f.write("Сравнение: Наблюдения Росгидрометра (Meteo) vs База данных CRU CEDA\n")
            f.write("========================================================================\n\n")
            
            f.write("БЛОК 1: СРАВНЕНИЕ ДАТАСЕТОВ (Meteo_Fact vs CRU_Fact)\n")
            f.write("------------------------------------------------------------------------\n")
            for param in ['DTR', 'Precip']:
                f.write(f"Параметр: {RU_VARS[param]} ({param})\n")
                for season_key in SEASONS.keys():
                    data = summary_data['block1'][param].get(season_key)
                    if data:
                        f.write(f"  Сезон: {RU_SEASONS_NAME[season_key]} ({season_key})\n")
                        f.write(f"    * Средний коэффициент корреляции по станциям: {data['mean_station_r']:8.4f}\n")
                        f.write(f"    * Общий коэффициент корреляции по всем точкам: {data['overall_r']:8.4f}\n")
                f.write("\n")
                
            f.write("\nБЛОК 2: КРОСС-КОРРЕЛЯЦИЯ (Meteo_Fact vs CRU_Forecast)\n")
            f.write("------------------------------------------------------------------------\n")
            for param in ['DTR', 'Precip']:
                f.write(f"Параметр: {RU_VARS[param]} ({param})\n")
                for model_name in RU_MODELS.keys():
                    f.write(f"  Модель: {RU_MODELS[model_name]} ({model_name})\n")
                    for season_key in SEASONS.keys():
                        data = summary_data['block2'][param].get(model_name, {}).get(season_key)
                        if data:
                            f.write(f"    Сезон: {RU_SEASONS_NAME[season_key]} ({season_key})\n")
                            f.write(f"      * Средний коэффициент корреляции по станциям: {data['mean_station_r']:8.4f}\n")
                            f.write(f"      * Общий коэффициент корреляции по всем точкам: {data['overall_r']:8.4f}\n")
                    f.write("\n")
                f.write("\n")
                
            f.write("========================================================================\n")
            
        print(f"   -> Сводный отчет сохранен в: {summary_path}")

    def cleanup_directories(self):
        """Удаляет пустые служебные папки (например, csv_data и maps), если они не заполнялись."""
        print("\n6. Очистка временных папок...")
        for folder_name in ["csv_data", "maps"]:
            folder_path = self.work_dir / folder_name
            if folder_path.exists() and folder_path.is_dir():
                if not any(folder_path.iterdir()):
                    try:
                        folder_path.rmdir()
                        print(f"   -> Удалена пустая папка: {folder_path.name}")
                    except Exception as e:
                        print(f"   ! Не удалось удалить папку {folder_path.name}: {e}")


def main():
    print(">>> СТАРТ РАБОТЫ КРОСС-КОРРЕЛЯЦИОННОГО МОДУЛЯ <<<\n")
    
    try:
        # Инициализация анализатора
        analyzer = CrossCorrelationAnalyzer(PROJECT_DIR, RESULTS_DIR)
        
        # Шаг 1. Загрузка и объединение данных
        merged_datasets = analyzer.load_and_merge_data()
        
        # Шаг 2. Расчет корреляций и построение графиков/карт
        analyzer.run_analysis(merged_datasets)
        
        # Шаг 3. Очистка неиспользуемых папок
        analyzer.cleanup_directories()
        
        print(f"\n>>> РАБОТА ЗАВЕРШЕНА! Все результаты кросс-корреляции в: {CORR_DIR} <<<")
        
    except Exception as e:
        print(f"\n!!! КРИТИЧЕСКАЯ ОШИБКА: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    main()
