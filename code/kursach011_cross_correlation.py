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
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

from kursach011_modules.forecast_engine import forecast_input_files, read_validation_csv
from kursach011_modules.meteo_data import load_monthly_meteo
from kursach011_modules.project_settings import (
    VALIDATION_END_YEAR, VALIDATION_PERIOD, VALIDATION_START_YEAR, results_dir, validation_filename,
)

warnings.filterwarnings('ignore')

# --- КОНСТАНТЫ ПУТЕЙ ---
PROJECT_DIR = Path(__file__).resolve().parent.parent
RESULTS_DIR = results_dir(PROJECT_DIR)
CORR_DIR = RESULTS_DIR / "cross_correlation_analysis"

# Добавляем путь к коду в sys.path, чтобы импортировать BaseVisualizer
sys.path.append(str(PROJECT_DIR / "code"))
try:
    from kursach011_modules.visualizer import BaseVisualizer, MAP_GRID_STEP, interpolate_map, plot_interpolated_map
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


def validation_months(rows):
    """Оставляет строки расчётного периода и приводит даты к началу месяца."""
    dates = pd.to_datetime(rows['Date']).dt.to_period('M').dt.to_timestamp()
    return rows.assign(Date=dates).loc[dates.dt.year.between(VALIDATION_START_YEAR, VALIDATION_END_YEAR)]


def finite_pairs(rows, reference):
    return rows.loc[np.isfinite(rows[['Fact_meteo', reference]]).all(axis=1)]


def summarize_correlations(rows, reference):
    station_rows = []
    for _, group in rows.groupby('WMO'):
        if len(group) < 12:
            continue
        r = group['Fact_meteo'].corr(group[reference])
        if np.isfinite(r):
            station_rows.append((group['Latitude_meteo'].iloc[0], group['Longitude_meteo'].iloc[0], r))
    stations = pd.DataFrame(station_rows, columns=['Latitude', 'Longitude', 'R'], dtype=float)
    overall_r = rows['Fact_meteo'].corr(rows[reference]) if len(rows) >= 2 else np.nan
    return stations, {
        'n_pairs': len(rows),
        'mean_station_r': stations['R'].mean(),
        'overall_r': overall_r,
    }


def write_season_summary(file, season, data, indent):
    file.write(f"{indent}Сезон: {RU_SEASONS_NAME[season]} ({season})\n")
    file.write(f"{indent}  * Число пар: {data['n_pairs']}\n")
    for key, label in (
        ('mean_station_r', 'Средний коэффициент корреляции по станциям'),
        ('overall_r', 'Общий коэффициент корреляции (по всем точкам)'),
    ):
        value = data[key]
        text = f'{value:8.4f}' if np.isfinite(value) else 'нет оценки'
        file.write(f"{indent}  * {label}: {text}\n")


class CrossCorrelationAnalyzer(BaseVisualizer):
    def __init__(self, base_dir, results_dir):
        """Инициализация путей, гео-маски и подложки через BaseVisualizer."""
        if not HAS_VISUALIZER:
            raise ImportError("Необходим класс BaseVisualizer из модуля visualizer.py")
        
        super().__init__(base_dir, results_dir, "cross_correlation_analysis")
        print(f"   -> Рабочая папка анализа: {self.work_dir}")

    def load_and_merge_data(self):
        """Сопоставляет METEO_RU с наблюдениями и прогнозами CRU по станции и месяцу."""
        print(f"\n2. Сопоставление месячных рядов за {VALIDATION_PERIOD} гг...")
        meteo = load_monthly_meteo(*forecast_input_files(self.base_dir, 'METEO_RU'))
        meteo = validation_months(meteo.assign(Date=pd.to_datetime(meteo[['Year', 'Month']].assign(day=1))))
        cru_path, = forecast_input_files(self.base_dir, 'CRU_CEDA')
        if cru_path.is_file():
            cru = pd.read_csv(cru_path)
            cru = validation_months(cru.assign(Date=pd.to_datetime(cru[['Year', 'Month']].assign(day=1))))
        else:
            print(f"   Фактические ряды CRU недоступны: {cru_path}. Запустите code/prepare_cru.py.")
            cru = meteo.iloc[:0]

        keys = ['Date', 'WMO']
        meteo_columns = [*keys, 'Latitude', 'Longitude']
        validation_dir = self.results_dir / 'validation_results' / 'csv_data'
        observation_pairs = {}
        forecast_pairs = {}
        for param in RU_VARS:
            meteo_values = meteo[[*meteo_columns, param]].rename(columns={
                param: 'Fact_meteo', 'Latitude': 'Latitude_meteo', 'Longitude': 'Longitude_meteo',
            })
            cru_values = cru[[*keys, param]].rename(columns={param: 'Fact_cru'})
            paired_facts = meteo_values.merge(cru_values, on=keys, validate='one_to_one')
            observation_pairs[param] = finite_pairs(paired_facts, 'Fact_cru')

            path = validation_dir / validation_filename('CRU_CEDA', param)
            try:
                predictions = read_validation_csv(path, 'CRU_CEDA')
            except (OSError, ValueError) as error:
                print(f"   Прогнозы CRU для {param} пропущены: {error}")
                forecast_pairs[param] = meteo_values.iloc[:0].assign(Model='', Forecast_cru=np.nan)
                continue

            predictions = validation_months(predictions.loc[predictions['Target_Var'] == param])
            predictions = predictions[[*keys, 'Model', 'Forecast']].rename(columns={'Forecast': 'Forecast_cru'})
            paired_forecasts = meteo_values.merge(predictions, on=keys, validate='one_to_many')
            forecast_pairs[param] = finite_pairs(paired_forecasts, 'Forecast_cru')

        return observation_pairs, forecast_pairs

    def plot_correlation_map(self, lats, lons, values, title, filename_base, season_name):
        """Интерполирует значения корреляции станций на регулярную сетку и строит карту в сезонной папке."""
        valid = np.isfinite(lats) & np.isfinite(lons) & np.isfinite(values)
        if not valid.any():
            print(f'  Карта {filename_base} пропущена. Нет оценок с координатами станций.')
            return
        lats, lons, values = lats[valid], lons[valid], values[valid]
        lat_min, lat_max = lats.min() - 0.5, lats.max() + 0.5
        lon_min, lon_max = lons.min() - 0.5, lons.max() + 0.5
        grid_lat, grid_lon = np.mgrid[lat_min:lat_max:MAP_GRID_STEP, lon_min:lon_max:MAP_GRID_STEP]
        points = np.column_stack((lats, lons))
        da = interpolate_map(points, values, grid_lat, grid_lon).clip(min=-1, max=1)
        vmin = -1.0 if min(values.min(), np.nanmin(da.values, initial=0)) < 0 else 0.0
        da = self.clip_map(da, filename_base)
        if da is None:
            return
        
        # Отрисовка карты
        fig, ax = plt.subplots(figsize=(18, 6))
        
        if self.world_map is not None:
            self.world_map.plot(ax=ax, color='#e0e0e0', edgecolor='white', linewidth=0.2, zorder=0)
            
        im = plot_interpolated_map(ax, da, points, values, cmap='coolwarm', vmin=vmin, vmax=1.0)
        ax.grid(True, linestyle=':', alpha=0.4, color='black', linewidth=0.8, zorder=2)
        
        # Легенда
        cbar = plt.colorbar(im, ax=ax, fraction=0.015, pad=0.02, shrink=0.7)
        cbar.set_label("Коэффициент линейной корреляции Пирсона", fontsize=10)
        
        ax.set_title(title, fontsize=12, fontweight='bold', pad=10, wrap=True)
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

    def run_analysis(self, observation_pairs, forecast_pairs):
        """Рассчитывает сезонные корреляции и сохраняет карты, графики и сводку."""
        summary_path = self.work_dir / "correlation_summary.txt"
        with summary_path.open('w', encoding='utf-8') as report:
            report.write(f"СВОДНЫЙ ОТЧЕТ ПО СЕЗОННОМУ КРОСС-КОРРЕЛЯЦИОННОМУ АНАЛИЗУ ({VALIDATION_PERIOD} гг.)\n\n")

            report.write("БЛОК 1: СРАВНЕНИЕ ДАТАСЕТОВ (Meteo_Fact vs CRU_Fact)\n")
            for param, rows in observation_pairs.items():
                print(f"   Сравнение наблюдений METEO_RU и CRU для {param}...")
                report.write(f"Параметр: {RU_VARS[param]} ({param})\n")
                for season, months in SEASONS.items():
                    seasonal = rows.loc[rows['Date'].dt.month.isin(months)]
                    summary = self.analyze_season(seasonal, param, season)
                    write_season_summary(report, season, summary, '  ')
                report.write("\n")

            report.write("БЛОК 2: КРОСС-КОРРЕЛЯЦИЯ (Meteo_Fact vs CRU_Forecast)\n")
            for param, rows in forecast_pairs.items():
                print(f"   Сравнение наблюдений METEO_RU с прогнозами CRU для {param}...")
                report.write(f"Параметр: {RU_VARS[param]} ({param})\n")
                for model, model_label in RU_MODELS.items():
                    report.write(f"  Модель: {model_label} ({model})\n")
                    model_rows = rows.loc[rows['Model'] == model]
                    for season, months in SEASONS.items():
                        seasonal = model_rows.loc[model_rows['Date'].dt.month.isin(months)]
                        summary = self.analyze_season(seasonal, param, season, model=model)
                        write_season_summary(report, season, summary, '    ')
                    report.write("\n")

        print(f"   -> Сводный отчет сохранен в: {summary_path}")

    def analyze_season(self, rows, param, season, *, model=None):
        reference = 'Fact_cru' if model is None else 'Forecast_cru'
        stations, summary = summarize_correlations(rows, reference)
        overall_r = summary['overall_r']
        if stations.empty and not np.isfinite(overall_r):
            return summary

        period = f"Параметр: {RU_VARS[param]} | Сезон: {RU_SEASONS_NAME[season]} | {VALIDATION_PERIOD} гг."
        if model is None:
            comparison = "Наблюдения METEO_RU и фактические значения CRU"
            ylabel = f"Фактические значения CRU, {RU_VARS_UNITS[param]}"
            map_name = f"Base_Correlation_Meteo_vs_CRU_{param}_{season}"
            scatter_name = f"Scatter_Meteo_vs_CRU_{param}_{season}"
        else:
            comparison = f"Наблюдения METEO_RU и прогноз CRU ({RU_MODELS[model]})"
            ylabel = f"Прогноз модели на CRU ({RU_MODELS[model]}), {RU_VARS_UNITS[param]}"
            clean_model = model.replace(" ", "_")
            map_name = f"Cross_Correlation_Meteo_vs_CRU_{param}_{clean_model}_{season}"
            scatter_name = f"Scatter_Cross_Meteo_vs_CRU_{param}_{clean_model}_{season}"

        if not stations.empty:
            self.plot_correlation_map(
                *stations.to_numpy().T, f"Корреляция по станциям\n{comparison}\n{period}", map_name, season,
            )
        if np.isfinite(overall_r):
            self.plot_scatter(
                rows['Fact_meteo'].to_numpy(), rows[reference].to_numpy(), overall_r,
                f"Фактические значения METEO_RU, {RU_VARS_UNITS[param]}", ylabel,
                f"{comparison}\n{period}", scatter_name, season,
            )
        return summary

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
        observation_pairs, forecast_pairs = analyzer.load_and_merge_data()
        
        # Шаг 2. Расчет корреляций и построение графиков/карт
        analyzer.run_analysis(observation_pairs, forecast_pairs)
        
        # Шаг 3. Очистка неиспользуемых папок
        analyzer.cleanup_directories()
        
        print(f"\n>>> РАБОТА ЗАВЕРШЕНА! Все результаты кросс-корреляции в: {CORR_DIR} <<<")
        
    except Exception as e:
        print(f"\n!!! КРИТИЧЕСКАЯ ОШИБКА: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    main()
