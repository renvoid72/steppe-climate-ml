# -*- coding: utf-8 -*-
"""
Аналитический модуль постобработки данных курсовой работы.
Задачи:
1. Создание структуры директорий.
2. Чтение данных долгосрочной валидации за общий период.
3. Группировка по макрорегионам России и сохранение 4 Excel-файлов (с двумя таблицами бок о бок на 5 вкладках).
4. Расчет статистических метрик точности моделей (R2, MAE, RMSE) по всей зоне степей.
5. Генерация атласа карт за общий расчётный год с динамической шкалой и кубической интерполяцией.
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

from kursach011_modules.forecast_engine import prepare_monthly_forecast_data, read_validation_csv
from kursach011_modules.forecast_metrics import (
    calculate_climatology_comparison, format_forecast_metrics, regional_forecast_summary, station_month_climatology,
)
from kursach011_modules.annual_forecast import FORECAST_MONTHS
from kursach011_modules.meteo_data import load_monthly_meteo
from kursach011_modules.project_settings import (
    ANALYSIS_YEAR, CRU_VERSION, VALIDATION_START_YEAR, VALIDATION_END_YEAR, VALIDATION_PERIOD,
    cru_csv_path, results_dir, validation_filename,
)

warnings.filterwarnings('ignore')

# --- КОНСТАНТЫ ПУТЕЙ ---
PROJECT_DIR = Path(__file__).resolve().parent.parent
RESULTS_DIR = results_dir(PROJECT_DIR)
ANALYTICS_DIR = RESULTS_DIR / "analytics_reports"
FORECAST_MODELS = ('Random Forest', 'K-Neighbors', 'MLP Neural Network')
FORECAST_VALUE_COLUMNS = ['Fact', *FORECAST_MODELS]

# Добавляем путь к коду в sys.path, чтобы импортировать BaseVisualizer
sys.path.append(str(PROJECT_DIR / "code"))
try:
    from kursach011_modules.visualizer import BaseVisualizer, MAP_GRID_STEP, interpolate_map, plot_interpolated_map
    HAS_VISUALIZER = True
except ImportError as e:
    HAS_VISUALIZER = False
    print(f"ВНИМАНИЕ: Не удалось импортировать BaseVisualizer: {e}")

# --- СЛОВАРИ ПЕРЕВОДА ДЛЯ КАРТ ---
RU_MONTHS = {
    1: "Январь", 2: "Февраль", 3: "Март", 4: "Апрель", 5: "Май", 6: "Июнь",
    7: "Июль", 8: "Август", 9: "Сентябрь", 10: "Октябрь", 11: "Ноябрь", 12: "Декабрь"
}

RU_DATASETS = {
    'METEO_RU': "Метео_РФ",
    'CRU_CEDA': "CRU_CEDA"
}

RU_VARS = {
    'DTR': "DTR",
    'Precip': "Осадки"
}

RU_MODELS_FILE = {
    'Fact': "Факт",
    'Random Forest': "Случайный_Лес",
    'K-Neighbors': "Ближайшие_Соседи",
    'MLP Neural Network': "Нейросеть"
}

RU_MODELS_TITLE = {
    'Fact': "Фактические данные",
    'Random Forest': "Модель Random Forest (RF)",
    'K-Neighbors': "Модель K-Neighbors (KNN)",
    'MLP Neural Network': "Модель MLP Neural Network"
}


def create_directories():
    """Создает структуру папок для отчетов и карт."""
    print("1. Создание структуры папок...")
    folders = [
        ANALYTICS_DIR,
        ANALYTICS_DIR / "maps_png" / "meteo_ru" / "DTR",
        ANALYTICS_DIR / "maps_png" / "meteo_ru" / "Precip",
        ANALYTICS_DIR / "maps_png" / "cru_ceda" / "DTR",
        ANALYTICS_DIR / "maps_png" / "cru_ceda" / "Precip",
        ANALYTICS_DIR / "maps_tiff" / "meteo_ru" / "DTR",
        ANALYTICS_DIR / "maps_tiff" / "meteo_ru" / "Precip",
        ANALYTICS_DIR / "maps_tiff" / "cru_ceda" / "DTR",
        ANALYTICS_DIR / "maps_tiff" / "cru_ceda" / "Precip",
    ]
    for folder in folders:
        folder.mkdir(parents=True, exist_ok=True)
    print("   -> Папки успешно созданы.")


def get_macroregion(lon):
    """Определяет макрорегион России на основе долготы."""
    if lon < 60:
        return 'Европейская часть'
    elif 60 <= lon < 90:
        return 'Западная Сибирь'
    elif 90 <= lon < 120:
        return 'Восточная Сибирь'
    else:
        return 'Дальний Восток'


def pivot_forecasts(forecasts):
    """Разворачивает результаты отдельных моделей по столбцам."""
    df = forecasts.copy()
    df['Date'] = pd.to_datetime(df['Date'])
    df['Year'] = df['Date'].dt.year
    df['Month'] = df['Date'].dt.month

    # Сводим таблицу
    pivoted = df.pivot_table(
        index=['Date', 'Year', 'Month', 'WMO', 'Latitude', 'Longitude', 'Fact'],
        columns='Model',
        values='Forecast'
    ).reset_index()

    # Проверка на наличие всех трех моделей
    for model in FORECAST_MODELS:
        if model not in pivoted.columns:
            pivoted[model] = np.nan

    # Добавляем макрорегион
    pivoted['Region'] = pivoted['Longitude'].apply(get_macroregion)
    return pivoted


def aggregate_annual_forecasts(monthly, group_columns, target_var):
    """Считает годовые значения по полным 12-месячным рядам."""
    groups = monthly.groupby(group_columns)[FORECAST_VALUE_COLUMNS]
    aggregate = groups.mean() if target_var == 'DTR' else groups.sum(min_count=len(FORECAST_MONTHS))
    return aggregate.where(groups.count().eq(len(FORECAST_MONTHS))).dropna(subset=['Fact']).reset_index()


def generate_excel_reports(dataset_info):
    """Генерирует Excel-отчеты с 5 листами и 2 таблицами бок о бок."""
    print(f"3. Генерация Excel-отчетов ({VALIDATION_PERIOD} гг.)...")
    
    regions_config = [
        ('Вся зона степей', None),
        ('Европейская часть', 'Европейская часть'),
        ('Западная Сибирь', 'Западная Сибирь'),
        ('Восточная Сибирь', 'Восточная Сибирь'),
        ('Дальний Восток', 'Дальний Восток')
    ]

    monthly_columns = ['Year', 'Month', *FORECAST_VALUE_COLUMNS, 'N_Stations']
    for info in dataset_info.values():
        raw = info['raw'].assign(Date=lambda rows: pd.to_datetime(rows['Date']))
        raw['Region'] = raw['Longitude'].apply(get_macroregion)
        present_models = set(raw['Model'].dropna())
        models = [model for model in FORECAST_MODELS if model in present_models]
        dates = pd.date_range(raw['Date'].min(), raw['Date'].max(), freq='MS') if not raw.empty else pd.DatetimeIndex([])
        var = info['var']
        dataset = info['dataset']
        
        file_name = f"{'Meteo_RU' if dataset == 'METEO_RU' else 'CRU_CEDA'}_{var}_{VALIDATION_PERIOD}.xlsx"
        output_path = ANALYTICS_DIR / file_name
        
        print(f"   Формирование отчета: {file_name}...")
        
        with pd.ExcelWriter(output_path, engine='openpyxl') as writer:
            for sheet_name, region_name in regions_config:
                region_df = raw if region_name is None else raw.loc[raw['Region'] == region_name]
                monthly = regional_forecast_summary(region_df, models)
                if not region_df.empty:
                    monthly = monthly.reindex(dates)
                monthly['N_Stations'] = monthly['N_Stations'].fillna(0).astype(int)
                monthly_table = monthly.assign(
                    Year=monthly.index.year, Month=monthly.index.month,
                ).reindex(columns=monthly_columns)
                annual_table = aggregate_annual_forecasts(monthly_table, ['Year'], var)

                monthly_table.to_excel(writer, sheet_name=sheet_name, index=False)
                annual_table.to_excel(writer, sheet_name=sheet_name, startcol=len(monthly_columns) + 1, index=False)

    print("   -> Отчеты Excel успешно сохранены.")


def load_full_fact_data():
    """Загружает фактические данные METEO_RU и CRU до общего расчётного года включительно."""
    print("3.1. Загрузка полнопериодных фактических данных...")
    fact_datasets = {}

    # 1. Метеостанции РФ
    meteo_txt = PROJECT_DIR / "meteo_ru" / "all_steppe_data.txt"
    coords_txt = PROJECT_DIR / "meteo_ru" / "stations_coords.txt"
    if meteo_txt.exists() and coords_txt.exists():
        print(f"   Загрузка фактических данных METEO_RU (1960-{ANALYSIS_YEAR} гг.)...")
        df_meteo_monthly = load_monthly_meteo(meteo_txt, coords_txt)
        df_meteo_monthly = df_meteo_monthly.loc[df_meteo_monthly['Year'] <= ANALYSIS_YEAR].copy()
        df_meteo_monthly['Region'] = df_meteo_monthly['Longitude'].apply(get_macroregion)
        fact_datasets['METEO_RU'] = df_meteo_monthly
    else:
        print("   ! ВНИМАНИЕ: Файлы данных Метео РФ не найдены.")

    # 2. CRU CEDA
    cru_csv = cru_csv_path(PROJECT_DIR)
    if cru_csv.exists():
        print(f"   Загрузка фактических данных CRU TS {CRU_VERSION} (1901-{ANALYSIS_YEAR} гг.)...")
        df_cru = pd.read_csv(cru_csv)
        df_cru = df_cru.loc[df_cru['Year'] <= ANALYSIS_YEAR].copy()
        df_cru['Region'] = df_cru['Longitude'].apply(get_macroregion)
        fact_datasets['CRU_CEDA'] = df_cru
    else:
        print("   ! ВНИМАНИЕ: Файл данных CRU CEDA не найден.")

    return fact_datasets


def generate_fact_only_excel_reports(fact_datasets):
    """Генерирует 4 Excel-отчета только с фактическими данными за полный исторический период."""
    print("3.2. Генерация Excel-отчетов с фактическими данными за весь период...")

    regions_config = [
        ('Вся зона степей', None),
        ('Европейская часть', 'Европейская часть'),
        ('Западная Сибирь', 'Западная Сибирь'),
        ('Восточная Сибирь', 'Восточная Сибирь'),
        ('Дальний Восток', 'Дальний Восток')
    ]

    for dataset_key, df_fact in fact_datasets.items():
        prefix = 'Meteo_RU' if dataset_key == 'METEO_RU' else 'CRU_CEDA'
        min_year = int(df_fact['Year'].min())
        max_year = int(df_fact['Year'].max())

        for var in ['DTR', 'Precip']:
            file_name = f"{prefix}_{var}_Fact_{min_year}-{max_year}.xlsx"
            output_path = ANALYTICS_DIR / file_name
            print(f"   Формирование отчета с фактом: {file_name}...")

            with pd.ExcelWriter(output_path, engine='openpyxl') as writer:
                for sheet_name, region_name in regions_config:
                    if region_name is None:
                        region_df = df_fact
                    else:
                        region_df = df_fact[df_fact['Region'] == region_name]

                    if region_df.empty:
                        empty_monthly = pd.DataFrame(columns=['Year', 'Month', 'Fact'])
                        empty_annual = pd.DataFrame(columns=['Year', 'Fact', 'N_Months'])
                        empty_monthly.to_excel(writer, sheet_name=sheet_name, startcol=0, index=False)
                        empty_annual.to_excel(writer, sheet_name=sheet_name, startcol=7, index=False)
                        continue

                    # Левая таблица: Месячные значения факта (среднее по станциям региона)
                    monthly_avg = region_df.groupby(['Year', 'Month'])[var].mean().reset_index()
                    monthly_avg = monthly_avg.rename(columns={var: 'Fact'})
                    monthly_avg = monthly_avg.sort_values(['Year', 'Month']).reset_index(drop=True)
                    monthly_table = monthly_avg[['Year', 'Month', 'Fact']]

                    # Правая таблица: Годовые значения факта
                    groups = monthly_avg.groupby('Year')['Fact']
                    count = groups.count().reindex(range(min_year, max_year + 1), fill_value=0)
                    expected_months = len(FORECAST_MONTHS)
                    annual = groups.mean() if var == 'DTR' else groups.sum(min_count=expected_months)
                    annual_table = pd.DataFrame({
                        'Fact': annual.where(count.eq(expected_months)),
                        'N_Months': count,
                    }).rename_axis('Year').reset_index()

                    # Записываем обе таблицы бок о бок (пропуская столбец G - index 6)
                    monthly_table.to_excel(writer, sheet_name=sheet_name, startcol=0, index=False)
                    annual_table.to_excel(writer, sheet_name=sheet_name, startcol=7, index=False)

    print("   -> Отчеты Excel с фактическими данными за весь период успешно сохранены.")



def calculate_and_save_metrics(raw_datasets, fact_datasets):
    """Рассчитывает метрики за общий период валидации и сохраняет их в metrics_summary.txt."""
    print(f"4. Расчет общих метрик точности моделей ({VALIDATION_PERIOD} гг.)...")
    output_path = ANALYTICS_DIR / "metrics_summary.txt"
    years = range(VALIDATION_START_YEAR, VALIDATION_END_YEAR + 1)
    climatologies = {
        dataset: station_month_climatology(prepare_monthly_forecast_data(monthly, dataset), years)
        for dataset, monthly in fact_datasets.items()
        if not monthly.empty
    }

    with open(output_path, 'w', encoding='utf-8') as f:
        f.write("========================================================================\n")
        f.write(f"МЕТРИКИ ПРЯМОГО ПРОГНОЗА НА 12 МЕСЯЦЕВ ({VALIDATION_PERIOD} гг., ежегодное переобучение)\n")
        f.write("========================================================================\n\n")

        for info in raw_datasets.values():
            df_raw = info['raw']
            var = info['var']
            dataset = info['dataset']
            climatology = climatologies.get(dataset)
            
            f.write(f"ДАТАСЕТ: {dataset} | ПОКАЗАТЕЛЬ: {var}\n")
            f.write("-" * 60 + "\n")
            if climatology is None:
                f.write("Для сравнения с климатологией нужны исходные месячные ряды.\n")
            
            rows = []
            for model in FORECAST_MODELS:
                predictions = df_raw.loc[df_raw['Model'] == model]
                metrics = calculate_climatology_comparison(predictions, climatology, var)
                rows.extend({'Модель': model, **row} for row in metrics)
            f.write(format_forecast_metrics(rows) + "\n\n")
            f.write("=" * 70 + "\n\n")

    print(f"   -> Метрики точности сохранены в: {output_path.name}")


def generate_map_atlas(dataset_info):
    """Генерирует атлас за последний год долгосрочной валидации."""
    print(f"5. Запуск генерации атласа карт за {VALIDATION_END_YEAR} год...")
    
    if not HAS_VISUALIZER:
        print("ОШИБКА: Визуализатор не найден. Пропуск карт.")
        return

    # Загружаем базовую гео-основу через BaseVisualizer (чтобы не читать файлы каждый раз)
    viz = BaseVisualizer(base_dir=PROJECT_DIR, results_dir=RESULTS_DIR, subfolder_name="analytics_reports")
    world_map = viz.world_map

    total_maps = 0

    for key, info in dataset_info.items():
        pivoted_df = info['pivoted']
        pivoted_df = pivoted_df.loc[np.isfinite(pivoted_df[['Latitude', 'Longitude']]).all(axis=1)]
        var = info['var']
        dataset = info['dataset']
        
        # Папки выхода для конкретного датасета и показателя
        dataset_folder = "meteo_ru" if dataset == 'METEO_RU' else "cru_ceda"
        png_dir = ANALYTICS_DIR / "maps_png" / dataset_folder / var
        tiff_dir = ANALYTICS_DIR / "maps_tiff" / dataset_folder / var

        # Настройки палитры
        cmap = 'plasma' if var == 'DTR' else 'YlGnBu'
        var_label = 'Суточная амплитуда температуры, °C' if var == 'DTR' else 'Количество осадков, мм'
        ru_var_title = RU_VARS[var]

        df_target_year = pivoted_df[pivoted_df['Year'] == VALIDATION_END_YEAR].copy()
        if df_target_year.empty:
            print(f"   ВНИМАНИЕ: Нет данных за {VALIDATION_END_YEAR} год для {dataset}_{var}!")
            continue

        # Определяем границы сетки по координатам всех станций этого датасета
        lat_min, lat_max = pivoted_df['Latitude'].min() - 0.5, pivoted_df['Latitude'].max() + 0.5
        lon_min, lon_max = pivoted_df['Longitude'].min() - 0.5, pivoted_df['Longitude'].max() + 0.5
        grid_lat, grid_lon = np.mgrid[lat_min:lat_max:MAP_GRID_STEP, lon_min:lon_max:MAP_GRID_STEP]

        # Генерируем 13 срезов: 12 месяцев + 1 годовой
        for month_idx in range(1, 14):
            is_annual = (month_idx == 13)
            
            # 1. Готовим данные среза для всех моделей и факта
            if is_annual:
                # Годовая агрегация для каждой станции
                df_slice = aggregate_annual_forecasts(df_target_year, ['WMO', 'Latitude', 'Longitude'], var)
                slice_name = "Год"
                title_time = f"Годовой срез {VALIDATION_END_YEAR} г."
            else:
                df_slice = df_target_year[df_target_year['Month'] == month_idx].copy()
                slice_name = RU_MONTHS[month_idx]
                title_time = f"{slice_name} {VALIDATION_END_YEAR} г."

            if df_slice.empty:
                continue

            points = df_slice[['Latitude', 'Longitude']].values

            # 2. Строим интерполированные гриды для Fact, RF, KNN, MLP
            maps = {}
            ru_dataset_name = RU_DATASETS[dataset]
            
            for m in FORECAST_VALUE_COLUMNS:
                vals = df_slice[m].to_numpy()
                if not np.isfinite(vals).any():
                    continue
                da = interpolate_map(points, vals, grid_lat, grid_lon)
                if var == 'Precip':
                    da = da.clip(min=0)
                filename = f'{ru_dataset_name}_{ru_var_title}_{RU_MODELS_FILE[m]}_{slice_name}_{VALIDATION_END_YEAR}'
                da = viz.clip_map(da, filename)
                if da is not None:
                    maps[m] = (filename, da, vals)

            if not maps:
                continue
            # Общая шкала учитывает заливку и значения в исходных точках.
            values = np.concatenate([np.r_[da.values.ravel(), vals] for _, da, vals in maps.values()])
            finite = values[np.isfinite(values)]
            vmin, vmax = 0, 1
            if finite.size:
                # Округлим для аккуратности
                vmin = np.floor(finite.min() * 10) / 10
                vmax = np.ceil(finite.max() * 10) / 10
                # Для осадков гарантируем vmin >= 0
                if var == 'Precip':
                    vmin = max(0.0, vmin)

            # 4. Отрисовка и сохранение карт
            for m, (file_name_base, da, vals) in maps.items():
                model_title = RU_MODELS_TITLE[m]
                
                png_path = png_dir / f"{file_name_base}.png"
                tiff_path = tiff_dir / f"{file_name_base}.tif"

                title = f"{ru_var_title} ({model_title})\n{ru_dataset_name} | {title_time}"
                
                # Отрисовка
                fig, ax = plt.subplots(figsize=(18, 6))
                
                if world_map is not None:
                    world_map.plot(ax=ax, color='#e0e0e0', edgecolor='white', linewidth=0.2, zorder=0)
                
                im = plot_interpolated_map(ax, da, points, vals, cmap=cmap, vmin=vmin, vmax=vmax)
                ax.grid(True, linestyle=':', alpha=0.4, color='black', linewidth=0.8, zorder=2)
                
                # Добавляем легенду
                cbar = plt.colorbar(im, ax=ax, fraction=0.015, pad=0.02, shrink=0.7)
                cbar.set_label(var_label, fontsize=10)
                
                ax.set_title(title, fontsize=12, fontweight='bold', pad=10)
                ax.set_ylabel("Широта", fontsize=10)
                
                # Зумируем маску
                ax.set_xlim(lon_min - 10, lon_max + 15) 
                ax.set_ylim(lat_min - 8, lat_max + 12)
                
                plt.tight_layout()
                
                # Сохраняем форматы
                plt.savefig(png_path, dpi=200)
                plt.savefig(tiff_path, dpi=300, format='tiff', pil_kwargs={'compression': 'tiff_lzw'})
                plt.close()
                
                total_maps += 1
                
            print(f"   [{dataset} {var}] {title_time}. Сохранено карт: {len(maps)}")

    print(f"\n   -> Атлас сгенерирован! Создано {total_maps} карт в форматах PNG и TIFF.")


def cleanup_empty_folders():
    """Удаляет пустые папки maps и plots, созданные конструктором BaseVisualizer."""
    print("\n6. Очистка пустых папок...")
    for folder_name in ["plots", "maps"]:
        folder_path = ANALYTICS_DIR / folder_name
        if folder_path.exists() and folder_path.is_dir():
            if not any(folder_path.iterdir()):
                try:
                    folder_path.rmdir()
                    print(f"   -> Удалена пустая папка: {folder_path.name}")
                except Exception as e:
                    print(f"   ! Не удалось удалить папку {folder_path.name}: {e}")


def main():
    print(">>> СТАРТ РАБОТЫ АНАЛИТИЧЕСКОГО МОДУЛЯ <<<\n")
    
    # Шаг 1. Папки
    create_directories()
    
    # Шаг 2. Загрузка исходных CSV
    print("\n2. Загрузка исходных данных долгосрочной валидации...")
    csv_configs = {
        'meteo_dtr': {
            'path': RESULTS_DIR / 'validation_results' / 'csv_data' / validation_filename('METEO_RU', 'DTR'),
            'dataset': 'METEO_RU',
            'var': 'DTR'
        },
        'meteo_precip': {
            'path': RESULTS_DIR / 'validation_results' / 'csv_data' / validation_filename('METEO_RU', 'Precip'),
            'dataset': 'METEO_RU',
            'var': 'Precip'
        },
        'cru_dtr': {
            'path': RESULTS_DIR / 'validation_results' / 'csv_data' / validation_filename('CRU_CEDA', 'DTR'),
            'dataset': 'CRU_CEDA',
            'var': 'DTR'
        },
        'cru_precip': {
            'path': RESULTS_DIR / 'validation_results' / 'csv_data' / validation_filename('CRU_CEDA', 'Precip'),
            'dataset': 'CRU_CEDA',
            'var': 'Precip'
        }
    }
    
    dataset_info = {}
    for key, config in csv_configs.items():
        if not config['path'].exists():
            print(f"ОШИБКА: Файл валидации не найден: {config['path']}")
            print("Сначала запустите основной расчет kursach011.py!")
            return
        
        print(f"   Загрузка {config['path'].name}...")
        try:
            df_raw = read_validation_csv(config['path'], config['dataset'])
        except ValueError as error:
            print(error)
            return
        df_pivoted = pivot_forecasts(df_raw)
        
        dataset_info[key] = {
            'raw': df_raw,
            'pivoted': df_pivoted,
            'dataset': config['dataset'],
            'var': config['var']
        }
    
    # Шаг 3. Отчеты Excel за общий период валидации
    print()
    generate_excel_reports(dataset_info)
    
    # Шаг 3.1 и 3.2. Отчеты Excel только с фактическими данными за весь доступный период
    print()
    fact_datasets = load_full_fact_data()
    if fact_datasets:
        generate_fact_only_excel_reports(fact_datasets)
    
    # Шаг 4. Метрики в txt
    print()
    calculate_and_save_metrics(dataset_info, fact_datasets)
    
    # Шаг 5. Атлас карт
    print()
    generate_map_atlas(dataset_info)
    
    # Шаг 6. Очистка пустых папок
    cleanup_empty_folders()
    
    print(f"\n>>> РАБОТА ЗАВЕРШЕНА! Отчеты и атлас находятся в: {ANALYTICS_DIR} <<<")


if __name__ == "__main__":
    main()
