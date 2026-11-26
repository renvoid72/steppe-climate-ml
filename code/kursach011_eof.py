# -*- coding: utf-8 -*-
"""
Модуль математического разложения метеорологических полей методом ЕОФ (PCA).
Задачи:
1. Расчет ЕОФ (PCA) по четырём сезонам за общий период валидации.
2. Сохранение графиков и карт пространственных мод в сезонные подпапки.
3. Группировка объясненной дисперсии по сезонам в variance_summary.txt.
4. Расчет и экспорт исторических временных серий ЕОФ (PC-1, PC-2, PC-3)
   по 5 регионам на отдельные вкладки в Excel файлы:
   - CRU (с 1920 г.)
   - Метеостанции РФ (с 1960 г.)
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
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler

from kursach011_modules.forecast_engine import read_validation_csv
from kursach011_modules.meteo_data import load_monthly_meteo
from kursach011_modules.project_settings import (
    ANALYSIS_YEAR, CRU_VERSION, VALIDATION_END_YEAR, VALIDATION_PERIOD, VALIDATION_START_YEAR,
    cru_csv_path, results_dir, validation_filename,
)

warnings.filterwarnings('ignore')

# --- КОНСТАНТЫ ПУТЕЙ ---
PROJECT_DIR = Path(__file__).resolve().parent.parent
RESULTS_DIR = results_dir(PROJECT_DIR)
EOF_DIR = RESULTS_DIR / "eof_results"

# Добавляем путь к коду, чтобы импортировать BaseVisualizer
sys.path.append(str(PROJECT_DIR / "code"))
try:
    from kursach011_modules.visualizer import BaseVisualizer, MAP_GRID_STEP, interpolate_map, plot_interpolated_map
    HAS_VISUALIZER = True
except ImportError as e:
    HAS_VISUALIZER = False
    print(f"ВНИМАНИЕ: Не удалось импортировать BaseVisualizer: {e}")


def create_directories():
    """Создает структуру папок для ЕОФ-анализа (включая сезонные подпапки)."""
    print("1. Создание структуры папок...")
    folders = [EOF_DIR, EOF_DIR / "plots"]
    seasons = ['Winter', 'Spring', 'Summer', 'Autumn']
    for s in seasons:
        folders.append(EOF_DIR / "plots" / s)
        for ds in ['meteo_ru', 'cru_ceda']:
            for param in ['DTR', 'Precip']:
                folders.append(EOF_DIR / "maps_png" / s / ds / param)
                folders.append(EOF_DIR / "maps_tiff" / s / ds / param)
                
    for folder in folders:
        folder.mkdir(parents=True, exist_ok=True)
    print("   -> Папки успешно созданы.")


def cleanup_empty_folders():
    """Удаляет пустые папки, созданные конструктором BaseVisualizer."""
    print("\nОчистка пустых папок...")
    for folder_name in ["plots", "maps"]:
        folder_path = EOF_DIR / folder_name
        if folder_path.exists() and folder_path.is_dir():
            if not any(folder_path.iterdir()):
                try:
                    folder_path.rmdir()
                    print(f"   -> Удалена пустая папка: {folder_path.name}")
                except Exception as e:
                    print(f"   ! Не удалось удалить папку {folder_path.name}: {e}")


# --- ВСПОМОГАТЕЛЬНЫЕ МЕТОДЫ ДЛЯ ИСТОРИЧЕСКОГО АНАЛИЗА ---

def load_meteo_ru_monthly():
    """Загружает и агрегирует данные метеостанций РФ из all_steppe_data.txt."""
    meteo_file = PROJECT_DIR / "meteo_ru" / "all_steppe_data.txt"
    coords_file = PROJECT_DIR / "meteo_ru" / "stations_coords.txt"
    if not meteo_file.exists() or not coords_file.exists():
        raise FileNotFoundError("Файлы данных METEO_RU не найдены в папке 'meteo_ru'.")
    
    df_monthly = load_monthly_meteo(meteo_file, coords_file)
    
    df_monthly['Date'] = pd.to_datetime(df_monthly[['Year', 'Month']].assign(day=1))
    return df_monthly


def load_cru_ceda_monthly():
    """Загружает готовые сеточные данные CRU CEDA."""
    cru_file = cru_csv_path(PROJECT_DIR)
    if not cru_file.exists():
        raise FileNotFoundError(f"Файл CRU TS {CRU_VERSION} не найден: {cru_file}. Запустите code/prepare_cru.py.")
    df = pd.read_csv(cru_file)
    df['Date'] = pd.to_datetime(df[['Year', 'Month']].assign(day=1))
    return df


EOF_REGIONS = (
    ('Вся зона степей', None, None),
    ('Европейская часть', None, 60),
    ('Западная Сибирь', 60, 90),
    ('Восточная Сибирь', 90, 120),
    ('Дальний Восток', 120, None),
)


class EOFDataUnavailable(ValueError):
    """Данных среза недостаточно для расчёта ЕОФ."""


def regional_rows(rows, lon_min, lon_max):
    if lon_min is not None:
        rows = rows.loc[rows['Longitude'] >= lon_min]
    if lon_max is not None:
        rows = rows.loc[rows['Longitude'] < lon_max]
    return rows


def eof_calendar_matrix(rows, variable, start_year, end_year):
    """Формирует календарную матрицу «месяц — станция» по доступным фактам."""
    calendar = pd.date_range(f'{start_year}-01-01', f'{end_year}-12-01', freq='MS', name='Date')
    dates = pd.to_datetime(rows['Date']).dt.to_period('M').dt.to_timestamp()
    rows = rows.assign(Date=dates).loc[dates.isin(calendar), ['Date', 'WMO', variable]]
    values = rows.replace({variable: {np.inf: np.nan, -np.inf: np.nan}})
    # В валидации один и тот же факт может встречаться в строках нескольких моделей.
    return values.groupby(['Date', 'WMO'])[variable].first().unstack('WMO').reindex(calendar)


def fit_eof(matrix, n_components=None):
    """Отбирает станции, вычисляет месячные аномалии и оценивает компоненты."""
    observed = matrix.dropna(how='all')
    if observed.empty:
        raise EOFDataUnavailable('Нет наблюдений для анализа.')
    missing_months = matrix.index.month.unique().difference(observed.index.month.unique())
    if len(missing_months):
        months = ', '.join(map(str, missing_months))
        raise EOFDataUnavailable(f'Нет наблюдений для календарных месяцев: {months}.')

    # Считаем долю пропусков от календаря среза, включая месяцы без единой записи.
    available = matrix.loc[:, matrix.notna().mean().ge(0.90)].dropna(how='all')
    normals = available.groupby(available.index.month).transform('mean')
    anomalies = available.fillna(normals).sub(normals).dropna(axis=1)
    # Не включаем в PCA даты, для которых у отобранных станций нет ни одного факта.
    anomalies = anomalies.loc[available[anomalies.columns].notna().any(axis=1)]
    n_months, n_stations = anomalies.shape
    if n_components is not None and min(n_months, n_stations) < n_components:
        raise EOFDataUnavailable(
            f'Недостаточно данных для {n_components} компонент (месяцев: {n_months}, станций: {n_stations}).'
        )
    if n_months < 2 or not anomalies.var(ddof=0).gt(0).any():
        raise EOFDataUnavailable('Недостаточно изменчивых наблюдений для оценки дисперсии.')

    pca = PCA(n_components=n_components)
    scores = pca.fit_transform(StandardScaler().fit_transform(anomalies))
    pc_names = [f'PC-{index}' for index in range(1, pca.n_components_ + 1)]
    pcs = pd.DataFrame(scores, index=anomalies.index, columns=pc_names)
    return pcs, pca, anomalies.columns


def calculate_regional_pcs(df_monthly, param, lon_min=None, lon_max=None, start_year=1920, end_year=ANALYSIS_YEAR):
    """Выполняет разложение PCA для конкретного региона, параметра и периода."""
    rows = regional_rows(df_monthly, lon_min, lon_max)
    matrix = eof_calendar_matrix(rows, param, start_year, end_year)
    try:
        pcs, _, _ = fit_eof(matrix, n_components=3)
    except EOFDataUnavailable as error:
        print(f'        {param}. Пропуск: {error}')
        return None
    return pcs


def export_historical_eof_series():
    """Экспортирует исторические ряды компонент по регионам в Excel."""
    print("\n4. Расчет и экспорт исторических временных серий ЕОФ (PC)...")
    datasets = (
        ('CRU_CEDA', load_cru_ceda_monthly, 1920, VALIDATION_END_YEAR),
        ('Meteo_RU', load_meteo_ru_monthly, 1960, ANALYSIS_YEAR),
    )
    for dataset, load_monthly, start_year, end_year in datasets:
        print(f'   Обработка {dataset} ({start_year}-{end_year} гг.)...')
        try:
            monthly = load_monthly()
        except (OSError, ValueError) as error:
            print(f'   ! Ошибка при чтении {dataset}: {error}')
            continue

        path = EOF_DIR / f'{dataset}_EOF_TimeSeries_{start_year}-{end_year}.xlsx'
        with pd.ExcelWriter(path, engine='openpyxl') as writer:
            for region, lon_min, lon_max in EOF_REGIONS:
                print(f'      Регион: {region}...')
                all_columns = []
                tables = []
                for variable in ('DTR', 'Precip'):
                    pcs = calculate_regional_pcs(monthly, variable, lon_min, lon_max, start_year, end_year)
                    all_columns.extend(f'PC-{i}_{variable}' for i in range(1, 4))
                    if pcs is not None:
                        tables.append(pcs.add_suffix(f'_{variable}'))

                if tables:
                    merged = pd.concat(tables, axis=1).sort_index().reindex(columns=all_columns).reset_index()
                    merged['Date'] = merged['Date'].dt.strftime('%Y-%m-%d')
                else:
                    merged = pd.DataFrame({'Статус': ['Недостаточно данных для трёх компонент']})
                merged.to_excel(writer, sheet_name=region, index=False)
        print(f'   -> Файл успешно сохранен: {path.name}')


def export_all_components_variance():
    """Рассчитывает долю объясненной дисперсии для ВСЕХ доступных компонент по 5 регионам."""
    print("\n5. Расчет и экспорт полной таблицы объясненной дисперсии (все компоненты)...")
    
    datasets_config = [
        {
            'name': 'METEO_RU',
            'file_name': validation_filename('METEO_RU', 'DTR'),
            'out_name': 'Meteo_RU_DTR_EOF_Variance_All.xlsx'
        },
        {
            'name': 'METEO_RU',
            'file_name': validation_filename('METEO_RU', 'Precip'),
            'out_name': 'Meteo_RU_Precip_EOF_Variance_All.xlsx'
        },
        {
            'name': 'CRU_CEDA',
            'file_name': validation_filename('CRU_CEDA', 'DTR'),
            'out_name': 'CRU_CEDA_DTR_EOF_Variance_All.xlsx'
        },
        {
            'name': 'CRU_CEDA',
            'file_name': validation_filename('CRU_CEDA', 'Precip'),
            'out_name': 'CRU_CEDA_Precip_EOF_Variance_All.xlsx'
        }
    ]
    
    for item in datasets_config:
        csv_path = RESULTS_DIR / 'validation_results' / 'csv_data' / item['file_name']
        if not csv_path.exists():
            print(f"Ошибка: Файл не найден: {csv_path.name}. Пропуск.")
            continue
            
        out_path = EOF_DIR / item['out_name']
        print(f"   Формирование файла: {item['out_name']}...")
        
        try:
            df = read_validation_csv(csv_path, item['name'])
        except ValueError as error:
            print(error)
            continue
        with pd.ExcelWriter(out_path, engine='openpyxl') as writer:
            for region, lon_min, lon_max in EOF_REGIONS:
                matrix = eof_calendar_matrix(
                    regional_rows(df, lon_min, lon_max), 'Fact', VALIDATION_START_YEAR, VALIDATION_END_YEAR,
                )
                try:
                    _, pca, _ = fit_eof(matrix)
                except EOFDataUnavailable as error:
                    print(f'      {region}. Пропуск: {error}')
                    pd.DataFrame({'Статус': [str(error)]}).to_excel(writer, sheet_name=region, index=False)
                    continue

                ev = pca.explained_variance_ratio_ * 100
                cumulative_ev = np.cumsum(ev)
                cumulative_ev[-1] = 100
                pd.DataFrame({
                    'Компонента': [f'ЕОФ-{i+1}' for i in range(len(ev))],
                    'Объясненная дисперсия (%)': np.round(ev, 4),
                    'Накопленная дисперсия (%)': np.round(cumulative_ev, 4),
                }).to_excel(writer, sheet_name=region, index=False)

        print(f"   -> Успешно сохранен: {item['out_name']}")


# --- ОСНОВНАЯ ЛОГИКА ---

def main():
    print(">>> СТАРТ ОБНОВЛЕННОГО МОДУЛЯ ЕОФ (PCA) С СЕЗОНАМИ И ЭКСПОРТОМ <<<\n")
    
    # Шаг 1. Создание папок
    create_directories()
    
    # Инициализация визуализатора для загрузки гео-основы
    viz = BaseVisualizer(PROJECT_DIR, RESULTS_DIR, 'eof_results') if HAS_VISUALIZER else None

    # Конфигурация для расчётов ЕОФ по сезонам
    datasets_config = [
        {
            'name': 'METEO_RU',
            'param': 'DTR',
            'file_name': validation_filename('METEO_RU', 'DTR'),
            'title_ds': 'Метеостанции РФ',
            'title_param': 'DTR',
            'dir_name': 'meteo_ru'
        },
        {
            'name': 'METEO_RU',
            'param': 'Precip',
            'file_name': validation_filename('METEO_RU', 'Precip'),
            'title_ds': 'Метеостанции РФ',
            'title_param': 'Осадки',
            'dir_name': 'meteo_ru'
        },
        {
            'name': 'CRU_CEDA',
            'param': 'DTR',
            'file_name': validation_filename('CRU_CEDA', 'DTR'),
            'title_ds': 'CRU CEDA',
            'title_param': 'DTR',
            'dir_name': 'cru_ceda'
        },
        {
            'name': 'CRU_CEDA',
            'param': 'Precip',
            'file_name': validation_filename('CRU_CEDA', 'Precip'),
            'title_ds': 'CRU CEDA',
            'title_param': 'Осадки',
            'dir_name': 'cru_ceda'
        }
    ]

    variance_results = []

    print(f"\n2. Анализ ЕОФ по сезонам ({VALIDATION_PERIOD} гг.)...")
    for item in datasets_config:
        csv_path = RESULTS_DIR / 'validation_results' / 'csv_data' / item['file_name']
        if not csv_path.exists():
            print(f"Ошибка: Файл не найден: {csv_path.name}. Пропуск.")
            continue
            
        print(f"\n   Обработка {item['title_ds']} | {item['title_param']}...")
        
        # 1. Загрузка данных
        try:
            df = read_validation_csv(csv_path, item['name'])
        except ValueError as error:
            print(error)
            continue
        
        coords_df = df[['WMO', 'Latitude', 'Longitude']].drop_duplicates(subset=['WMO'])
        wmo_to_coords = coords_df.set_index('WMO')[['Latitude', 'Longitude']].to_dict('index')
        matrix = eof_calendar_matrix(df, 'Fact', VALIDATION_START_YEAR, VALIDATION_END_YEAR)

        # Определение сезонов
        seasons_def = {
            'Winter': {'months': [12, 1, 2], 'name_ru': 'Зима'},
            'Spring': {'months': [3, 4, 5], 'name_ru': 'Весна'},
            'Summer': {'months': [6, 7, 8], 'name_ru': 'Лето'},
            'Autumn': {'months': [9, 10, 11], 'name_ru': 'Осень'}
        }
        
        # Расчет PCA отдельно по сезонам
        for s_name, s_info in seasons_def.items():
            print(f"      Разложение для сезона: {s_info['name_ru']}...")
            season_matrix = matrix.loc[matrix.index.month.isin(s_info['months'])]
            try:
                pcs, pca, stations = fit_eof(season_matrix, n_components=3)
            except EOFDataUnavailable as error:
                print(f"        Пропуск ({s_info['name_ru']}): {error}")
                continue
            loadings = pca.components_
            explained_var = pca.explained_variance_ratio_ * 100

            print(f"        Объясненная дисперсия: ЕОФ-1: {explained_var[0]:.2f}%, ЕОФ-2: {explained_var[1]:.2f}%, ЕОФ-3: {explained_var[2]:.2f}%")
            
            # Сохранение дисперсии для отчета
            variance_results.append({
                'season': s_info['name_ru'],
                'ds': item['title_ds'],
                'param': item['title_param'],
                'ev1': explained_var[0],
                'ev2': explained_var[1],
                'ev3': explained_var[2]
            })
            
            # Отрисовка графиков временного хода (Principal Components)
            plt.figure(figsize=(14, 5))
            plt.plot(pcs.index, pcs['PC-1'], label=f'PC-1 ({explained_var[0]:.2f}%)', color='#1f77b4', linewidth=1.5, marker='o', markersize=3)
            plt.plot(pcs.index, pcs['PC-2'], label=f'PC-2 ({explained_var[1]:.2f}%)', color='#2ca02c', linewidth=1.5, marker='s', markersize=3)
            plt.plot(pcs.index, pcs['PC-3'], label=f'PC-3 ({explained_var[2]:.2f}%)', color='#d62728', linewidth=1.5, marker='^', markersize=3)
            plt.axhline(0, color='black', linestyle='--', linewidth=0.8, alpha=0.7)
            
            plt.title(f"Временной ход глав. компонент (PCs) | Сезон: {s_info['name_ru']}\n{item['title_ds']} | {item['title_param']} ({VALIDATION_PERIOD} гг.)", fontsize=12, fontweight='bold')
            plt.xlabel("Годы", fontsize=10)
            plt.ylabel("Амплитуда", fontsize=10)
            plt.legend(loc='upper right', fontsize=10)
            plt.grid(True, linestyle=':', alpha=0.5)
            plt.tight_layout()
            
            plot_name = f"PC_timeseries_{item['name']}_{item['param']}.png"
            plt.savefig(EOF_DIR / "plots" / s_name / plot_name, dpi=200)
            plt.close()
            
            # Генерация карт пространственных мод
            if viz is None:
                continue
            png_dir = EOF_DIR / "maps_png" / s_name / item['dir_name'] / item['param']
            tiff_dir = EOF_DIR / "maps_tiff" / s_name / item['dir_name'] / item['param']
            
            lats = np.array([wmo_to_coords[wmo]['Latitude'] for wmo in stations])
            lons = np.array([wmo_to_coords[wmo]['Longitude'] for wmo in stations])
            
            located = np.isfinite(lats) & np.isfinite(lons)
            if not located.any():
                continue
            lat_min, lat_max = lats[located].min() - 0.5, lats[located].max() + 0.5
            lon_min, lon_max = lons[located].min() - 0.5, lons[located].max() + 0.5
            grid_lat, grid_lon = np.mgrid[lat_min:lat_max:MAP_GRID_STEP, lon_min:lon_max:MAP_GRID_STEP]
            points = np.column_stack((lats[located], lons[located]))
            clean_ds = item['title_ds'].replace(' ', '_')
            saved_maps = 0
            
            for m in range(3):
                eof_num = m + 1
                vals = loadings[m][located]
                file_name_base = f"Пространственная_мода_ЕОФ-{eof_num}_{clean_ds}_{item['title_param']}_{s_name}"
                da = viz.clip_map(interpolate_map(points, vals, grid_lat, grid_lon), file_name_base)
                if da is None:
                    continue

                max_abs = max(np.nanmax(np.abs(da.values), initial=0), np.nanmax(np.abs(vals), initial=0)) or 1.0
                vmin, vmax = -max_abs, max_abs
                
                # Отрисовка карты
                fig, ax = plt.subplots(figsize=(18, 6))
                
                if viz.world_map is not None:
                    viz.world_map.plot(ax=ax, color='#e0e0e0', edgecolor='white', linewidth=0.2, zorder=0)
                
                im = plot_interpolated_map(ax, da, points, vals, cmap='coolwarm', vmin=vmin, vmax=vmax)
                ax.grid(True, linestyle=':', alpha=0.4, color='black', linewidth=0.8, zorder=2)
                
                cbar = plt.colorbar(im, ax=ax, fraction=0.015, pad=0.02, shrink=0.7)
                cbar.set_label("Нагрузка ЕОФ (безразм.)", fontsize=10)
                
                title = f"Пространственная мода ЕОФ-{eof_num} | Сезон: {s_info['name_ru']} | {item['title_ds']} | {item['title_param']}\nДоля объясненной дисперсии: {explained_var[m]:.2f}%"
                ax.set_title(title, fontsize=12, fontweight='bold', pad=10)
                ax.set_ylabel("Широта", fontsize=10)
                
                ax.set_xlim(lon_min - 10, lon_max + 15)
                ax.set_ylim(lat_min - 8, lat_max + 12)
                
                plt.tight_layout()
                
                png_path = png_dir / f"{file_name_base}.png"
                tiff_path = tiff_dir / f"{file_name_base}.tif"
                
                plt.savefig(png_path, dpi=200)
                plt.savefig(tiff_path, dpi=300, format='tiff', pil_kwargs={'compression': 'tiff_lzw'})
                plt.close()
                saved_maps += 1
                
            print(f"        {s_info['name_ru']}. Сохранено карт: {saved_maps}")

    # Шаг 3. Сохранение отчета по дисперсии
    print("\n3. Сохранение отчета по дисперсии...")
    variance_summary_path = EOF_DIR / "variance_summary.txt"
    with open(variance_summary_path, 'w', encoding='utf-8') as f:
        f.write("========================================================================\n")
        f.write(f"ДОЛИ ОБЪЯСНЕННОЙ ДИСПЕРСИИ ДЛЯ ПЕРВЫХ ТРЕХ МОД ЕОФ ПО СЕЗОНАМ ({VALIDATION_PERIOD} гг.)\n")
        f.write("========================================================================\n\n")
        
        # Группируем по сезонам
        seasons_in_results = ['Зима', 'Весна', 'Лето', 'Осень']
        for s in seasons_in_results:
            f.write(f"========================================================================\n")
            f.write(f"СЕЗОН: {s}\n")
            f.write(f"========================================================================\n\n")
            
            s_res = [res for res in variance_results if res['season'] == s]
            for res in s_res:
                f.write(f"  ДАТАСЕТ: {res['ds']} | ПОКАЗАТЕЛЬ: {res['param']}\n")
                f.write(f"  " + "-" * 50 + "\n")
                f.write(f"    ЕОФ-1: {res['ev1']:8.2f}%\n")
                f.write(f"    ЕОФ-2: {res['ev2']:8.2f}%\n")
                f.write(f"    ЕОФ-3: {res['ev3']:8.2f}%\n")
                f.write(f"    Суммарно (3 моды): {res['ev1']+res['ev2']+res['ev3']:8.2f}%\n\n")
            f.write("\n")
            
    print(f"   -> Отчет успешно сохранен в: {variance_summary_path.name}")

    # Шаг 4. Расчет и экспорт исторических временных серий ЕОФ
    export_historical_eof_series()
    
    # Шаг 5. Расчет и экспорт полной таблицы объясненной дисперсии (все компоненты)
    export_all_components_variance()
    
    # Шаг 6. Очистка пустых папок
    cleanup_empty_folders()
    
    print(f"\n>>> РАБОТА ЗАВЕРШЕНА! Все результаты ЕОФ-анализа находятся в: {EOF_DIR} <<<")


if __name__ == '__main__':
    main()
