# -*- coding: utf-8 -*-
"""
Модуль математического разложения метеорологических полей методом ЕОФ (PCA).
Задачи:
1. Расчет ЕОФ (PCA) по 4 сезонам (Зима, Весна, Лето, Осень) за 1990-2020 гг.
2. Сохранение графиков и карт пространственных мод в сезонные подпапки.
3. Группировка объясненной дисперсии по сезонам в variance_summary.txt.
4. Расчет и экспорт исторических временных серий ЕОФ (PC-1, PC-2, PC-3)
   по 5 регионам на отдельные вкладки в Excel файлы:
   - CRU CEDA (1920-2020 гг.)
   - Метеостанции РФ (1960-2024 гг.)
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
from scipy.interpolate import griddata
import xarray as xr
from shapely.geometry import mapping
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler

from kursach011_modules.meteo_data import load_monthly_meteo

warnings.filterwarnings('ignore')

# --- КОНСТАНТЫ ПУТЕЙ ---
PROJECT_DIR = Path(__file__).resolve().parent.parent
RESULTS_DIR = PROJECT_DIR / "output011_modules"
EOF_DIR = RESULTS_DIR / "eof_results"

# Добавляем путь к коду, чтобы импортировать BaseVisualizer
sys.path.append(str(PROJECT_DIR / "code"))
try:
    from kursach011_modules.visualizer import BaseVisualizer
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
    cru_file = PROJECT_DIR / "cru ceda" / "cru_aligned_data.csv"
    if not cru_file.exists():
        raise FileNotFoundError("Файл cru_aligned_data.csv не найден в папке 'cru ceda'.")
    df = pd.read_csv(cru_file)
    df['Date'] = pd.to_datetime(df[['Year', 'Month']].assign(day=1))
    return df


def calculate_regional_pcs(df_monthly, param, lon_min=None, lon_max=None, start_year=1920, end_year=2024):
    """Выполняет разложение PCA для конкретного региона, параметра и периода."""
    # Фильтрация по годам
    df_sub = df_monthly[(df_monthly['Year'] >= start_year) & (df_monthly['Year'] <= end_year)].copy()
    
    # Фильтрация по географической долготе (региону)
    if lon_min is not None:
        df_sub = df_sub[df_sub['Longitude'] >= lon_min]
    if lon_max is not None:
        df_sub = df_sub[df_sub['Longitude'] < lon_max]
        
    if df_sub.empty:
        return None
        
    # Сводная таблица (строки - даты, столбцы - станции WMO)
    pivot_df = df_sub.pivot(index='Date', columns='WMO', values=param)
    pivot_df.index = pd.to_datetime(pivot_df.index)
    pivot_df = pivot_df.sort_index()
    
    # Мягкий фильтр пропусков: не более 10% пропущенных месяцев
    max_missing = int(0.10 * len(pivot_df))
    missing_counts = pivot_df.isnull().sum()
    keep_stations = missing_counts[missing_counts <= max_missing].index
    pivot_df = pivot_df[keep_stations].copy()
    
    if pivot_df.shape[1] < 3:
        # Недостаточно станций для разложения на 3 моды ЕОФ
        return None
        
    # Заполнение редких пропусков климатологической нормой
    climatology = pivot_df.groupby(pivot_df.index.month).mean()
    months_series = pd.Series(pivot_df.index.month, index=pivot_df.index)
    for col in pivot_df.columns:
        if pivot_df[col].isnull().any():
            fill_values = months_series.map(climatology[col])
            pivot_df[col] = pivot_df[col].fillna(fill_values)
            
    # Расчет климатических аномалий (центрирование)
    climatology_filled = pivot_df.groupby(pivot_df.index.month).mean()
    anomalies = pivot_df.copy()
    for month in range(1, 13):
        anomalies.loc[anomalies.index.month == month] -= climatology_filled.loc[month]
        
    # Стандартизация дисперсии
    scaler = StandardScaler()
    scaled_data = scaler.fit_transform(anomalies)
    
    # Разложение PCA
    pca = PCA(n_components=3)
    pcs = pca.fit_transform(scaled_data)
    
    res_df = pd.DataFrame(pcs, columns=['PC-1', 'PC-2', 'PC-3'], index=pivot_df.index)
    return res_df


def export_historical_eof_series():
    """Реализует второй этап: расчет и экспорт PC серий в Excel файлы."""
    print("\n4. Расчет и экспорт исторических временных серий ЕОФ (PC)...")
    
    regions_config = [
        {'name': 'Вся зона степей', 'lon_min': None, 'lon_max': None},
        {'name': 'Европейская часть', 'lon_min': None, 'lon_max': 60},
        {'name': 'Западная Сибирь', 'lon_min': 60, 'lon_max': 90},
        {'name': 'Восточная Сибирь', 'lon_min': 90, 'lon_max': 120},
        {'name': 'Дальний Восток', 'lon_min': 120, 'lon_max': None}
    ]

    # 1. CRU CEDA (1920-2020 гг.)
    print("   Обработка CRU CEDA (1920-2020 гг.)...")
    try:
        df_cru = load_cru_ceda_monthly()
        cru_out_path = EOF_DIR / "CRU_CEDA_EOF_TimeSeries_1920-2020.xlsx"
        
        with pd.ExcelWriter(cru_out_path, engine='openpyxl') as writer:
            for reg in regions_config:
                sheet_name = reg['name']
                print(f"      Регион: {sheet_name}...")
                
                dtr_pcs = calculate_regional_pcs(df_cru, 'DTR', reg['lon_min'], reg['lon_max'], 1920, 2020)
                precip_pcs = calculate_regional_pcs(df_cru, 'Precip', reg['lon_min'], reg['lon_max'], 1920, 2020)
                
                if dtr_pcs is not None and precip_pcs is not None:
                    # Объединяем PC для температуры и осадков на одной вкладке
                    merged = pd.DataFrame(index=dtr_pcs.index)
                    merged['PC-1_DTR'] = dtr_pcs['PC-1']
                    merged['PC-2_DTR'] = dtr_pcs['PC-2']
                    merged['PC-3_DTR'] = dtr_pcs['PC-3']
                    merged['PC-1_Precip'] = precip_pcs['PC-1']
                    merged['PC-2_Precip'] = precip_pcs['PC-2']
                    merged['PC-3_Precip'] = precip_pcs['PC-3']
                    
                    merged = merged.reset_index()
                    merged['Date'] = merged['Date'].dt.strftime('%Y-%m-%d')
                    merged.to_excel(writer, sheet_name=sheet_name, index=False)
                else:
                    # Пометка, если станций нет
                    empty_df = pd.DataFrame({"Статус": ["Нет станций для анализа"]})
                    empty_df.to_excel(writer, sheet_name=sheet_name, index=False)
        print(f"   -> Файл успешно сохранен: {cru_out_path.name}")
    except Exception as e:
        print(f"   ! Ошибка при экспорте CRU CEDA: {e}")

    # 2. МЕТЕО СТАНЦИИ РФ (1960-2024 гг.)
    print("   Обработка Метеостанций РФ (1960-2024 гг.)...")
    try:
        df_meteo = load_meteo_ru_monthly()
        meteo_out_path = EOF_DIR / "Meteo_RU_EOF_TimeSeries_1960-2024.xlsx"
        
        with pd.ExcelWriter(meteo_out_path, engine='openpyxl') as writer:
            for reg in regions_config:
                sheet_name = reg['name']
                print(f"      Регион: {sheet_name}...")
                
                dtr_pcs = calculate_regional_pcs(df_meteo, 'DTR', reg['lon_min'], reg['lon_max'], 1960, 2024)
                precip_pcs = calculate_regional_pcs(df_meteo, 'Precip', reg['lon_min'], reg['lon_max'], 1960, 2024)
                
                if dtr_pcs is not None and precip_pcs is not None:
                    # Объединяем PC для температуры и осадков на одной вкладке
                    merged = pd.DataFrame(index=dtr_pcs.index)
                    merged['PC-1_DTR'] = dtr_pcs['PC-1']
                    merged['PC-2_DTR'] = dtr_pcs['PC-2']
                    merged['PC-3_DTR'] = dtr_pcs['PC-3']
                    merged['PC-1_Precip'] = precip_pcs['PC-1']
                    merged['PC-2_Precip'] = precip_pcs['PC-2']
                    merged['PC-3_Precip'] = precip_pcs['PC-3']
                    
                    merged = merged.reset_index()
                    merged['Date'] = merged['Date'].dt.strftime('%Y-%m-%d')
                    merged.to_excel(writer, sheet_name=sheet_name, index=False)
                else:
                    # Пометка, если станций нет
                    empty_df = pd.DataFrame({"Статус": ["Нет станций для анализа"]})
                    empty_df.to_excel(writer, sheet_name=sheet_name, index=False)
        print(f"   -> Файл успешно сохранен: {meteo_out_path.name}")
    except Exception as e:
        print(f"   ! Ошибка при экспорте METEO_RU: {e}")


def export_all_components_variance():
    """Рассчитывает долю объясненной дисперсии для ВСЕХ доступных компонент по 5 регионам."""
    print("\n5. Расчет и экспорт полной таблицы объясненной дисперсии (все компоненты)...")
    
    datasets_config = [
        {
            'file_name': 'Validation_METEO_RU_DTR_1990-2020.csv',
            'out_name': 'Meteo_RU_DTR_EOF_Variance_All.xlsx'
        },
        {
            'file_name': 'Validation_METEO_RU_Precip_1990-2020.csv',
            'out_name': 'Meteo_RU_Precip_EOF_Variance_All.xlsx'
        },
        {
            'file_name': 'Validation_CRU_CEDA_DTR_1990-2020.csv',
            'out_name': 'CRU_CEDA_DTR_EOF_Variance_All.xlsx'
        },
        {
            'file_name': 'Validation_CRU_CEDA_Precip_1990-2020.csv',
            'out_name': 'CRU_CEDA_Precip_EOF_Variance_All.xlsx'
        }
    ]
    
    regions_config = [
        {'name': 'Вся зона степей', 'lon_min': None, 'lon_max': None},
        {'name': 'Европейская часть', 'lon_min': None, 'lon_max': 60},
        {'name': 'Западная Сибирь', 'lon_min': 60, 'lon_max': 90},
        {'name': 'Восточная Сибирь', 'lon_min': 90, 'lon_max': 120},
        {'name': 'Дальний Восток', 'lon_min': 120, 'lon_max': None}
    ]
    
    for item in datasets_config:
        csv_path = RESULTS_DIR / 'validation_results' / 'csv_data' / item['file_name']
        if not csv_path.exists():
            print(f"Ошибка: Файл не найден: {csv_path.name}. Пропуск.")
            continue
            
        out_path = EOF_DIR / item['out_name']
        print(f"   Формирование файла: {item['out_name']}...")
        
        df = pd.read_csv(csv_path)
        df_unique = df.drop_duplicates(subset=['Date', 'WMO']).copy()
        
        with pd.ExcelWriter(out_path, engine='openpyxl') as writer:
            for reg in regions_config:
                sheet_name = reg['name']
                
                # Фильтрация по долготе
                df_sub = df_unique.copy()
                if reg['lon_min'] is not None:
                    df_sub = df_sub[df_sub['Longitude'] >= reg['lon_min']]
                if reg['lon_max'] is not None:
                    df_sub = df_sub[df_sub['Longitude'] < reg['lon_max']]
                    
                if df_sub.empty:
                    empty_df = pd.DataFrame({"Статус": ["Нет станций для анализа"]})
                    empty_df.to_excel(writer, sheet_name=sheet_name, index=False)
                    continue
                    
                # Pivot
                pivot_df = df_sub.pivot(index='Date', columns='WMO', values='Fact')
                pivot_df.index = pd.to_datetime(pivot_df.index)
                pivot_df = pivot_df.sort_index()
                
                # Мягкий фильтр пропусков (<= 10% месяцев)
                max_missing = int(0.10 * len(pivot_df))
                missing_counts = pivot_df.isnull().sum()
                keep_stations = missing_counts[missing_counts <= max_missing].index
                pivot_df = pivot_df[keep_stations].copy()
                
                n_stations = pivot_df.shape[1]
                if n_stations == 0:
                    empty_df = pd.DataFrame({"Статус": ["Нет станций для анализа"]})
                    empty_df.to_excel(writer, sheet_name=sheet_name, index=False)
                    continue
                    
                # Заполнение пропусков
                climatology = pivot_df.groupby(pivot_df.index.month).mean()
                months_series = pd.Series(pivot_df.index.month, index=pivot_df.index)
                for col in pivot_df.columns:
                    if pivot_df[col].isnull().any():
                        fill_values = months_series.map(climatology[col])
                        pivot_df[col] = pivot_df[col].fillna(fill_values)
                        
                # Центрирование (аномалии)
                climatology_filled = pivot_df.groupby(pivot_df.index.month).mean()
                anomalies = pivot_df.copy()
                for month in range(1, 13):
                    anomalies.loc[anomalies.index.month == month] -= climatology_filled.loc[month]
                    
                # Стандартизация
                scaler = StandardScaler()
                scaled_data = scaler.fit_transform(anomalies)
                
                # PCA для всех компонент
                pca = PCA()
                pca.fit(scaled_data)
                
                ev = pca.explained_variance_ratio_ * 100
                cum_ev = np.cumsum(ev)
                
                # Таблица результатов
                res_df = pd.DataFrame({
                    'Компонента': [f'ЕОФ-{i+1}' for i in range(len(ev))],
                    'Объясненная дисперсия (%)': np.round(ev, 4),
                    'Накопленная дисперсия (%)': np.round(cum_ev, 4)
                })
                
                # Корректировка последней строчки до ровно 100%
                if len(res_df) > 0:
                    res_df.loc[res_df.index[-1], 'Накопленная дисперсия (%)'] = 100.0000
                    
                res_df.to_excel(writer, sheet_name=sheet_name, index=False)
                
        print(f"   -> Успешно сохранен: {item['out_name']}")


# --- ОСНОВНАЯ ЛОГИКА ---

def main():
    print(">>> СТАРТ ОБНОВЛЕННОГО МОДУЛЯ ЕОФ (PCA) С СЕЗОНАМИ И ЭКСПОРТОМ <<<\n")
    
    # Шаг 1. Создание папок
    create_directories()
    
    # Инициализация визуализатора для загрузки гео-основы
    if HAS_VISUALIZER:
        viz = BaseVisualizer(base_dir=PROJECT_DIR, results_dir=RESULTS_DIR, subfolder_name="eof_results")
        mask = viz.mask
        world_map = viz.world_map
    else:
        mask = None
        world_map = None

    # Конфигурация для расчетов ЕОФ по сезонам (1990-2020 гг.)
    datasets_config = [
        {
            'name': 'METEO_RU',
            'param': 'DTR',
            'file_name': 'Validation_METEO_RU_DTR_1990-2020.csv',
            'title_ds': 'Метеостанции РФ',
            'title_param': 'DTR',
            'dir_name': 'meteo_ru'
        },
        {
            'name': 'METEO_RU',
            'param': 'Precip',
            'file_name': 'Validation_METEO_RU_Precip_1990-2020.csv',
            'title_ds': 'Метеостанции РФ',
            'title_param': 'Осадки',
            'dir_name': 'meteo_ru'
        },
        {
            'name': 'CRU_CEDA',
            'param': 'DTR',
            'file_name': 'Validation_CRU_CEDA_DTR_1990-2020.csv',
            'title_ds': 'CRU CEDA',
            'title_param': 'DTR',
            'dir_name': 'cru_ceda'
        },
        {
            'name': 'CRU_CEDA',
            'param': 'Precip',
            'file_name': 'Validation_CRU_CEDA_Precip_1990-2020.csv',
            'title_ds': 'CRU CEDA',
            'title_param': 'Осадки',
            'dir_name': 'cru_ceda'
        }
    ]

    variance_results = []

    print("\n2. Анализ ЕОФ по сезонам (1990-2020 гг.)...")
    for item in datasets_config:
        csv_path = RESULTS_DIR / 'validation_results' / 'csv_data' / item['file_name']
        if not csv_path.exists():
            print(f"Ошибка: Файл не найден: {csv_path.name}. Пропуск.")
            continue
            
        print(f"\n   Обработка {item['title_ds']} | {item['title_param']}...")
        
        # 1. Загрузка данных
        df = pd.read_csv(csv_path)
        
        # Убираем дублирование из-за разных моделей, берем только факт
        df_unique = df.drop_duplicates(subset=['Date', 'WMO']).copy()
        
        # Сохраняем координаты станций
        coords_df = df_unique[['WMO', 'Latitude', 'Longitude']].drop_duplicates(subset=['WMO'])
        wmo_to_coords = coords_df.set_index('WMO')[['Latitude', 'Longitude']].to_dict('index')
        
        # 2. Формирование матрицы "Время - Пространство"
        pivot_df = df_unique.pivot(index='Date', columns='WMO', values='Fact')
        pivot_df.index = pd.to_datetime(pivot_df.index)
        pivot_df = pivot_df.sort_index()
        
        # Фильтрация пропусков (мягкая очистка: оставляем станции с пропуском не более 10% месяцев)
        initial_stations = pivot_df.shape[1]
        max_missing_months = 36  # ~10% от 372 месяцев
        missing_counts = pivot_df.isnull().sum()
        keep_stations = missing_counts[missing_counts <= max_missing_months].index
        pivot_df = pivot_df[keep_stations].copy()
        final_stations = pivot_df.shape[1]
        print(f"      Станций до очистки: {initial_stations}, после мягкой фильтрации: {final_stations} (удалено с пропусками > 10%: {initial_stations - final_stations})")
        
        if final_stations == 0:
            print("      Ошибка: Не осталось станций после фильтрации. Пропуск.")
            continue

        # Заполнение редких пропусков климатической нормой для данной станции
        climatology_temp = pivot_df.groupby(pivot_df.index.month).mean()
        months_series = pd.Series(pivot_df.index.month, index=pivot_df.index)
        for col in pivot_df.columns:
            if pivot_df[col].isnull().any():
                fill_values = months_series.map(climatology_temp[col])
                pivot_df[col] = pivot_df[col].fillna(fill_values)

        # 3. Расчет климатических аномалий (центрирование по месяцам)
        climatology = pivot_df.groupby(pivot_df.index.month).mean()
        anomalies = pivot_df.copy()
        for month in range(1, 13):
            anomalies.loc[anomalies.index.month == month] -= climatology.loc[month]
            
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
            anomalies_season = anomalies[anomalies.index.month.isin(s_info['months'])].copy()
            
            if anomalies_season.empty:
                continue
                
            # Стандартизация дисперсии сезонной матрицы
            scaler = StandardScaler()
            scaled_data = scaler.fit_transform(anomalies_season)
            
            # PCA
            pca = PCA(n_components=3)
            pcs = pca.fit_transform(scaled_data)  # Временной ход (PC)
            loadings = pca.components_            # Пространственные веса (EOFs)
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
            plt.plot(anomalies_season.index, pcs[:, 0], label=f'PC-1 ({explained_var[0]:.2f}%)', color='#1f77b4', linewidth=1.5, marker='o', markersize=3)
            plt.plot(anomalies_season.index, pcs[:, 1], label=f'PC-2 ({explained_var[1]:.2f}%)', color='#2ca02c', linewidth=1.5, marker='s', markersize=3)
            plt.plot(anomalies_season.index, pcs[:, 2], label=f'PC-3 ({explained_var[2]:.2f}%)', color='#d62728', linewidth=1.5, marker='^', markersize=3)
            plt.axhline(0, color='black', linestyle='--', linewidth=0.8, alpha=0.7)
            
            plt.title(f"Временной ход глав. компонент (PCs) | Сезон: {s_info['name_ru']}\n{item['title_ds']} | {item['title_param']} (1990-2020 гг.)", fontsize=12, fontweight='bold')
            plt.xlabel("Годы", fontsize=10)
            plt.ylabel("Амплитуда", fontsize=10)
            plt.legend(loc='upper right', fontsize=10)
            plt.grid(True, linestyle=':', alpha=0.5)
            plt.tight_layout()
            
            plot_name = f"PC_timeseries_{item['name']}_{item['param']}.png"
            plt.savefig(EOF_DIR / "plots" / s_name / plot_name, dpi=200)
            plt.close()
            
            # Генерация карт пространственных мод
            png_dir = EOF_DIR / "maps_png" / s_name / item['dir_name'] / item['param']
            tiff_dir = EOF_DIR / "maps_tiff" / s_name / item['dir_name'] / item['param']
            
            lats = np.array([wmo_to_coords[wmo]['Latitude'] for wmo in anomalies_season.columns])
            lons = np.array([wmo_to_coords[wmo]['Longitude'] for wmo in anomalies_season.columns])
            
            lat_min, lat_max = lats.min() - 0.5, lats.max() + 0.5
            lon_min, lon_max = lons.min() - 0.5, lons.max() + 0.5
            grid_lat, grid_lon = np.mgrid[lat_min:lat_max:0.05, lon_min:lon_max:0.05]
            points = np.column_stack((lats, lons))
            
            for m in range(3):
                eof_num = m + 1
                vals = loadings[m]
                
                # Интерполяция
                grid = griddata(points, vals, (grid_lat, grid_lon), method='cubic')
                
                da = xr.DataArray(
                    grid,
                    coords={'latitude': grid_lat[:, 0], 'longitude': grid_lon[0, :]},
                    dims=('latitude', 'longitude')
                )
                da.rio.write_crs("epsg:4326", inplace=True)
                
                if mask is not None:
                    try:
                        da = da.rio.clip(mask.apply(mapping), da.rio.crs, drop=False)
                    except:
                        pass
                
                max_abs = np.nanmax(np.abs(da.values))
                if np.isnan(max_abs) or max_abs == 0:
                    max_abs = 1.0
                vmin, vmax = -max_abs, max_abs
                
                # Отрисовка карты
                fig, ax = plt.subplots(figsize=(18, 6))
                
                if world_map is not None:
                    world_map.plot(ax=ax, color='#e0e0e0', edgecolor='white', linewidth=0.2, zorder=0)
                
                da.plot(ax=ax, cmap='coolwarm', vmin=vmin, vmax=vmax, add_colorbar=False, zorder=1, alpha=1.0)
                ax.grid(True, linestyle=':', alpha=0.4, color='black', linewidth=0.8, zorder=2)
                
                im = ax.collections[-1]
                cbar = plt.colorbar(im, ax=ax, fraction=0.015, pad=0.02, shrink=0.7)
                cbar.set_label("Нагрузка ЕОФ (безразм.)", fontsize=10)
                
                title = f"Пространственная мода ЕОФ-{eof_num} | Сезон: {s_info['name_ru']} | {item['title_ds']} | {item['title_param']}\nДоля объясненной дисперсии: {explained_var[m]:.2f}%"
                ax.set_title(title, fontsize=12, fontweight='bold', pad=10)
                ax.set_xlabel("Долгота", fontsize=10)
                ax.set_ylabel("Широта", fontsize=10)
                
                ax.set_xlim(lon_min - 10, lon_max + 15)
                ax.set_ylim(lat_min - 8, lat_max + 12)
                
                plt.tight_layout()
                
                clean_ds = item['title_ds'].replace(" ", "_")
                clean_param = item['title_param']
                file_name_base = f"Пространственная_мода_ЕОФ-{eof_num}_{clean_ds}_{clean_param}_{s_name}"
                
                png_path = png_dir / f"{file_name_base}.png"
                tiff_path = tiff_dir / f"{file_name_base}.tif"
                
                plt.savefig(png_path, dpi=200)
                plt.savefig(tiff_path, dpi=300, format='tiff', pil_kwargs={'compression': 'tiff_lzw'})
                plt.close()
                
            print(f"        Сгенерированы 3 карты для {s_info['name_ru']}")

    # Шаг 3. Сохранение отчета по дисперсии
    print("\n3. Сохранение отчета по дисперсии...")
    variance_summary_path = EOF_DIR / "variance_summary.txt"
    with open(variance_summary_path, 'w', encoding='utf-8') as f:
        f.write("========================================================================\n")
        f.write("ДОЛИ ОБЪЯСНЕННОЙ ДИСПЕРСИИ ДЛЯ ПЕРВЫХ ТРЕХ МОД ЕОФ ПО СЕЗОНАМ (1990-2020 гг.)\n")
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
