# -*- coding: utf-8 -*-
"""
МОДУЛЬ: ЛАБОРАТОРИЯ ПОДБОРА ПАРАМЕТРОВ (FAST & UNIVERSAL)
Версия: 3.1 (Regression Only + Universal Grids)

Логика:
1.  ИСКЛЮЧИТЕЛЬНО РЕГРЕССИЯ: Классификатор убран полностью.
2.  УНИВЕРСАЛЬНЫЕ СЕТКИ: Одни и те же диапазоны параметров для всех задач.
3.  БЫСТРОДЕЙСТВИЕ: Используется 25% станций для ускорения.
4.  WALK-FORWARD: Валидация на 3 годах (2021-2023 для Метео, 2017-2019 для CRU).
"""
import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestRegressor
from sklearn.neighbors import KNeighborsRegressor
from sklearn.neural_network import MLPRegressor
from sklearn.metrics import r2_score, mean_absolute_error
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import ParameterSampler
import warnings
import time
from pathlib import Path
import random

if __package__:
    from .monthly_features import create_time_features
else:
    from monthly_features import create_time_features

warnings.filterwarnings('ignore')

# ============================================================================
# НАСТРОЙКИ
# ============================================================================
BASE_DIR = Path(__file__).resolve().parent.parent.parent
METEO_DATA_FILE = BASE_DIR / "meteo_ru" / "all_steppe_data.txt"
COORDS_FILE = BASE_DIR / "meteo_ru" / "stations_coords.txt"
CRU_FILE = BASE_DIR / "cru ceda" / "cru_aligned_data.csv"

# Ускорение: берем только 25% станций
STATION_FRACTION = 0.45
# Количество попыток для каждой модели
N_ITER = 10

# === УНИВЕРСАЛЬНЫЕ СЕТКИ ПАРАМЕТРОВ ===
# (Подходят и для DTR, и для Осадков)
PARAM_GRIDS = {
    'Random Forest': {
        'n_estimators': [100, 150, 200], 
        'max_depth': [7, 10, 15, 20], 
        'min_samples_leaf': [5, 10, 15, 20],
        'max_features': ['sqrt', 1.0] # 1.0 = все признаки
    },
    'K-Neighbors': {
        'n_neighbors': [15, 25, 40, 50], 
        'weights': ['uniform', 'distance']
    },
    'MLP Neural Network': {
        'hidden_layer_sizes': [(50,), (100,), (50, 25)], 
        'alpha': [0.1, 1, 5], 
        'activation': ['tanh'], # tanh стабильнее для климата
        'early_stopping': [True]
    }
}

# ============================================================================
# ФУНКЦИИ
# ============================================================================
def load_data_sampled(dataset_name):
    print(f"--- Загрузка и сэмплирование: {dataset_name} ---")
    
    if dataset_name == 'METEO_RU':
        cols = ['WMO', 'Year', 'Month', 'Day', 'Quality', 'Tmin', 'Tavg', 'Tmax', 'Precip']
        df = pd.read_csv(METEO_DATA_FILE, sep=r'\s+', names=cols, na_values=[-999.9])
        df['DTR'] = df['Tmax'] - df['Tmin']
        
        # Координаты нужны для группировки при сэмплировании
        coords = pd.read_csv(COORDS_FILE, sep=r'\s+')
        lat_map = dict(zip(coords['WMO'], coords['Lat']))
        lon_map = dict(zip(coords['WMO'], coords['Lon']))
        df['Latitude'] = df['WMO'].map(lat_map)
        df['Longitude'] = df['WMO'].map(lon_map)
        
        # Агрегация до месяца
        df_monthly = df.groupby(['WMO', 'Year', 'Month', 'Latitude', 'Longitude']).agg(
            DTR=('DTR', 'mean'), Precip=('Precip', 'sum')
        ).reset_index()
        
    elif dataset_name == 'CRU_CEDA':
        df_monthly = pd.read_csv(CRU_FILE)
        
    df_monthly['Date'] = pd.to_datetime(df_monthly[['Year', 'Month']].assign(day=1))
    
    # === ОБРЕЗКА ПО ВРЕМЕНИ ===
    df_monthly = df_monthly[df_monthly['Year'] >= 1960]
    df_monthly = df_monthly.dropna(subset=['DTR', 'Precip', 'Latitude', 'Longitude'])

    # === СЭМПЛИРОВАНИЕ (Берем 25% станций) ===
    all_stations = df_monthly['WMO'].unique()
    n_samples = max(5, int(len(all_stations) * STATION_FRACTION))
    selected_stations = np.random.choice(all_stations, n_samples, replace=False)
    
    df_sampled = df_monthly[df_monthly['WMO'].isin(selected_stations)].copy()
    
    print(f"  Выбрано {len(selected_stations)} из {len(all_stations)} станций.")
    print(f"  Строк данных: {len(df_sampled)}")
    return df_sampled

# ============================================================================
# ЯДРО ПОДБОРА
# ============================================================================
def tune_model(dataset_name, target_var, test_years):
    print(f"\n{'='*60}")
    print(f"ПОДБОР: {dataset_name} | Цель: {target_var}")
    print(f"Тест: {test_years}")
    print(f"{'='*60}")

    df_full = load_data_sampled(dataset_name)
    actuals = df_full[df_full['Year'].isin(test_years)].groupby('Date')[target_var].mean()
    
    if actuals.empty:
        print("ОШИБКА: Нет данных для теста!")
        return

    model_classes = {
        'Random Forest': RandomForestRegressor,
        'K-Neighbors': KNeighborsRegressor,
        'MLP Neural Network': MLPRegressor
    }
    
    for model_name, reg_class in model_classes.items():
        print(f"\n>>> Модель: {model_name} <<<")
        
        # Берем параметры из универсальной сетки
        grid = PARAM_GRIDS[model_name]
        search_combinations = list(ParameterSampler(grid, n_iter=N_ITER, random_state=42))

        best_score = -np.inf
        best_params = None
        
        for i, params in enumerate(search_combinations):
            start_time = time.time()
            all_preds = []
            
            try:
                # Walk-Forward Loop
                for year in sorted(test_years):
                    for month in range(1, 13):
                        forecast_date = pd.to_datetime(f"{year}-{month}-01")
                        
                        df_history = df_full[df_full['Date'] < forecast_date].copy()
                        
                        stations = df_history[['WMO', 'Latitude', 'Longitude']].drop_duplicates()
                        future = stations.copy()
                        future['Date'] = forecast_date
                        for c in ['DTR', 'Precip']: 
                            if c in df_history.columns: future[c] = np.nan
                        
                        df_combined = pd.concat([df_history, future], ignore_index=True)
                        df_features = create_time_features(df_combined)
                        
                        # Строки с пропусками исключаются только из обучающей выборки.
                        df_train = df_features[df_features['Date'] < forecast_date].dropna().reset_index(drop=True)
                        X_pred_input = df_features[df_features['Date'] == forecast_date].reset_index(drop=True).fillna(0)
                        
                        if df_train.empty: continue

                        model_args = {'random_state': 42} if 'Random' in model_name or 'MLP' in model_name else {}
                        if 'Random' in model_name or 'Neighbors' in model_name: model_args['n_jobs'] = -1
                        
                        # Колонки для обучения (без классификаторов)
                        cols_exclude = ['Date', 'WMO', 'Year', 'Month', 'Quality', 'Tmin', 'Tavg', 'Tmax', 'Precip', 'DTR', 'Latitude', 'Longitude']
                        feature_cols = [c for c in df_train.columns if c not in cols_exclude]

                        model = reg_class(**params, **model_args)
                        scaler = StandardScaler().fit(df_train[feature_cols])
                        
                        # Логарифмирование для осадков (ОБЯЗАТЕЛЬНО для регрессии)
                        if target_var == 'Precip':
                            y_train = np.log1p(df_train[target_var])
                        else:
                            y_train = df_train[target_var]
                            
                        model.fit(scaler.transform(df_train[feature_cols]), y_train)
                        
                        raw_preds = model.predict(scaler.transform(X_pred_input[feature_cols]))
                        
                        # Обратное преобразование
                        if target_var == 'Precip':
                            val = np.mean(np.expm1(raw_preds))
                        else:
                            val = np.mean(raw_preds)
                        
                        val = max(0, val)
                        all_preds.append(pd.Series([val], index=[forecast_date]))

                if all_preds:
                    full_preds = pd.concat(all_preds)
                    act, pred = actuals.align(full_preds.squeeze(), join='inner')
                    if len(act) > 0:
                        r2 = r2_score(act, pred)
                        mae = mean_absolute_error(act, pred)
                        print(f"  [{i+1}/{N_ITER}] R2={r2:.4f} | MAE={mae:.3f} | {params} ({time.time()-start_time:.1f}s)")
                        
                        if r2 > best_score:
                            best_score, best_params = r2, params
            except Exception as e:
                print(f"  [{i+1}/{N_ITER}] ОШИБКА: {e}")

        print(f"\n>>> ЛУЧШИЙ РЕЗУЛЬТАТ ({model_name}): R2 = {best_score:.4f}")
        print(f"    {best_params}")

# ============================================================================
# ЗАПУСК
# ============================================================================
if __name__ == "__main__":
    print("ЗАПУСК БЫСТРОГО ПОДБОРА (REGR ONLY, UNIVERSAL GRIDS)")
    
    # 1. МЕТЕО.РУ (3 года тест)
    test_years_meteo = [2021, 2022, 2023]
    
    # Если хочешь перепроверить DTR - раскомментируй
    # tune_model('METEO_RU', 'DTR', test_years_meteo)
    tune_model('METEO_RU', 'Precip', test_years_meteo)
    
    # 2. CRU CEDA (3 года тест)
    test_years_cru = [2017, 2018, 2019]
    
    # tune_model('CRU_CEDA', 'DTR', test_years_cru)
    tune_model('CRU_CEDA', 'Precip', test_years_cru)