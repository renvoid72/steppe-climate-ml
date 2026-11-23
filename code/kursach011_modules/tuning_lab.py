# -*- coding: utf-8 -*-
"""
МОДУЛЬ: ЛАБОРАТОРИЯ ПОДБОРА ПАРАМЕТРОВ (FAST & UNIVERSAL)
Версия: 3.1 (Regression Only + Universal Grids)

Логика:
1.  ИСКЛЮЧИТЕЛЬНО РЕГРЕССИЯ: Классификатор убран полностью.
2.  УНИВЕРСАЛЬНЫЕ СЕТКИ: Одни и те же диапазоны параметров для всех задач.
3.  БЫСТРОДЕЙСТВИЕ: Используется 25% станций для ускорения.
4.  WALK-FORWARD: Валидация на трёх годах до общего года оценки.
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
    from .meteo_data import load_monthly_meteo
    from .monthly_features import create_forecast_features, model_feature_columns, select_model_rows
    from .project_settings import ANALYSIS_YEAR, cru_csv_path
else:
    from meteo_data import load_monthly_meteo
    from monthly_features import create_forecast_features, model_feature_columns, select_model_rows
    from project_settings import ANALYSIS_YEAR, cru_csv_path

warnings.filterwarnings('ignore')

# ============================================================================
# НАСТРОЙКИ
# ============================================================================
BASE_DIR = Path(__file__).resolve().parent.parent.parent
METEO_DATA_FILE = BASE_DIR / "meteo_ru" / "all_steppe_data.txt"
COORDS_FILE = BASE_DIR / "meteo_ru" / "stations_coords.txt"
CRU_FILE = cru_csv_path(BASE_DIR)

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
        df_monthly = load_monthly_meteo(METEO_DATA_FILE, COORDS_FILE)
        
    elif dataset_name == 'CRU_CEDA':
        df_monthly = pd.read_csv(CRU_FILE)
        
    df_monthly['Date'] = pd.to_datetime(df_monthly[['Year', 'Month']].assign(day=1))
    
    # === ОБРЕЗКА ПО ВРЕМЕНИ ===
    df_monthly = df_monthly[df_monthly['Year'] >= 1960]
    required_columns = ['Latitude', 'Longitude']
    if dataset_name == 'CRU_CEDA':
        required_columns.extend(['DTR', 'Precip'])
    df_monthly = df_monthly.dropna(subset=required_columns)

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
    actuals = df_full.loc[df_full['Year'].isin(test_years), ['Date', 'WMO', target_var]].dropna(subset=[target_var])
    
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
            all_facts = []
            
            try:
                # Walk-Forward Loop
                for year in sorted(test_years):
                    for month in range(1, 13):
                        forecast_date = pd.to_datetime(f"{year}-{month}-01")

                        df_features = create_forecast_features(df_full, forecast_date)
                        df_train = select_model_rows(df_features[df_features['Date'] < forecast_date], target_var)
                        df_pred = select_model_rows(df_features[df_features['Date'] == forecast_date])
                        facts = actuals[actuals['Date'] == forecast_date].set_index('WMO')[target_var]
                        df_pred = df_pred[df_pred['WMO'].isin(facts.index)]

                        if df_pred.empty or len(df_train) < params.get('n_neighbors', 1):
                            continue

                        model_args = {'random_state': 42} if 'Random' in model_name or 'MLP' in model_name else {}
                        if 'Random' in model_name or 'Neighbors' in model_name: model_args['n_jobs'] = -1
                        
                        feature_cols = model_feature_columns(df_train)

                        model = reg_class(**params, **model_args)
                        scaler = StandardScaler().fit(df_train[feature_cols])
                        
                        # Логарифмирование для осадков (ОБЯЗАТЕЛЬНО для регрессии)
                        if target_var == 'Precip':
                            y_train = np.log1p(df_train[target_var])
                        else:
                            y_train = df_train[target_var]
                            
                        model.fit(scaler.transform(df_train[feature_cols]), y_train)
                        
                        raw_preds = model.predict(scaler.transform(df_pred[feature_cols]))
                        
                        # Обратное преобразование
                        if target_var == 'Precip':
                            val = np.mean(np.expm1(raw_preds))
                        else:
                            val = np.mean(raw_preds)
                        
                        all_preds.append(max(0, val))
                        all_facts.append(facts.loc[df_pred['WMO']].mean())

                if all_preds:
                    r2 = r2_score(all_facts, all_preds) if len(all_preds) > 1 else np.nan
                    mae = mean_absolute_error(all_facts, all_preds)
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
    test_years = list(range(ANALYSIS_YEAR - 3, ANALYSIS_YEAR))
    
    # Если хочешь перепроверить DTR - раскомментируй
    # tune_model('METEO_RU', 'DTR', test_years)
    tune_model('METEO_RU', 'Precip', test_years)
    
    # 2. CRU CEDA (3 года тест)
    
    # tune_model('CRU_CEDA', 'DTR', test_years)
    tune_model('CRU_CEDA', 'Precip', test_years)
