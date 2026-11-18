# -*- coding: utf-8 -*-
import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestRegressor
from sklearn.neighbors import KNeighborsRegressor
from sklearn.neural_network import MLPRegressor
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import r2_score, mean_absolute_error
from pathlib import Path
import warnings
import sys
import os

from .monthly_features import create_time_features

warnings.filterwarnings('ignore')

# Параметры (Regression Only)
BEST_PARAMS = {
    'METEO_RU': {
        'DTR': {
            'Random Forest': {'n_estimators': 150, 'max_depth': 15, 'min_samples_leaf': 10},
            'K-Neighbors': {'n_neighbors': 25, 'weights': 'uniform'},
            'MLP Neural Network': {'hidden_layer_sizes': (100,), 'alpha': 0.1, 'activation': 'tanh'}
        },
        'Precip': {
            'Random Forest': {'n_estimators': 150, 'max_depth': 10, 'min_samples_leaf': 15},
            'K-Neighbors': {'n_neighbors': 25, 'weights': 'distance'},
            'MLP Neural Network': {'hidden_layer_sizes': (50, 25), 'alpha': 1, 'activation': 'tanh'}
        }
    },
    'CRU_CEDA': {
        'DTR': {
            'Random Forest': {'n_estimators': 200, 'max_depth': 20, 'min_samples_leaf': 10},
            'K-Neighbors': {'n_neighbors': 15, 'weights': 'uniform'},
            'MLP Neural Network': {'hidden_layer_sizes': (100,), 'alpha': 0.1, 'activation': 'tanh'}
        },
        'Precip': {
            'Random Forest': {'n_estimators': 200, 'max_depth': 10, 'min_samples_leaf': 10},
            'K-Neighbors': {'n_neighbors': 15, 'weights': 'distance'},
            'MLP Neural Network': {'hidden_layer_sizes': (50, 25), 'alpha': 1, 'activation': 'tanh'}
        }
    }
}

# ==============================================================================
# БАЗОВЫЙ КЛАСС (BaseForecaster) - Должен быть объявлен ПЕРВЫМ
# ==============================================================================
class BaseForecaster:
    def __init__(self, base_dir, root_output_dir, subfolder_name):
        self.base_dir = Path(base_dir)
        # Папка для сохранения CSV результатов
        self.output_dir = Path(root_output_dir) / subfolder_name / "csv_data"
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        self.model_classes = {
            'Random Forest': RandomForestRegressor,
            'K-Neighbors': KNeighborsRegressor,
            'MLP Neural Network': MLPRegressor
        }
        print(f"[{self.__class__.__name__}] Инициализирован. Папка: {self.output_dir}")

    def _is_file_valid(self, filename):
        """Проверяет, существует ли файл и содержит ли он нужные данные"""
        filepath = self.output_dir / filename
        if not filepath.exists():
            return False
        
        try:
            # Читаем только заголовок, чтобы было быстро
            df = pd.read_csv(filepath, nrows=5)
            required_cols = ['Date', 'WMO', 'Forecast', 'Fact', 'Model']
            
            # Если всех нужных колонок нет - файл невалиден
            if not all(col in df.columns for col in required_cols):
                return False
                
            return True
        except:
            return False

    def load_data(self, dataset_name):
        print(f"  Загрузка данных для {dataset_name}...")
        
        if dataset_name == 'METEO_RU':
            file_path = self.base_dir / "meteo_ru" / "all_steppe_data.txt"
            coords_path = self.base_dir / "meteo_ru" / "stations_coords.txt"
            
            if not file_path.exists() or not coords_path.exists():
                print(f"ОШИБКА: Не найдены файлы данных в {self.base_dir}/meteo_ru")
                return pd.DataFrame()

            cols = ['WMO', 'Year', 'Month', 'Day', 'Quality', 'Tmin', 'Tavg', 'Tmax', 'Precip']
            df = pd.read_csv(file_path, sep=r'\s+', names=cols, na_values=[-999.9])
            df['DTR'] = df['Tmax'] - df['Tmin']
            
            coords = pd.read_csv(coords_path, sep=r'\s+')
            lat_map = dict(zip(coords['WMO'], coords['Lat']))
            lon_map = dict(zip(coords['WMO'], coords['Lon']))
            
            df['Latitude'] = df['WMO'].map(lat_map)
            df['Longitude'] = df['WMO'].map(lon_map)
            
            df_monthly = df.groupby(['WMO', 'Year', 'Month', 'Latitude', 'Longitude']).agg(
                DTR=('DTR', 'mean'), 
                Precip=('Precip', 'sum')
            ).reset_index()
            
        elif dataset_name == 'CRU_CEDA':
            file_path = self.base_dir / "cru ceda" / "cru_aligned_data.csv"
            if not file_path.exists():
                print(f"ОШИБКА: Не найден файл {file_path}")
                return pd.DataFrame()
            df_monthly = pd.read_csv(file_path)
        else:
            raise ValueError("Неизвестный dataset_name")

        df_monthly['Date'] = pd.to_datetime(df_monthly[['Year', 'Month']].assign(day=1))
        
        # Обрезка с 1960 года для корректности
        df_monthly = df_monthly[df_monthly['Year'] >= 1960]
        
        df_monthly = df_monthly.dropna(subset=['DTR', 'Precip', 'Latitude', 'Longitude'])
        
        print(f"    Загружено {len(df_monthly)} записей.")
        return df_monthly

    def create_features(self, df):
        return create_time_features(df)

    def _fit_predict_model(self, df_train, df_test, model_name, target_var, dataset_name):
        """Инкапсулированная логика обучения одной модели"""
        params = BEST_PARAMS[dataset_name][target_var][model_name]
        
        cols_exclude = ['Date', 'WMO', 'Year', 'Month', 'Quality', 'Tmin', 'Tavg', 'Tmax', 'Precip', 'DTR', 'Latitude', 'Longitude']
        feature_cols = [c for c in df_train.columns if c not in cols_exclude]
        
        scaler = StandardScaler().fit(df_train[feature_cols])
        
        model_args = {'random_state': 42} if 'Random' in model_name or 'MLP' in model_name else {}
        if 'Random' in model_name or 'Neighbors' in model_name: model_args['n_jobs'] = -1
        model = self.model_classes[model_name](**params, **model_args)
        
        y_train = np.log1p(df_train[target_var]) if target_var == 'Precip' else df_train[target_var]
        
        model.fit(scaler.transform(df_train[feature_cols]), y_train)
        
        raw_preds = model.predict(scaler.transform(df_test[feature_cols]))
        
        final_preds = np.expm1(raw_preds) if target_var == 'Precip' else raw_preds
        return np.maximum(0, final_preds)

# ============================================================================
# НАСЛЕДНИК 1: СТАНДАРТНЫЙ ПРОГНОЗАТОР (на 1 год)
# ============================================================================
class StandardForecaster(BaseForecaster):
    def __init__(self, base_dir, root_output_dir):
        super().__init__(base_dir, root_output_dir, "standard_forecast")
    
    def run(self, target_year, dataset_name):
        df_full = None # Ленивая загрузка
        
        for target_var in ['DTR', 'Precip']:
            fname = f"{dataset_name}_{target_var}_{target_year}.csv"
            
            # ПРОВЕРКА ФАЙЛА
            if self._is_file_valid(fname):
                print(f"  [SKIP] Файл {fname} уже существует и корректен.")
                continue

            if df_full is None: 
                df_full = self.load_data(dataset_name)
            
            if df_full.empty: return

            # Получаем реальные данные для сравнения
            actuals = df_full[df_full['Year'] == target_year].groupby('Date')

            print(f"\n>>> Стандартный прогноз: {target_var} на {target_year} ({dataset_name}) <<<")
            all_predictions = []
            
            for model_name in self.model_classes.keys():
                print(f"  Модель: {model_name}...", end=" ", flush=True)
                preds_list = []
                
                # Walk-Forward по месяцам
                for month_num in range(1, 13):
                    forecast_date = pd.to_datetime(f"{target_year}-{month_num}-01")
                    
                    df_hist = df_full[df_full['Date'] < forecast_date].copy()
                    future = df_hist[['WMO', 'Latitude', 'Longitude']].drop_duplicates()
                    future['Date'] = forecast_date
                    for c in ['DTR', 'Precip']: future[c] = np.nan
                    
                    df_combined = pd.concat([df_hist, future], ignore_index=True)
                    df_features = self.create_features(df_combined)
                    
                    df_train = df_features[df_features['Date'] < forecast_date].dropna().reset_index(drop=True)
                    df_pred = df_features[df_features['Date'] == forecast_date].fillna(0)
                    
                    preds = self._fit_predict_model(df_train, df_pred, model_name, target_var, dataset_name)
                    
                    res = df_pred[['Date', 'WMO', 'Latitude', 'Longitude']].copy()
                    res['Forecast'] = preds
                    
                    # ===> СОХРАНЕНИЕ ФАКТА <===
                    facts_slice = df_full[df_full['Date'] == forecast_date][['WMO', target_var]]
                    facts_map = dict(zip(facts_slice['WMO'], facts_slice[target_var]))
                    res['Fact'] = res['WMO'].map(facts_map)
                    
                    res['Model'] = model_name
                    res['Target_Var'] = target_var
                    all_predictions.append(res)
                    preds_list.append(pd.Series([np.mean(preds)], index=[forecast_date]))
                
                # Вывод метрик
                model_year_preds = pd.concat(preds_list)
                act_year = actuals[target_var].mean()
                
                aligned_act, aligned_pred = act_year.align(model_year_preds.squeeze(), join='inner')
                if not aligned_act.empty:
                    r2 = r2_score(aligned_act, aligned_pred)
                    mae = mean_absolute_error(aligned_act, aligned_pred)
                    print(f"R²={r2:.4f}, MAE={mae:.4f}")
                else:
                    print("Готово")
            
            final_df = pd.concat(all_predictions)
            final_df.to_csv(self.output_dir / fname, index=False)
            print(f"  -> Сохранено: {fname}")

# ============================================================================
# НАСЛЕДНИК 2: ВАЛИДАТОР (ДЛИННЫЙ ТЕСТ)
# ============================================================================
class LongTermValidator(BaseForecaster):
    def __init__(self, base_dir, root_output_dir):
        super().__init__(base_dir, root_output_dir, "validation_results")

    def run(self, split_year, dataset_name, end_year=None):
        df_full = None

        for target_var in ['DTR', 'Precip']:
            # Логика определения конечного года для имени файла
            # Нам нужно загрузить данные хотя бы раз, чтобы узнать max_year, если end_year не задан
            if df_full is None:
                df_full = self.load_data(dataset_name)
            
            if df_full.empty: return

            data_max_year = df_full['Year'].max()
            final_year = end_year if end_year else data_max_year
            final_year = min(final_year, data_max_year)
            
            test_years = list(range(split_year, final_year + 1))
            
            # ПРОВЕРКА ФАЙЛА
            fname = f"Validation_{dataset_name}_{target_var}_{split_year}-{final_year}.csv"
            if self._is_file_valid(fname):
                print(f"  [SKIP] Файл {fname} уже существует и корректен.")
                continue

            print(f"\n>>> Валидация (Backtesting): {dataset_name} | Тест: {split_year}-{final_year} <<<")
            print(f"  Переменная: {target_var}...")
            all_predictions = []
            
            for model_name in self.model_classes.keys():
                print(f"    {model_name}:", end=" ", flush=True)
                
                model_full_preds = []
                model_full_facts = []

                # Yearly Walk-Forward
                for year in test_years:
                    #print(f"{year}", end=" ", flush=True)
                    split_date = pd.to_datetime(f"{year}-01-01")
                    
                    df_current_slice = df_full[df_full['Year'] <= year].copy()
                    df_features = self.create_features(df_current_slice)
                    
                    df_train = df_features[df_features['Date'] < split_date].dropna().reset_index(drop=True)
                    df_test = df_features[df_features['Year'] == year].reset_index(drop=True).fillna(0)
                    
                    if df_train.empty or df_test.empty: continue

                    preds = self._fit_predict_model(df_train, df_test, model_name, target_var, dataset_name)
                    
                    res = df_test[['Date', 'WMO', 'Latitude', 'Longitude']].copy()
                    res['Forecast'] = preds
                    res['Fact'] = df_test[target_var] # Сохраняем факт
                    res['Model'] = model_name
                    res['Target_Var'] = target_var
                    all_predictions.append(res)
                    
                    model_full_preds.extend(preds)
                    model_full_facts.extend(df_test[target_var].values)
                
                if model_full_preds:
                    r2 = r2_score(model_full_facts, model_full_preds)
                    mae = mean_absolute_error(model_full_facts, model_full_preds)
                    print(f" | Avg R²={r2:.4f}, Avg MAE={mae:.4f}")
                else:
                    print(" | Нет данных.")

            final_df = pd.concat(all_predictions)
            final_df.to_csv(self.output_dir / fname, index=False)
            print(f"  -> Результаты валидации сохранены в {fname}")