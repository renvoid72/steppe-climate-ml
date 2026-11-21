# -*- coding: utf-8 -*-
import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestRegressor
from sklearn.neighbors import KNeighborsRegressor
from sklearn.neural_network import MLPRegressor
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import r2_score, mean_absolute_error
from pathlib import Path
from tempfile import TemporaryDirectory
import warnings

from .meteo_data import load_monthly_meteo
from .monthly_features import (
    create_forecast_features,
    create_time_features,
    model_feature_columns,
    select_model_rows,
)

warnings.filterwarnings('ignore')

FORECAST_COLUMNS = ['Date', 'WMO', 'Latitude', 'Longitude', 'Forecast', 'Fact', 'Model', 'Target_Var']
HISTORY_POLICY = 'complete_features'

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
        """Проверяет, сохранены ли прогнозы с действующим правилом отбора строк."""
        try:
            df = pd.read_csv(self.output_dir / filename, nrows=1)
        except (OSError, ValueError):
            return False

        return (
            not df.empty
            and set(FORECAST_COLUMNS).issubset(df.columns)
            and 'History_Policy' in df.columns
            and df['History_Policy'].iat[0] == HISTORY_POLICY
        )

    def _save_predictions(self, predictions, filename):
        df = pd.concat(predictions) if predictions else pd.DataFrame(columns=FORECAST_COLUMNS)
        df['History_Policy'] = HISTORY_POLICY
        with TemporaryDirectory(dir=self.output_dir) as temp_dir:
            temp_file = Path(temp_dir) / filename
            df.to_csv(temp_file, index=False)
            temp_file.replace(self.output_dir / filename)

    def _can_predict(self, df_train, df_test, model_name, target_var, dataset_name):
        params = BEST_PARAMS[dataset_name][target_var][model_name]
        return not df_test.empty and len(df_train) >= params.get('n_neighbors', 1)

    @staticmethod
    def _print_coverage(candidates, eligible, predicted, paired):
        print(
            f"кандидатов: {candidates}, допущено: {eligible}, исключено: {candidates - eligible}, "
            f"прогнозов: {predicted}, пар с фактом: {paired}"
        )

    def load_data(self, dataset_name):
        print(f"  Загрузка данных для {dataset_name}...")
        
        if dataset_name == 'METEO_RU':
            file_path = self.base_dir / "meteo_ru" / "all_steppe_data.txt"
            coords_path = self.base_dir / "meteo_ru" / "stations_coords.txt"
            
            if not file_path.exists() or not coords_path.exists():
                print(f"ОШИБКА: Не найдены файлы данных в {self.base_dir}/meteo_ru")
                return pd.DataFrame()

            df_monthly = load_monthly_meteo(file_path, coords_path)
            
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
        
        feature_cols = model_feature_columns(df_train)
        
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

            print(f"\n>>> Стандартный прогноз: {target_var} на {target_year} ({dataset_name}) <<<")
            all_predictions = []
            
            for model_name in self.model_classes.keys():
                print(f"  Модель: {model_name}...", end=" ", flush=True)
                model_results = []
                candidate_count = 0
                eligible_count = 0
                
                # Walk-Forward по месяцам
                for month_num in range(1, 13):
                    forecast_date = pd.to_datetime(f"{target_year}-{month_num}-01")
                    
                    df_features = create_forecast_features(df_full, forecast_date)
                    
                    df_train = select_model_rows(df_features[df_features['Date'] < forecast_date], target_var)
                    candidates = df_features[df_features['Date'] == forecast_date]
                    df_pred = select_model_rows(candidates)
                    candidate_count += len(candidates)
                    eligible_count += len(df_pred)

                    if not self._can_predict(df_train, df_pred, model_name, target_var, dataset_name):
                        continue
                    
                    preds = self._fit_predict_model(df_train, df_pred, model_name, target_var, dataset_name)
                    
                    res = df_pred[['Date', 'WMO', 'Latitude', 'Longitude']].copy()
                    res['Forecast'] = preds
                    
                    # ===> СОХРАНЕНИЕ ФАКТА <===
                    facts_slice = df_full[df_full['Date'] == forecast_date][['WMO', target_var]]
                    facts_map = dict(zip(facts_slice['WMO'], facts_slice[target_var]))
                    res['Fact'] = res['WMO'].map(facts_map)
                    
                    res['Model'] = model_name
                    res['Target_Var'] = target_var
                    model_results.append(res)

                self._print_coverage(
                    candidate_count,
                    eligible_count,
                    sum(len(result) for result in model_results),
                    sum(result['Fact'].count() for result in model_results),
                )

                if not model_results:
                    print("    Нет строк для прогноза.")
                    continue

                model_predictions = pd.concat(model_results)
                all_predictions.append(model_predictions)
                estimated = model_predictions.dropna(subset=['Fact'])
                monthly = estimated.groupby('Date')[['Fact', 'Forecast']].mean()
                if monthly.empty:
                    print("    Нет фактов для оценки.")
                    continue

                r2 = r2_score(monthly['Fact'], monthly['Forecast']) if len(monthly) > 1 else np.nan
                mae = mean_absolute_error(monthly['Fact'], monthly['Forecast'])
                print(f"    R2={r2:.4f}, MAE={mae:.4f}")
            
            self._save_predictions(all_predictions, fname)
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
                candidate_count = 0
                eligible_count = 0

                # Yearly Walk-Forward
                for year in test_years:
                    #print(f"{year}", end=" ", flush=True)
                    split_date = pd.to_datetime(f"{year}-01-01")
                    
                    df_current_slice = df_full[df_full['Year'] <= year].copy()
                    df_features = self.create_features(df_current_slice)
                    
                    df_train = select_model_rows(df_features[df_features['Date'] < split_date], target_var)
                    candidates = df_features[df_features['Year'] == year]
                    df_test = select_model_rows(candidates, target_var)
                    candidate_count += len(candidates)
                    eligible_count += len(df_test)
                    
                    if not self._can_predict(df_train, df_test, model_name, target_var, dataset_name):
                        continue

                    preds = self._fit_predict_model(df_train, df_test, model_name, target_var, dataset_name)
                    
                    res = df_test[['Date', 'WMO', 'Latitude', 'Longitude']].copy()
                    res['Forecast'] = preds
                    res['Fact'] = df_test[target_var] # Сохраняем факт
                    res['Model'] = model_name
                    res['Target_Var'] = target_var
                    all_predictions.append(res)
                    
                    model_full_preds.extend(preds)
                    model_full_facts.extend(df_test[target_var].values)

                self._print_coverage(candidate_count, eligible_count, len(model_full_preds), len(model_full_facts))

                if model_full_preds:
                    r2 = r2_score(model_full_facts, model_full_preds) if len(model_full_preds) > 1 else np.nan
                    mae = mean_absolute_error(model_full_facts, model_full_preds)
                    print(f"      Avg R2={r2:.4f}, Avg MAE={mae:.4f}")
                else:
                    print("      Нет данных.")

            self._save_predictions(all_predictions, fname)
            print(f"  -> Результаты валидации сохранены в {fname}")
