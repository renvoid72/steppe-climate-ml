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

from .annual_forecast import (
    ANNUAL_FORECAST_MODE, annual_prediction_rows, annual_splits, annual_targets, prepare_annual_features,
)
from .meteo_data import load_monthly_meteo
from .project_settings import CRU_VERSION, cru_csv_path, forecast_filename, validation_filename
from .monthly_features import (
    prepare_forecast_period,
    create_time_features,
    model_feature_columns,
    select_model_rows,
)

warnings.filterwarnings('ignore')

FORECAST_COLUMNS = ['Date', 'WMO', 'Latitude', 'Longitude', 'Forecast', 'Fact', 'Model', 'Target_Var']
HISTORY_POLICIES = {
    'METEO_RU': 'complete_features_calendar_coverage',
    'CRU_CEDA': 'complete_features',
}
CRU_SOURCE_VERSION = f'CRU TS {CRU_VERSION}'


def mark_prediction_source(df, dataset_name):
    if dataset_name == 'CRU_CEDA':
        df['Source_Version'] = CRU_SOURCE_VERSION


def read_prediction_csv(path, dataset_name, *, nrows=None):
    df = pd.read_csv(path, nrows=nrows)
    if dataset_name == 'CRU_CEDA' and (
        'Source_Version' not in df or not df['Source_Version'].eq(CRU_SOURCE_VERSION).all()
    ):
        raise ValueError(
            f'{path}: требуется результат расчёта на {CRU_SOURCE_VERSION}. '
            'Пересчитайте прогноз или валидацию.'
        )
    return df


def read_validation_csv(path, dataset_name):
    df = read_prediction_csv(path, dataset_name)
    if 'Forecast_Mode' not in df or not df['Forecast_Mode'].eq(ANNUAL_FORECAST_MODE).all():
        raise ValueError(f'{path}: пересчитайте валидацию для прямого прогноза на 12 месяцев.')
    return df


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
    forecast_mode = None

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

    def _is_file_valid(self, filename, dataset_name):
        """Проверяет версию CRU и действующее правило отбора строк."""
        try:
            # Проверяем версию во всех строках CRU, включая вручную объединённые CSV.
            nrows = None if dataset_name == 'CRU_CEDA' else 1
            path = self.output_dir / filename
            if self.forecast_mode == ANNUAL_FORECAST_MODE:
                df = read_validation_csv(path, dataset_name)
            else:
                df = read_prediction_csv(path, dataset_name, nrows=nrows)
        except (OSError, ValueError):
            return False

        return (
            not df.empty
            and set(FORECAST_COLUMNS).issubset(df.columns)
            and 'History_Policy' in df.columns
            and df['History_Policy'].iat[0] == HISTORY_POLICIES[dataset_name]
        )

    def _save_predictions(self, predictions, filename, dataset_name):
        df = pd.concat(predictions) if predictions else pd.DataFrame(columns=FORECAST_COLUMNS)
        df['History_Policy'] = HISTORY_POLICIES[dataset_name]
        if self.forecast_mode is not None:
            df['Forecast_Mode'] = self.forecast_mode
        mark_prediction_source(df, dataset_name)
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
            file_path = cru_csv_path(self.base_dir)
            if not file_path.exists():
                print(f"ОШИБКА: Не найден файл {file_path}")
                return pd.DataFrame()
            df_monthly = pd.read_csv(file_path)
        else:
            raise ValueError("Неизвестный dataset_name")

        df_monthly['Date'] = pd.to_datetime(df_monthly[['Year', 'Month']].assign(day=1))
        
        # Обрезка с 1960 года для корректности
        df_monthly = df_monthly[df_monthly['Year'] >= 1960]
        
        required_columns = ['Latitude', 'Longitude']
        if dataset_name == 'CRU_CEDA':
            required_columns.extend(['DTR', 'Precip'])
        df_monthly = df_monthly.dropna(subset=required_columns)
        
        print(f"    Загружено {len(df_monthly)} записей.")
        return df_monthly

    def create_features(self, df):
        return create_time_features(df)

    def _fit_predict_model(self, df_train, df_test, model_name, target_var, dataset_name, *, y_train=None):
        """Инкапсулированная логика обучения одной модели"""
        params = BEST_PARAMS[dataset_name][target_var][model_name]
        
        feature_cols = model_feature_columns(df_train)
        
        scaler = StandardScaler().fit(df_train[feature_cols])
        
        model_args = {'random_state': 42} if 'Random' in model_name or 'MLP' in model_name else {}
        if 'Random' in model_name or 'Neighbors' in model_name: model_args['n_jobs'] = -1
        model = self.model_classes[model_name](**params, **model_args)
        
        if y_train is None:
            y_train = df_train[target_var]
        if target_var == 'Precip':
            y_train = np.log1p(y_train)
        
        model.fit(scaler.transform(df_train[feature_cols]), y_train)
        
        raw_preds = model.predict(scaler.transform(df_test[feature_cols]))
        
        final_preds = np.expm1(raw_preds) if target_var == 'Precip' else raw_preds
        return np.maximum(0, final_preds)

    def annual_model_splits(self, features, targets, test_years, model_name, target_var, dataset_name):
        for train, test in annual_splits(features, targets, test_years):
            df_train, df_test = features.loc[train], features.loc[test]
            if self._can_predict(df_train, df_test, model_name, target_var, dataset_name):
                yield df_train, df_test, targets[train], targets[test]

# ============================================================================
# СТАНДАРТНЫЙ ПРОГНОЗ НА СЛЕДУЮЩИЙ МЕСЯЦ
# ============================================================================
class StandardForecaster(BaseForecaster):
    def __init__(self, base_dir, root_output_dir):
        super().__init__(base_dir, root_output_dir, "standard_forecast")
    
    def run(self, target_year, dataset_name):
        df_full = None
        forecast_dates = pd.date_range(f'{target_year}-01-01', periods=12, freq='MS')
        
        for target_var in ['DTR', 'Precip']:
            fname = forecast_filename(dataset_name, target_var, target_year)
            
            # ПРОВЕРКА ФАЙЛА
            if self._is_file_valid(fname, dataset_name):
                print(f"  [SKIP] Файл {fname} уже существует и корректен.")
                continue

            if df_full is None:
                df_full = self.load_data(dataset_name)
                if df_full.empty:
                    return
                history, forecast_rows = prepare_forecast_period(df_full, forecast_dates)
                eligible_forecasts = select_model_rows(forecast_rows)

            print(f"\n>>> Стандартный прогноз: {target_var} на {target_year} ({dataset_name}) <<<")
            all_predictions = []
            train_rows = select_model_rows(history, target_var)
            
            for model_name in self.model_classes.keys():
                print(f"  Модель: {model_name}...", end=" ", flush=True)
                model_results = []
                
                # Walk-Forward по месяцам
                for forecast_date in forecast_dates:
                    df_train = train_rows.loc[train_rows['Date'] < forecast_date]
                    df_pred = eligible_forecasts.loc[eligible_forecasts['Date'] == forecast_date]

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
                    len(forecast_rows),
                    len(eligible_forecasts),
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
            
            self._save_predictions(all_predictions, fname, dataset_name)
            print(f"  -> Сохранено: {fname}")

# ============================================================================
# ДОЛГОСРОЧНАЯ ВАЛИДАЦИЯ ПРЯМОГО ПРОГНОЗА НА 12 МЕСЯЦЕВ
# ============================================================================
class LongTermValidator(BaseForecaster):
    forecast_mode = ANNUAL_FORECAST_MODE

    def __init__(self, base_dir, root_output_dir):
        super().__init__(base_dir, root_output_dir, "validation_results")

    def run(self, split_year, dataset_name, end_year=None):
        df_full = self.load_data(dataset_name)
        if df_full.empty:
            return
        final_year = min(end_year, df_full['Year'].max()) if end_year is not None else df_full['Year'].max()
        test_years = range(split_year, final_year + 1)
        if not test_years:
            return

        features = None
        for target_var in ['DTR', 'Precip']:
            fname = validation_filename(dataset_name, target_var, split_year, final_year)
            if self._is_file_valid(fname, dataset_name):
                print(f"  [SKIP] Файл {fname} уже существует и корректен.")
                continue

            if features is None:
                candidates = prepare_annual_features(df_full, final_year)
                features = select_model_rows(candidates)
                candidate_count = (candidates['Date'].dt.year >= split_year).sum()
                eligible_count = (features['Date'].dt.year >= split_year).sum()
                years_without_features = sorted(set(test_years) - set(features['Date'].dt.year))
                if years_without_features:
                    years = ', '.join(map(str, years_without_features))
                    print(f"  {dataset_name}: нет станций с полным набором признаков за {years} гг.")

            print(f"\n>>> Прямой прогноз на 12 месяцев: {dataset_name} | Валидация: {split_year}-{final_year} <<<")
            print(f"  Переменная: {target_var}...")
            targets = annual_targets(df_full, features, target_var)
            all_predictions = []

            for model_name in self.model_classes:
                print(f"    {model_name}:")
                print(f"      Пар станция-год до отбора: {candidate_count}, после отбора: {eligible_count}")
                model_results = []
                splits = self.annual_model_splits(features, targets, test_years, model_name, target_var, dataset_name)
                for df_train, df_test, y_train, facts in splits:
                    predictions = self._fit_predict_model(
                        df_train, df_test, model_name, target_var, dataset_name, y_train=y_train,
                    )
                    model_results.append(annual_prediction_rows(df_test, predictions, facts, model_name, target_var))

                if not model_results:
                    print("      Нет строк для прогноза.")
                    continue

                results = pd.concat(model_results)
                all_predictions.append(results)
                paired = results.dropna(subset=['Fact'])
                print(f"      Месячных прогнозов: {len(results)}, пар с фактом: {len(paired)}")
                if not paired.empty:
                    r2 = r2_score(paired['Fact'], paired['Forecast']) if len(paired) > 1 else np.nan
                    mae = mean_absolute_error(paired['Fact'], paired['Forecast'])
                    print(f"      R2={r2:.4f}, MAE={mae:.4f}")

            self._save_predictions(all_predictions, fname, dataset_name)
            print(f"  -> Результаты валидации сохранены в {fname}")
