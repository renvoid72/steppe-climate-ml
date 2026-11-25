# -*- coding: utf-8 -*-
import json
from hashlib import sha256

import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestRegressor
from sklearn.neighbors import KNeighborsRegressor
from sklearn.neural_network import MLPRegressor
from sklearn.preprocessing import StandardScaler
from pathlib import Path
from tempfile import TemporaryDirectory
import warnings

from .annual_forecast import (
    ANNUAL_FORECAST_MODE, annual_prediction_rows, annual_splits, annual_targets, prepare_annual_features,
)
from .forecast_metrics import (
    calculate_climatology_comparison, format_forecast_metrics, paired_predictions, station_month_climatology,
)
from .meteo_data import load_monthly_meteo
from .model_parameters import BEST_PARAMS, MODEL_PARAMS_FILENAME, MONTHLY_FORECAST_MODE, read_model_parameters
from .project_settings import CRU_VERSION, cru_csv_path, forecast_filename, validation_filename
from .monthly_features import (
    prepare_forecast_period,
    create_time_features,
    model_feature_columns,
    monthly_splits,
    select_model_rows,
)

warnings.filterwarnings('ignore')

FORECAST_COLUMNS = ['Date', 'WMO', 'Latitude', 'Longitude', 'Forecast', 'Fact', 'Model', 'Target_Var']
HISTORY_POLICIES = {
    'METEO_RU': 'complete_features_calendar_coverage',
    'CRU_CEDA': 'complete_features',
}
CRU_SOURCE_VERSION = f'CRU TS {CRU_VERSION}'
REGRESSORS = {
    'Random Forest': RandomForestRegressor,
    'K-Neighbors': KNeighborsRegressor,
    'MLP Neural Network': MLPRegressor,
}


def prepare_monthly_forecast_data(monthly, dataset_name):
    """Приводит исходные месячные ряды к рабочему периоду и календарю прогноза."""
    required_columns = ['Latitude', 'Longitude']
    if dataset_name == 'CRU_CEDA':
        required_columns.extend(['DTR', 'Precip'])
    history = monthly.loc[monthly['Year'] >= 1960].dropna(subset=required_columns)
    return history.assign(Date=pd.to_datetime(history[['Year', 'Month']].assign(day=1)))


def forecast_input_files(base_dir, dataset_name):
    if dataset_name == 'METEO_RU':
        meteo_dir = base_dir / 'meteo_ru'
        return meteo_dir / 'all_steppe_data.txt', meteo_dir / 'stations_coords.txt'
    if dataset_name == 'CRU_CEDA':
        return (cru_csv_path(base_dir),)
    raise ValueError(f'Неизвестный набор данных: {dataset_name}')


def load_forecast_data(base_dir, dataset_name):
    print(f'  Загрузка данных для {dataset_name}...')
    files = forecast_input_files(base_dir, dataset_name)
    for path in files:
        if not path.is_file():
            print(f'ОШИБКА: Не найден файл {path}')
            return pd.DataFrame()
    monthly = load_monthly_meteo(*files) if dataset_name == 'METEO_RU' else pd.read_csv(files[0])
    history = prepare_monthly_forecast_data(monthly, dataset_name)
    print(f'    Загружено {len(history)} записей.')
    return history


def build_regressor(model_class, model_name, params):
    model_args = {}
    if model_name in ('Random Forest', 'MLP Neural Network'):
        model_args['random_state'] = 42
    if model_name in ('Random Forest', 'K-Neighbors'):
        model_args['n_jobs'] = -1
    return model_class(**params, **model_args)


def prepare_regression_inputs(df_train, df_test, target_var, y_train=None):
    feature_cols = model_feature_columns(df_train)
    scaler = StandardScaler().fit(df_train[feature_cols])
    if y_train is None:
        y_train = df_train[target_var]
    if target_var == 'Precip':
        y_train = np.log1p(y_train)
    return scaler.transform(df_train[feature_cols]), scaler.transform(df_test[feature_cols]), y_train


def restore_regression_predictions(predictions, target_var):
    if target_var == 'Precip':
        predictions = np.expm1(predictions)
    return np.maximum(0, predictions)


def parameter_digest(params):
    serialized = json.dumps(params, sort_keys=True)
    return sha256(serialized.encode('utf-8')).hexdigest()


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


class BaseForecaster:
    forecast_mode = MONTHLY_FORECAST_MODE

    def __init__(self, base_dir, root_output_dir, subfolder_name):
        self.base_dir = Path(base_dir)
        params_path = Path(root_output_dir) / MODEL_PARAMS_FILENAME
        self.parameter_records = read_model_parameters(params_path)
        self.output_dir = Path(root_output_dir) / subfolder_name / 'csv_data'
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.model_classes = REGRESSORS.copy()
        print(f"[{self.__class__.__name__}] Инициализирован. Папка: {self.output_dir}")
        print(f"  Параметры моделей: {params_path if params_path.is_file() else 'BEST_PARAMS'}")

    def model_parameters(self, model_name, target_var, dataset_name, *, mode=None):
        records = self.parameter_records[mode or self.forecast_mode]
        return records[dataset_name][target_var][model_name]['parameters']

    def _parameter_digest(self, dataset_name, target_var):
        records = self.parameter_records[self.forecast_mode][dataset_name][target_var]
        return parameter_digest({name: record['parameters'] for name, record in records.items()})

    def _load_cached_predictions(self, filename, dataset_name, *, target_var=None):
        """Читает CSV, совместимый с настройками прогноза."""
        try:
            path = self.output_dir / filename
            if self.forecast_mode == ANNUAL_FORECAST_MODE:
                df = read_validation_csv(path, dataset_name)
            else:
                df = read_prediction_csv(path, dataset_name)
        except (OSError, ValueError):
            return None

        if (
            df.empty
            or not set(FORECAST_COLUMNS).issubset(df.columns)
            or 'History_Policy' not in df.columns
            or not df['History_Policy'].eq(HISTORY_POLICIES[dataset_name]).all()
        ):
            return None
        if target_var is None:
            target_var = df['Target_Var'].iat[0]
        if target_var not in BEST_PARAMS[dataset_name] or not df['Target_Var'].eq(target_var).all():
            return None

        expected = self._parameter_digest(dataset_name, target_var)
        if 'Model_Params_Hash' in df.columns:
            compatible = df['Model_Params_Hash'].eq(expected).all()
        else:
            # До появления JSON эти CSV строились на BEST_PARAMS.
            compatible = expected == parameter_digest(BEST_PARAMS[dataset_name][target_var])
        return df if compatible else None

    def _report_cached_metrics(self, filename, dataset_name, target_var, climatology):
        cached = self._load_cached_predictions(filename, dataset_name, target_var=target_var)
        if cached is None:
            return False
        print(f"  [SKIP] Файл {filename} уже существует и корректен.")
        if climatology is None:
            print("    Для сравнения с климатологией нужны исходные месячные ряды.")
        for model_name, rows in cached.groupby('Model', sort=False):
            print(f"    {model_name}")
            metrics = calculate_climatology_comparison(rows, climatology, target_var)
            print(format_forecast_metrics(metrics, padding='      '))
        return True

    def _save_predictions(self, predictions, filename, dataset_name):
        df = pd.concat(predictions) if predictions else pd.DataFrame(columns=FORECAST_COLUMNS)
        df['History_Policy'] = HISTORY_POLICIES[dataset_name]
        df['Model_Params_Hash'] = ''
        if not df.empty:
            df['Model_Params_Hash'] = self._parameter_digest(dataset_name, df['Target_Var'].iat[0])
        if self.forecast_mode == ANNUAL_FORECAST_MODE:
            df['Forecast_Mode'] = self.forecast_mode
        mark_prediction_source(df, dataset_name)
        with TemporaryDirectory(dir=self.output_dir) as temp_dir:
            temp_file = Path(temp_dir) / filename
            df.to_csv(temp_file, index=False)
            temp_file.replace(self.output_dir / filename)

    def _can_predict(self, df_train, df_test, model_name, target_var, dataset_name, *, mode=None):
        params = self.model_parameters(model_name, target_var, dataset_name, mode=mode)
        return not df_test.empty and len(df_train) >= params.get('n_neighbors', 1)

    @staticmethod
    def _print_coverage(candidates, eligible, predicted, paired):
        print(
            f"кандидатов: {candidates}, допущено: {eligible}, исключено: {candidates - eligible}, "
            f"прогнозов: {predicted}, пар с фактом: {paired}"
        )

    def load_data(self, dataset_name):
        return load_forecast_data(self.base_dir, dataset_name)

    def create_features(self, df):
        return create_time_features(df)

    def _fit_predict_model(self, df_train, df_test, model_name, target_var, dataset_name, *, y_train=None):
        params = self.model_parameters(model_name, target_var, dataset_name)
        model = build_regressor(self.model_classes[model_name], model_name, params)
        X_train, X_test, y_train = prepare_regression_inputs(df_train, df_test, target_var, y_train)
        model.fit(X_train, y_train)
        return restore_regression_predictions(model.predict(X_test), target_var)

    def annual_model_splits(self, features, targets, test_years, model_name, target_var, dataset_name):
        for train, test in annual_splits(features, targets, test_years):
            df_train, df_test = features.loc[train], features.loc[test]
            if self._can_predict(df_train, df_test, model_name, target_var, dataset_name, mode=ANNUAL_FORECAST_MODE):
                yield df_train, df_test, targets[train], targets[test]

# ============================================================================
# СТАНДАРТНЫЙ ПРОГНОЗ НА СЛЕДУЮЩИЙ МЕСЯЦ
# ============================================================================
class StandardForecaster(BaseForecaster):
    def __init__(self, base_dir, root_output_dir):
        super().__init__(base_dir, root_output_dir, "standard_forecast")
    
    def run(self, target_year, dataset_name):
        df_full = self.load_data(dataset_name)
        climatology = None if df_full.empty else station_month_climatology(df_full, [target_year])
        forecast_dates = pd.date_range(f'{target_year}-01-01', periods=12, freq='MS')
        eligible_forecasts = None

        for target_var in ['DTR', 'Precip']:
            fname = forecast_filename(dataset_name, target_var, target_year)

            if self._report_cached_metrics(fname, dataset_name, target_var, climatology):
                continue

            if df_full.empty:
                continue

            if eligible_forecasts is None:
                history, forecast_rows = prepare_forecast_period(df_full, forecast_dates)
                eligible_forecasts = select_model_rows(forecast_rows)

            print(f"\n>>> Стандартный прогноз: {target_var} на {target_year} ({dataset_name}) <<<")
            all_predictions = []
            train_rows = select_model_rows(history, target_var)
            
            for model_name in self.model_classes.keys():
                print(f"  Модель: {model_name}...", end=" ", flush=True)
                model_results = []
                
                # Walk-Forward по месяцам
                for forecast_date, df_train, df_pred in monthly_splits(train_rows, eligible_forecasts, forecast_dates):
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

                results = pd.concat(model_results) if model_results else pd.DataFrame(columns=FORECAST_COLUMNS)
                pairs = paired_predictions(results)
                self._print_coverage(len(forecast_rows), len(eligible_forecasts), len(results), len(pairs))
                if model_results:
                    all_predictions.append(results)
                else:
                    print("    Нет строк для прогноза.")

                metrics = calculate_climatology_comparison(results, climatology, target_var)
                print(format_forecast_metrics(metrics, padding='    '))
            
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

        climatology = station_month_climatology(df_full, test_years)
        features = None
        for target_var in ['DTR', 'Precip']:
            fname = validation_filename(dataset_name, target_var, split_year, final_year)
            if self._report_cached_metrics(fname, dataset_name, target_var, climatology):
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

                results = pd.concat(model_results) if model_results else pd.DataFrame(columns=FORECAST_COLUMNS)
                if model_results:
                    all_predictions.append(results)
                else:
                    print("      Нет строк для прогноза.")

                pairs = paired_predictions(results)
                print(f"      Месячных прогнозов: {len(results)}, пар с фактом: {len(pairs)}")
                metrics = calculate_climatology_comparison(results, climatology, target_var)
                print(format_forecast_metrics(metrics, padding='      '))

            self._save_predictions(all_predictions, fname, dataset_name)
            print(f"  -> Результаты валидации сохранены в {fname}")
