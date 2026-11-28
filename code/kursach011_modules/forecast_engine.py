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
import warnings

from .annual_forecast import (
    ANNUAL_FORECAST_MODE, annual_prediction_rows, annual_splits, annual_targets, prepare_annual_features,
)
from .forecast_metrics import (
    calculate_climatology_comparison, format_forecast_metrics, paired_predictions, station_month_climatology,
)
from .forecast_metadata import (
    check_forecast_metadata, input_file_state, save_forecast_result, sha256_file, training_split,
)
from .meteo_data import load_monthly_meteo
from .mlp_diagnostics import MlpDiagnostics
from .model_parameters import MODEL_PARAMS_FILENAME, MONTHLY_FORECAST_MODE, load_model_parameters
from .project_settings import CRU_VERSION, cru_csv_path, forecast_filename, mlp_diagnostics_dir, validation_filename
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


def load_forecast_input_snapshot(base_dir, dataset_name, load_data):
    paths = forecast_input_files(base_dir, dataset_name)
    initial_state = input_file_state(paths)
    files = {path.relative_to(base_dir).as_posix(): sha256_file(path) for path in paths}
    data = load_data(dataset_name)
    if input_file_state(paths) != initial_state:
        raise ValueError(f'{dataset_name}: входные файлы изменились при загрузке. Повторите запуск.')
    source = {'dataset': dataset_name, 'files_sha256': files}
    if dataset_name == 'CRU_CEDA':
        source['version'] = CRU_SOURCE_VERSION
    return data, source


def build_regressor(model_class, model_name, params, *, seed=42):
    model_args = {}
    if model_name in ('Random Forest', 'MLP Neural Network'):
        model_args['random_state'] = seed
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


def combine_file_checksums(checksums):
    digest = sha256()
    for checksum in checksums:
        digest.update(bytes.fromhex(checksum))
    return digest.hexdigest()


def files_digest(paths):
    return combine_file_checksums(sha256_file(path) for path in paths)


# Хеш фиксируется при импорте, чтобы соответствовать загруженному коду.
FORECAST_CODE_HASH = files_digest(
    Path(__file__).with_name(name) for name in (
        'forecast_engine.py', 'meteo_data.py', 'monthly_features.py', 'annual_forecast.py', 'mlp_diagnostics.py',
    )
)


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
        self.parameter_records, self.parameter_source = load_model_parameters(params_path)
        self.output_dir = Path(root_output_dir) / subfolder_name / 'csv_data'
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.diagnostics_dir = mlp_diagnostics_dir(self.base_dir) / subfolder_name
        self.mlp_diagnostics = MlpDiagnostics()
        self.model_classes = REGRESSORS.copy()
        print(f"[{self.__class__.__name__}] Инициализирован. Папка: {self.output_dir}")
        print(f"  Параметры моделей: {params_path if self.parameter_source == MODEL_PARAMS_FILENAME else self.parameter_source}")

    def model_parameters(self, model_name, target_var, dataset_name, *, mode=None):
        records = self.parameter_records[mode or self.forecast_mode]
        return records[dataset_name][target_var][model_name]['parameters']

    def _parameter_digest(self, dataset_name, target_var):
        records = self.parameter_records[self.forecast_mode][dataset_name][target_var]
        return parameter_digest({name: record['parameters'] for name, record in records.items()})

    def _load_cached_predictions(self, filename, dataset_name, target_var, hashes):
        """Читает совместимый CSV или сообщает причину пересчёта."""
        path = self.output_dir / filename
        if self.forecast_mode == ANNUAL_FORECAST_MODE:
            df = read_validation_csv(path, dataset_name)
        else:
            df = read_prediction_csv(path, dataset_name)

        if df.empty or not set(FORECAST_COLUMNS).issubset(df.columns):
            raise ValueError('CSV пуст или отсутствуют обязательные колонки')
        expected_values = (
            ('History_Policy', HISTORY_POLICIES[dataset_name], 'изменились правила отбора истории'),
            ('Target_Var', target_var, 'в CSV указан другой показатель'),
            ('Input_Data_Hash', hashes['Input_Data_Hash'], 'изменились входные данные'),
            ('Forecast_Code_Hash', hashes['Forecast_Code_Hash'], 'изменился код расчёта'),
            ('Model_Params_Hash', self._parameter_digest(dataset_name, target_var), 'изменились параметры моделей'),
        )
        for column, expected, reason in expected_values:
            if column not in df.columns:
                raise ValueError(f'в CSV отсутствует поле {column}')
            if not df[column].eq(expected).all():
                raise ValueError(reason)
        return df

    def _report_cached_metrics(self, filename, dataset_name, target_var, climatology, hashes, evaluation_period, *, force):
        if force:
            print(f'  [RUN] {filename}: задан --force.')
            return False
        path = self.output_dir / filename
        try:
            initial_state = input_file_state((path,))
            cached = self._load_cached_predictions(filename, dataset_name, target_var, hashes)
            check_forecast_metadata(path, evaluation_period)
            if input_file_state((path,)) != initial_state:
                raise ValueError('CSV изменился при чтении')
        except FileNotFoundError:
            print(f'  [RUN] {filename}: CSV отсутствует.')
            return False
        except (OSError, ValueError) as error:
            print(f'  [RUN] {filename}: {error}')
            return False
        print(f'  [SKIP] {filename}: используется актуальный CSV.')
        if climatology is None:
            print("    Для сравнения с климатологией нужны исходные месячные ряды.")
        for model_name, rows in cached.groupby('Model', sort=False):
            print(f"    {model_name}")
            metrics = calculate_climatology_comparison(rows, climatology, target_var)
            print(format_forecast_metrics(metrics, padding='      '))
        return True

    def _save_predictions(self, predictions, filename, dataset_name, hashes, *, source, target, evaluation_period, models):
        df = pd.concat(predictions) if predictions else pd.DataFrame(columns=FORECAST_COLUMNS)
        df = df.assign(**hashes)
        df['History_Policy'] = HISTORY_POLICIES[dataset_name]
        df['Model_Params_Hash'] = ''
        if not df.empty:
            df['Model_Params_Hash'] = self._parameter_digest(dataset_name, df['Target_Var'].iat[0])
        if self.forecast_mode == ANNUAL_FORECAST_MODE:
            df['Forecast_Mode'] = self.forecast_mode
        mark_prediction_source(df, dataset_name)
        save_forecast_result(
            self.output_dir / filename, df, source=source, mode=self.forecast_mode, target=target,
            evaluation_period=evaluation_period, models=models,
        )

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

    def _load_data_and_hashes(self, dataset_name):
        data, source = load_forecast_input_snapshot(self.base_dir, dataset_name, self.load_data)
        hashes = {
            'Input_Data_Hash': combine_file_checksums(source['files_sha256'].values()),
            'Forecast_Code_Hash': FORECAST_CODE_HASH,
        }
        return data, hashes, source

    def create_features(self, df):
        return create_time_features(df)

    def _fit_regressor(self, model, X_train, y_train, df_test, target_var, dataset_name, *, mode=None):
        date = df_test['Date'].iat[0] if isinstance(model, MLPRegressor) else None
        self.mlp_diagnostics.fit(
            model, X_train, y_train, date=date, mode=mode or self.forecast_mode, dataset=dataset_name, target=target_var,
        )

    def _save_mlp_diagnostics(self, filename):
        self.mlp_diagnostics.save(self.diagnostics_dir / Path(filename).with_suffix('.json'))
        self.mlp_diagnostics = MlpDiagnostics()

    def _build_model(self, model_name, target_var, dataset_name):
        params = self.model_parameters(model_name, target_var, dataset_name)
        return build_regressor(self.model_classes[model_name], model_name, params)

    def _model_metadata(self, model_name, target_var, dataset_name, training):
        model = self._build_model(model_name, target_var, dataset_name)
        return {
            'parameter_source': self.parameter_source,
            'parameters': model.get_params(deep=False),
            'training': training,
        }

    def _fit_predict_model(self, df_train, df_test, model_name, target_var, dataset_name, *, y_train=None):
        model = self._build_model(model_name, target_var, dataset_name)
        X_train, X_test, y_train = prepare_regression_inputs(df_train, df_test, target_var, y_train)
        self._fit_regressor(model, X_train, y_train, df_test, target_var, dataset_name)
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
    
    def run(self, target_year, dataset_name, *, target_month=None, force=False):
        self.mlp_diagnostics = MlpDiagnostics()
        df_full, hashes, source = self._load_data_and_hashes(dataset_name)
        climatology = None if df_full.empty else station_month_climatology(df_full, [target_year])
        period = str(target_year) if target_month is None else f'{target_year}-{target_month:02d}'
        forecast_dates = pd.date_range(period, periods=12 if target_month is None else 1, freq='MS')
        evaluation_period = (forecast_dates[0].strftime('%Y-%m'), forecast_dates[-1].strftime('%Y-%m'))
        eligible_forecasts = None

        for target_var in ['DTR', 'Precip']:
            fname = forecast_filename(dataset_name, target_var, target_year, month=target_month)

            if self._report_cached_metrics(
                fname, dataset_name, target_var, climatology, hashes, evaluation_period, force=force,
            ):
                continue

            if df_full.empty:
                raise ValueError(f'{dataset_name}: нет месячных рядов для пересчёта {fname}.')

            if eligible_forecasts is None:
                history, forecast_rows = prepare_forecast_period(df_full, forecast_dates)
                eligible_forecasts = select_model_rows(forecast_rows)

            print(f"\n>>> Стандартный прогноз: {target_var} на {period} ({dataset_name}) <<<")
            all_predictions = []
            models = {}
            train_rows = select_model_rows(history, target_var)
            
            for model_name in self.model_classes.keys():
                print(f"  Модель: {model_name}...", end=" ", flush=True)
                model_results = []
                training = []
                
                # Walk-Forward по месяцам
                for forecast_date, df_train, df_pred in monthly_splits(train_rows, eligible_forecasts, forecast_dates):
                    if not self._can_predict(df_train, df_pred, model_name, target_var, dataset_name):
                        continue
                    
                    preds = self._fit_predict_model(df_train, df_pred, model_name, target_var, dataset_name)
                    training.append(training_split(forecast_date, df_train['Date']))
                    
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
                    models[model_name] = self._model_metadata(model_name, target_var, dataset_name, training)
                else:
                    print("    Нет строк для прогноза.")

                metrics = calculate_climatology_comparison(results, climatology, target_var)
                print(format_forecast_metrics(metrics, padding='    '))
            
            self._save_mlp_diagnostics(fname)
            self._save_predictions(
                all_predictions, fname, dataset_name, hashes, source=source, target=target_var,
                evaluation_period=evaluation_period, models=models,
            )
            print(f"  -> Сохранено: {fname}")

# ============================================================================
# ДОЛГОСРОЧНАЯ ВАЛИДАЦИЯ ПРЯМОГО ПРОГНОЗА НА 12 МЕСЯЦЕВ
# ============================================================================
class LongTermValidator(BaseForecaster):
    forecast_mode = ANNUAL_FORECAST_MODE

    def __init__(self, base_dir, root_output_dir):
        super().__init__(base_dir, root_output_dir, "validation_results")

    def run(self, split_year, dataset_name, end_year=None, *, force=False):
        self.mlp_diagnostics = MlpDiagnostics()
        df_full, hashes, source = self._load_data_and_hashes(dataset_name)
        if df_full.empty:
            print(f'  {dataset_name}: нет месячных рядов для валидации.')
            return
        final_year = min(end_year, df_full['Year'].max()) if end_year is not None else df_full['Year'].max()
        test_years = range(split_year, final_year + 1)
        if not test_years:
            print(f'  {dataset_name}: нет данных за запрошенный период валидации.')
            return

        climatology = station_month_climatology(df_full, test_years)
        evaluation_period = (f'{split_year}-01', f'{end_year}-12' if end_year is not None else None)
        features = None
        for target_var in ['DTR', 'Precip']:
            fname = validation_filename(dataset_name, target_var, split_year, final_year)
            if self._report_cached_metrics(
                fname, dataset_name, target_var, climatology, hashes, evaluation_period, force=force,
            ):
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
            models = {}

            for model_name in self.model_classes:
                print(f"    {model_name}:")
                print(f"      Пар станция-год до отбора: {candidate_count}, после отбора: {eligible_count}")
                model_results = []
                training = []
                splits = self.annual_model_splits(features, targets, test_years, model_name, target_var, dataset_name)
                for df_train, df_test, y_train, facts in splits:
                    predictions = self._fit_predict_model(
                        df_train, df_test, model_name, target_var, dataset_name, y_train=y_train,
                    )
                    training.append(training_split(df_test['Date'].iat[0], df_train['Date'], horizon_months=12))
                    model_results.append(annual_prediction_rows(df_test, predictions, facts, model_name, target_var))

                results = pd.concat(model_results) if model_results else pd.DataFrame(columns=FORECAST_COLUMNS)
                if model_results:
                    all_predictions.append(results)
                    models[model_name] = self._model_metadata(model_name, target_var, dataset_name, training)
                else:
                    print("      Нет строк для прогноза.")

                pairs = paired_predictions(results)
                print(f"      Месячных прогнозов: {len(results)}, пар с фактом: {len(pairs)}")
                metrics = calculate_climatology_comparison(results, climatology, target_var)
                print(format_forecast_metrics(metrics, padding='      '))

            self._save_mlp_diagnostics(fname)
            self._save_predictions(
                all_predictions, fname, dataset_name, hashes, source=source, target=target_var,
                evaluation_period=evaluation_period, models=models,
            )
            print(f"  -> Результаты валидации сохранены в {fname}")
        return final_year
