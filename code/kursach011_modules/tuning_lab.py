"""Подбор параметров помесячного и прямого годового прогноза."""

import argparse
from functools import partial
from hashlib import file_digest
from pathlib import Path
import sys

import numpy as np
import pandas as pd
from sklearn.model_selection import ParameterSampler

if not __package__:
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
    __package__ = 'kursach011_modules'

from .annual_forecast import ANNUAL_FORECAST_MODE, annual_splits, annual_targets, prepare_annual_features
from .forecast_engine import (
    CRU_SOURCE_VERSION, REGRESSORS, build_regressor, forecast_input_files, load_forecast_data, parameter_digest,
    prepare_regression_inputs, restore_regression_predictions,
)
from .model_parameters import (
    BEST_PARAMS, FORECAST_MODES, MODEL_PARAMS_FILENAME, MONTHLY_FORECAST_MODE,
    read_model_parameters, save_model_parameters,
)
from .monthly_features import monthly_splits, prepare_forecast_period, select_model_rows
from .project_settings import results_dir

BASE_DIR = Path(__file__).resolve().parents[2]
TUNING_YEARS = {
    MONTHLY_FORECAST_MODE: (2022, 2023),
    ANNUAL_FORECAST_MODE: tuple(range(1980, 1990)),
}
N_ITER = 10
PARAM_GRIDS = {
    'Random Forest': {
        'n_estimators': [100, 150, 200],
        'max_depth': [7, 10, 15, 20],
        'min_samples_leaf': [5, 10, 15, 20],
        'max_features': ['sqrt', 1.0],
    },
    'K-Neighbors': {
        'n_neighbors': [15, 25, 40, 50],
        'weights': ['uniform', 'distance'],
    },
    'MLP Neural Network': {
        'hidden_layer_sizes': [(50,), (100,), (50, 25)],
        'alpha': [0.1, 1, 5],
        'activation': ['tanh'],
        'early_stopping': [True],
    },
}


def parameter_candidates(model_name, active_params):
    sampled = ParameterSampler(PARAM_GRIDS[model_name], n_iter=N_ITER, random_state=42)
    active_digest = parameter_digest(active_params)
    return [active_params, *(params for params in sampled if parameter_digest(params) != active_digest)]


def monthly_tuning_splits(history, forecasts, monthly, dates, target_var):
    train_rows = select_model_rows(history, target_var)
    facts = monthly.set_index(['Date', 'WMO'])[target_var]
    for date, train, test in monthly_splits(train_rows, forecasts, dates):
        actual = facts.reindex(pd.MultiIndex.from_frame(test[['Date', 'WMO']]))
        yield train, test, None, actual.to_numpy(), np.repeat(date.to_datetime64(), len(test))


def annual_tuning_splits(features, monthly, years, target_var):
    targets = annual_targets(monthly, features, target_var)
    for train, test in annual_splits(features, targets, years):
        forecast = features.loc[test]
        if forecast.empty:
            continue
        months = pd.date_range(forecast['Date'].iat[0], periods=targets.shape[1], freq='MS')
        yield features.loc[train], forecast, targets[train], targets[test], np.tile(months.to_numpy(), len(forecast))


def prepare_tuning_splits(monthly, mode, years):
    """Готовит признаки один раз для обоих показателей и всех кандидатов."""
    if mode not in FORECAST_MODES:
        raise ValueError(f'Неизвестный режим подбора: {mode}')
    monthly = monthly.loc[monthly['Year'] <= max(years)]
    if monthly.empty:
        raise ValueError(f'Нет данных для подбора до {max(years)} года.')

    if mode == ANNUAL_FORECAST_MODE:
        features = select_model_rows(prepare_annual_features(monthly, max(years)))
        return partial(annual_tuning_splits, features, monthly, years)

    dates = pd.DatetimeIndex([pd.Timestamp(year, month, 1) for year in years for month in range(1, 13)])
    history, forecasts = prepare_forecast_period(monthly, dates)
    return partial(monthly_tuning_splits, history, select_model_rows(forecasts), monthly, dates)


def select_parameters(splits, target_var, model_name, candidates):
    """Сравнивает региональные прогнозы кандидатов на общей выборке."""
    model_class = REGRESSORS[model_name]
    min_train_size = max(params.get('n_neighbors', 1) for params in candidates)
    results = []
    for train, test, train_targets, facts, dates in splits:
        observed = np.isfinite(facts.ravel())
        if len(train) < min_train_size or not observed.any():
            continue
        X_train, X_test, train_targets = prepare_regression_inputs(train, test, target_var, train_targets)
        forecasts = {'Date': dates[observed], 'Fact': facts.ravel()[observed]}
        for index, params in enumerate(candidates):
            model = build_regressor(model_class, model_name, params)
            model.fit(X_train, train_targets)
            forecasts[index] = restore_regression_predictions(model.predict(X_test), target_var).ravel()[observed]
        results.append(pd.DataFrame(forecasts))

    if not results:
        raise ValueError(f'{model_name}, {target_var}: нет общей выборки для оценки кандидатов.')
    paired = pd.concat(results, ignore_index=True)
    candidate_columns = list(range(len(candidates)))
    if not np.isfinite(paired[candidate_columns].to_numpy()).all():
        raise ValueError(f'{model_name}, {target_var}: получен нечисловой или бесконечный прогноз.')
    by_month = paired.groupby('Date').mean()
    mae = by_month[candidate_columns].sub(by_month['Fact'], axis=0).abs().mean()
    winner = mae.idxmin()
    return {
        'parameters': candidates[winner],
        'selection': 'tuning',
        'metric': 'regional_monthly_mae',
        'score': float(mae[winner]),
        'n_months': len(by_month),
        'n_station_months': len(paired),
    }


def tuning_data_source(base_dir, dataset_name):
    if dataset_name == 'CRU_CEDA':
        return {'version': CRU_SOURCE_VERSION}
    checksums = {}
    for path in forecast_input_files(base_dir, dataset_name):
        with path.open('rb') as source:
            checksums[path.name] = file_digest(source, 'sha256').hexdigest()
    return {'sha256': checksums}


def tune_dataset(monthly, dataset_name, source, mode, years, active_records):
    splits = prepare_tuning_splits(monthly, mode, years)
    results = {}
    for target_var, models in BEST_PARAMS[dataset_name].items():
        for model_name in models:
            print(f'{dataset_name} | {mode} | {target_var} | {model_name}', flush=True)
            candidates = parameter_candidates(model_name, active_records[target_var][model_name]['parameters'])
            result = select_parameters(splits(target_var), target_var, model_name, candidates)
            result.update(years=list(years), data_source=source)
            results[mode, dataset_name, target_var, model_name] = result
            print(f"  MAE={result['score']:.4f}, месяцев={result['n_months']}, пар={result['n_station_months']}")
            print(f"  {result['parameters']}")
    return results


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--mode', choices=('both', *FORECAST_MODES), default='both')
    args = parser.parse_args()
    modes = FORECAST_MODES if args.mode == 'both' else (args.mode,)
    path = results_dir(BASE_DIR) / MODEL_PARAMS_FILENAME
    active_records = read_model_parameters(path)
    updates = {}
    for dataset_name in BEST_PARAMS:
        monthly = load_forecast_data(BASE_DIR, dataset_name)
        if monthly.empty:
            parser.exit(1, f'Нет рабочих данных для {dataset_name}. Проверьте подготовку данных по README.\n')
        source = tuning_data_source(BASE_DIR, dataset_name)
        for mode in modes:
            years = TUNING_YEARS[mode]
            print(f'Подбор {mode} за {years[0]}–{years[-1]} гг.')
            updates.update(tune_dataset(monthly, dataset_name, source, mode, years, active_records[mode][dataset_name]))

    save_model_parameters(path, updates)
    print(f'Параметры сохранены в {path}')


if __name__ == '__main__':
    main()
