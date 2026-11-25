"""Фиксированные настройки и результаты подбора моделей."""

import json
from pathlib import Path
from tempfile import TemporaryDirectory

from .annual_forecast import ANNUAL_FORECAST_MODE

MONTHLY_FORECAST_MODE = 'monthly'
FORECAST_MODES = (MONTHLY_FORECAST_MODE, ANNUAL_FORECAST_MODE)
MODEL_PARAMS_FILENAME = 'model_params.json'

BEST_PARAMS = {
    'METEO_RU': {
        'DTR': {
            'Random Forest': {'n_estimators': 150, 'max_depth': 15, 'min_samples_leaf': 10},
            'K-Neighbors': {'n_neighbors': 25, 'weights': 'uniform'},
            'MLP Neural Network': {'hidden_layer_sizes': (100,), 'alpha': 0.1, 'activation': 'tanh'},
        },
        'Precip': {
            'Random Forest': {'n_estimators': 150, 'max_depth': 10, 'min_samples_leaf': 15},
            'K-Neighbors': {'n_neighbors': 25, 'weights': 'distance'},
            'MLP Neural Network': {'hidden_layer_sizes': (50, 25), 'alpha': 1, 'activation': 'tanh'},
        },
    },
    'CRU_CEDA': {
        'DTR': {
            'Random Forest': {'n_estimators': 200, 'max_depth': 20, 'min_samples_leaf': 10},
            'K-Neighbors': {'n_neighbors': 15, 'weights': 'uniform'},
            'MLP Neural Network': {'hidden_layer_sizes': (100,), 'alpha': 0.1, 'activation': 'tanh'},
        },
        'Precip': {
            'Random Forest': {'n_estimators': 200, 'max_depth': 10, 'min_samples_leaf': 10},
            'K-Neighbors': {'n_neighbors': 15, 'weights': 'distance'},
            'MLP Neural Network': {'hidden_layer_sizes': (50, 25), 'alpha': 1, 'activation': 'tanh'},
        },
    },
}


def default_parameter_records():
    return {
        mode: {
            dataset: {
                target: {
                    model: {'parameters': params.copy(), 'selection': 'BEST_PARAMS'}
                    for model, params in models.items()
                }
                for target, models in targets.items()
            }
            for dataset, targets in BEST_PARAMS.items()
        }
        for mode in FORECAST_MODES
    }


def read_model_parameters(path):
    """Загружает конфигурацию целиком. При отсутствии файла использует BEST_PARAMS."""
    try:
        content = path.read_text(encoding='utf-8')
    except FileNotFoundError:
        return default_parameter_records()

    try:
        records = json.loads(content)
        for mode in FORECAST_MODES:
            for dataset, targets in BEST_PARAMS.items():
                for target, models in targets.items():
                    for model in models:
                        params = records[mode][dataset][target][model]['parameters']
                        if not isinstance(params, dict):
                            raise TypeError(f'{mode}/{dataset}/{target}/{model}: parameters')
    except (KeyError, TypeError, ValueError) as exc:
        raise ValueError(f'{path}: не удалось прочитать параметры моделей ({exc}).') from exc
    return records


def save_model_parameters(path, updates):
    """Обновляет выбранные настройки, сохраняя остальную конфигурацию."""
    path.parent.mkdir(parents=True, exist_ok=True)
    lock_path = path.with_suffix('.lock')
    try:
        lock_path.mkdir()
    except FileExistsError as exc:
        raise ValueError(f'{path}: другой запуск записывает параметры. Повторите попытку позже.') from exc
    try:
        records = read_model_parameters(path)
        for (mode, dataset, target, model), record in updates.items():
            records[mode][dataset][target][model] = record
        content = json.dumps(records, ensure_ascii=False, indent=2, allow_nan=False)
        with TemporaryDirectory(dir=path.parent) as temp_dir:
            temp_path = Path(temp_dir) / path.name
            temp_path.write_text(content + '\n', encoding='utf-8')
            temp_path.replace(path)
    finally:
        lock_path.rmdir()
