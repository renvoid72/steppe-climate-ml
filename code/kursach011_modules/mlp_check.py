"""Короткая проверка лимита эпох MLP и разброса RF и MLP по seed."""

import argparse
from pathlib import Path
import sys

import numpy as np
import pandas as pd

if not __package__:
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
    __package__ = 'kursach011_modules'

from .annual_forecast import ANNUAL_FORECAST_MODE
from .forecast_engine import (
    REGRESSORS, build_regressor, load_forecast_data, parameter_digest, prepare_regression_inputs,
    restore_regression_predictions,
)
from .forecast_metrics import calculate_forecast_metrics
from .mlp_diagnostics import MlpDiagnostics
from .model_parameters import FORECAST_MODES, MODEL_PARAMS_FILENAME, MONTHLY_FORECAST_MODE, read_model_parameters
from .project_settings import results_dir
from .tuning_lab import prepare_tuning_splits

BASE_DIR = Path(__file__).resolve().parents[2]
MLP = 'MLP Neural Network'
RF = 'Random Forest'
SEEDS = (42, 43, 44)
EPOCH_LIMITS = (200, 400)
ANNUAL_RELEASES = ('1988-01-01', '1989-01-01')
CASES = (
    (ANNUAL_FORECAST_MODE, 'CRU_CEDA', 'DTR', ANNUAL_RELEASES),
    (ANNUAL_FORECAST_MODE, 'METEO_RU', 'Precip', ANNUAL_RELEASES),
    (MONTHLY_FORECAST_MODE, 'CRU_CEDA', 'DTR', ('2023-01-01',)),
)


def compare_release(split, *, mode, dataset, target, active_params, diagnostics):
    train, test, train_targets, facts, dates = split
    release = test['Date'].iat[0]
    observed = np.isfinite(facts.ravel())
    if train.empty or not observed.any():
        raise ValueError(f'{dataset} {target} {release.date()} не хватает данных для сравнения.')

    X_train, X_test, y_train = prepare_regression_inputs(train, test, target, train_targets)
    pairs = pd.DataFrame({'Date': dates[observed], 'Fact': facts.ravel()[observed]})
    trials = [(MLP, f'epochs_{limit}', SEEDS[0], {
        **active_params[MLP]['parameters'], 'max_iter': limit, 'early_stopping': False,
    }) for limit in EPOCH_LIMITS]
    if mode == ANNUAL_FORECAST_MODE:
        trials.extend(
            (model, 'active', seed, active_params[model]['parameters'])
            for model in (MLP, RF) for seed in SEEDS
        )
    else:
        trials.append((MLP, 'active', SEEDS[0], active_params[MLP]['parameters']))

    results = []
    scored = {}
    for name, variant, seed, params in trials:
        model = build_regressor(REGRESSORS[name], name, params, seed=seed)
        key = (name, parameter_digest(model.get_params()))
        if key not in scored:
            print(f'  {dataset} {mode} {target} {release.date()} | {name} | {variant} | seed={seed}', flush=True)
            diagnostics.fit(model, X_train, y_train, date=release, mode=mode, dataset=dataset, target=target)
            forecast = restore_regression_predictions(model.predict(X_test), target).ravel()[observed]
            if not np.isfinite(forecast).all():
                raise ValueError(f'{dataset} {target} {release.date()} {name} выдал нечисловой прогноз.')
            scored[key] = calculate_forecast_metrics(pairs.assign(Forecast=forecast), target)
        regional, stations = scored[key]
        results.append({
            'mode': mode, 'dataset': dataset, 'target': target, 'release': release.strftime('%Y-%m-%d'),
            'model': name, 'variant': variant, 'seed': seed,
            'n_months': regional['N'], 'n_station_months': stations['N'],
            'regional_mae': regional['MAE'], 'station_mae': stations['MAE'], 'unit': stations['Ед. ошибки'],
        })
    return results


def compare_releases(monthly, *, mode, dataset, target, releases, active_params, diagnostics):
    expected = set(pd.DatetimeIndex(releases))
    years = tuple(sorted({date.year for date in expected}))
    results = []
    for split in prepare_tuning_splits(monthly, mode, years)(target):
        test = split[1]
        if test.empty or test['Date'].iat[0] not in expected:
            continue
        results.extend(compare_release(
            split, mode=mode, dataset=dataset, target=target, active_params=active_params, diagnostics=diagnostics,
        ))
        expected.remove(test['Date'].iat[0])
    if expected:
        dates = ', '.join(str(date.date()) for date in sorted(expected))
        raise ValueError(f'{dataset} {target} нет выпусков с полными признаками за {dates}.')
    return results


def summarize_seeds(comparison):
    active = comparison.loc[comparison['variant'].eq('active') & comparison['mode'].eq(ANNUAL_FORECAST_MODE)]
    keys = ['mode', 'dataset', 'target', 'release', 'model', 'unit', 'n_months', 'n_station_months']
    errors = active.melt(
        id_vars=keys, value_vars=['regional_mae', 'station_mae'], var_name='metric', value_name='mae',
    )
    return errors.groupby([*keys, 'metric'], as_index=False)['mae'].agg(
        num_seeds='size', mean='mean', std='std', min='min', max='max',
    )


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--mode', choices=('both', *FORECAST_MODES), default='both')
    parser.add_argument('--output-dir', type=Path)
    args = parser.parse_args()

    root = results_dir(BASE_DIR)
    output = args.output_dir or root / 'mlp_diagnostics' / 'short_check' / args.mode
    active = read_model_parameters(root / MODEL_PARAMS_FILENAME)
    diagnostics = MlpDiagnostics()
    sources = {}
    results = []
    for mode, dataset, target, releases in CASES:
        if args.mode not in ('both', mode):
            continue
        if dataset not in sources:
            sources[dataset] = load_forecast_data(BASE_DIR, dataset)
        if sources[dataset].empty:
            parser.exit(1, f'Нет рабочих данных для {dataset}. Проверьте подготовку данных по README.\n')
        results.extend(compare_releases(
            sources[dataset], mode=mode, dataset=dataset, target=target, releases=releases,
            active_params=active[mode][dataset][target], diagnostics=diagnostics,
        ))

    diagnostics.save(output / 'mlp_fits.json')
    comparison = pd.DataFrame(results)
    comparison.to_csv(output / 'comparison.csv', index=False)
    summarize_seeds(comparison).to_csv(output / 'seed_summary.csv', index=False)
    print(f'  Результаты короткой проверки сохранены в {output}')


if __name__ == '__main__':
    main()
