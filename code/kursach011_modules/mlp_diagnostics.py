"""Диагностика обучения MLP для прогноза, отчёта и подбора параметров."""

import json
from math import isfinite
from pathlib import Path
from tempfile import TemporaryDirectory
import warnings

from sklearn.exceptions import ConvergenceWarning
from sklearn.neural_network import MLPRegressor


class MlpDiagnostics:
    def __init__(self):
        self.fits = []

    def fit(self, model, X, y, *, date, mode, dataset, target):
        if not isinstance(model, MLPRegressor):
            model.fit(X, y)
            return

        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter('always', ConvergenceWarning)
            model.fit(X, y)

        convergence_warnings = []
        for warning in caught:
            if issubclass(warning.category, ConvergenceWarning):
                convergence_warnings.append(str(warning.message))
            else:
                warnings.warn_explicit(warning.message, warning.category, warning.filename, warning.lineno)

        loss_curve = getattr(model, 'loss_curve_', None)
        self.fits.append({
            'release': date.strftime('%Y-%m-%d'),
            'mode': mode,
            'dataset': dataset,
            'target': target,
            'seed': model.random_state,
            'max_iter': model.max_iter,
            'n_iter_': int(model.n_iter_),
            'reached_limit': bool(model.n_iter_ >= model.max_iter),
            'convergence_warnings': convergence_warnings,
            'loss_curve_': None if loss_curve is None else [
                float(loss) if isfinite(loss) else None for loss in loss_curve
            ],
            'parameters': model.get_params(),
        })

    def save(self, path):
        if not self.fits:
            return

        total = len(self.fits)
        limits = sum(fit['reached_limit'] for fit in self.fits)
        warnings_count = sum(bool(fit['convergence_warnings']) for fit in self.fits)
        summary = {'fits': total, 'reached_limit': limits, 'with_convergence_warning': warnings_count}
        content = json.dumps({'summary': summary, 'fits': self.fits}, ensure_ascii=False, indent=2, allow_nan=False)
        path.parent.mkdir(parents=True, exist_ok=True)
        with TemporaryDirectory(dir=path.parent) as temp_dir:
            temporary_path = Path(temp_dir) / path.name
            temporary_path.write_text(content + '\n', encoding='utf-8')
            temporary_path.replace(path)
        print(f'  MLP достигла лимита в {limits} из {total} обучений, предупреждений в {warnings_count} обучениях.')
        print(f'  Диагностика MLP сохранена в {path}')
