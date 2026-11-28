"""Сведения о расчёте и их привязка к прогнозному CSV."""

import json
from datetime import datetime, timezone
from hashlib import file_digest
from pathlib import Path
from shutil import copyfile
from tempfile import TemporaryDirectory

import pandas as pd

from .forecast_metrics import paired_predictions


def sha256_file(path):
    with path.open('rb') as source:
        return file_digest(source, 'sha256').hexdigest()


def input_file_state(paths):
    states = [path.stat() for path in paths]
    return [(state.st_ino, state.st_size, state.st_mtime_ns) for state in states]


def training_split(release_date, train_dates, *, horizon_months=1):
    last_month = pd.Timestamp(train_dates.max()) + pd.offsets.MonthBegin(horizon_months - 1)
    return {
        'release_month': pd.Timestamp(release_date).strftime('%Y-%m'),
        'target_start_month': pd.Timestamp(train_dates.min()).strftime('%Y-%m'),
        'target_end_month': last_month.strftime('%Y-%m'),
        'examples': len(train_dates),
    }


def save_forecast_result(csv_path, rows, *, source, mode, target, evaluation_period, models):
    start, end = evaluation_period
    evaluated_months = {}
    for model_name, predictions in rows.groupby('Model', sort=False):
        pairs = paired_predictions(predictions)
        evaluated_months[model_name] = sorted(pairs['Date'].dt.strftime('%Y-%m').unique())
    metadata = {
        'calculated_at': datetime.now(timezone.utc).isoformat(),
        'source': source,
        'forecast_mode': mode,
        'target': target,
        'evaluation': {
            'requested_start_month': start,
            'requested_end_month': end,
            'months_with_pairs_by_model': {name: evaluated_months.get(name, []) for name in models},
        },
        'models': models,
    }
    json_path = csv_path.with_suffix('.json')
    with TemporaryDirectory(dir=csv_path.parent) as directory:
        staging = Path(directory)
        csv = staging / csv_path.name
        sidecar = staging / json_path.name
        rows.to_csv(csv, index=False)
        metadata['csv_sha256'] = sha256_file(csv)
        sidecar.write_text(json.dumps(metadata, ensure_ascii=False, indent=2, allow_nan=False) + '\n', encoding='utf-8')

        # Два файла заменяются по очереди. Блокировка защищает от смешения двух записей,
        # а контрольная сумма позволяет распознать незавершённую запись при чтении.
        lock = csv_path.with_suffix('.lock')
        try:
            lock.mkdir()
        except FileExistsError as exc:
            raise ValueError(f'{csv_path}: другой запуск сохраняет прогноз.') from exc
        try:
            backup = staging / 'previous.json'
            if json_path.exists():
                copyfile(json_path, backup)
            sidecar.replace(json_path)
            try:
                csv.replace(csv_path)
            except OSError:
                if backup.exists():
                    backup.replace(json_path)
                else:
                    json_path.unlink()
                raise
        finally:
            lock.rmdir()


def check_forecast_metadata(csv_path, evaluation_period):
    json_path = csv_path.with_suffix('.json')
    try:
        metadata = json.loads(json_path.read_text(encoding='utf-8'))
        evaluation = metadata['evaluation']
        saved_period = (evaluation['requested_start_month'], evaluation['requested_end_month'])
        expected_hash = metadata['csv_sha256']
    except FileNotFoundError as exc:
        raise ValueError('отсутствует JSON со сведениями о расчёте') from exc
    except (ValueError, KeyError, TypeError) as exc:
        raise ValueError('не удалось прочитать JSON со сведениями о расчёте') from exc
    if saved_period != evaluation_period:
        raise ValueError('изменился запрошенный период оценки')
    if sha256_file(csv_path) != expected_hash:
        raise ValueError('контрольная сумма CSV не совпадает со сведениями о расчёте')
