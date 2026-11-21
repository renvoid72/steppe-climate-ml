# -*- coding: utf-8 -*-
"""Агрегация суточных осадков в месячные суммы.

Отдельный модуль без тяжёлых зависимостей, чтобы правило было проверяемым
тестами: `kursach011_analytics` тянет xarray, shapely и sklearn, и импорт
ради одной функции там не оправдан.
"""

import numpy as np
import pandas as pd


def sum_with_gaps(series: pd.Series) -> float:
    """Сумма суточных значений, которая не превращает пропуск в ноль.

    Пропуск в сутках означает, что месяц неполный: сумма по неполному месяцу
    несопоставима с полным, поэтому результат — NaN. Группа из одних нулей
    даёт честный 0.0.

    >>> sum_with_gaps(pd.Series([0.0, 0.0]))
    0.0
    >>> sum_with_gaps(pd.Series([np.nan]))
    nan
    """
    values = pd.to_numeric(series, errors="coerce")
    if values.isna().any():
        return float("nan")
    return float(values.sum())


__all__ = ["sum_with_gaps"]