# -*- coding: utf-8 -*-
"""Тесты правила «пропуск осадков не превращается в ноль»."""

import unittest

import numpy as np
import pandas as pd

from kursach011_modules.aggregation import sum_with_gaps


class SumWithGapsTest(unittest.TestCase):
    def test_all_zero_days_are_a_real_zero(self):
        self.assertEqual(sum_with_gaps(pd.Series([0.0, 0.0, 0.0])), 0.0)

    def test_single_missing_day_gives_nan(self):
        self.assertTrue(np.isnan(sum_with_gaps(pd.Series([np.nan]))))

    def test_zero_and_missing_gives_nan(self):
        self.assertTrue(np.isnan(sum_with_gaps(pd.Series([0.0, np.nan]))))

    def test_normal_month_is_summed(self):
        self.assertAlmostEqual(sum_with_gaps(pd.Series([0.4, 1.2, 0.0, 3.4])), 5.0, places=6)

    def test_integer_and_float_mix(self):
        self.assertAlmostEqual(sum_with_gaps(pd.Series([1, 2.5])), 3.5, places=6)

    def test_object_dtype_with_empty_string_is_a_gap(self):
        self.assertTrue(np.isnan(sum_with_gaps(pd.Series(["1.0", "", "2.0"]))))


if __name__ == "__main__":
    unittest.main()