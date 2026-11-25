# -*- coding: utf-8 -*-
"""
Автономный скрипт для расширенного хронометрического и аналитического отчета.
Выполняет обучение и прогнозирование с помощью моделей Random Forest, K-Neighbors и MLP,
замеряет точное время обучения, прогноза, скорость обработки данных и точность (R2, MAE, RMSE).

Метрики сохраняются в forecast_report.txt, диагностика MLP — в отдельных JSON.
"""

import sys
import os
import time
from datetime import datetime
from pathlib import Path
import pandas as pd
import warnings
from kursach011_modules.project_settings import (
    ANALYSIS_YEAR, VALIDATION_START_YEAR, VALIDATION_END_YEAR, forecast_filename, results_dir, validation_filename,
)

warnings.filterwarnings('ignore')

# --- КОНСТАНТЫ ПУТЕЙ ---
PROJECT_DIR = Path(__file__).resolve().parent.parent
RESULTS_DIR = results_dir(PROJECT_DIR)
REPORT_FILE = RESULTS_DIR / "forecast_report.txt"

# Добавляем путь к модулям проекта
sys.path.append(str(PROJECT_DIR / "code"))

from kursach011_modules.forecast_engine import (
    BaseForecaster, FORECAST_COLUMNS, build_regressor, prepare_regression_inputs, restore_regression_predictions,
)
from kursach011_modules.model_parameters import MONTHLY_FORECAST_MODE
from kursach011_modules.annual_forecast import (
    ANNUAL_FORECAST_MODE, annual_prediction_rows, annual_targets, prepare_annual_features, print_monthly_errors,
)
from kursach011_modules.forecast_metrics import (
    calculate_climatology_comparison, format_forecast_metrics, paired_predictions, station_month_climatology,
)
from kursach011_modules.monthly_features import monthly_splits, prepare_forecast_period, select_model_rows


class DualLogger:
    """Класс для дублирования всех print-выводов одновременно в консоль и в файл."""
    def __init__(self, filepath):
        self.filepath = filepath
        self.filepath.parent.mkdir(parents=True, exist_ok=True)
        self.file = open(filepath, 'w', encoding='utf-8')
        self.terminal = sys.stdout

    def write(self, message):
        self.terminal.write(message)
        self.file.write(message)

    def flush(self):
        self.terminal.flush()
        self.file.flush()

    def close(self):
        if not self.file.closed:
            self.file.close()


class ForecastReporter(BaseForecaster):
    """
    Класс-отчетник, наследующий логику подготовки данных и признаков из BaseForecaster,
    но производящий замеры времени, скорости и точности без сохранения CSV-файлов.
    """
    def __init__(self, base_dir, root_output_dir):
        super().__init__(base_dir, root_output_dir, "temp_report")
        self.diagnostics_dir = Path(root_output_dir) / 'mlp_diagnostics' / 'forecast_report'
        self.summary_table = []
        self.accuracy_table = []

    def _timed_fit_predict(
        self, df_train, df_test, model_name, target_var, dataset_name, *, y_train=None, mode=MONTHLY_FORECAST_MODE,
    ):
        """Использует рабочую конфигурацию и раздельно измеряет fit() и predict()."""
        params = self.model_parameters(model_name, target_var, dataset_name, mode=mode)
        model = build_regressor(self.model_classes[model_name], model_name, params)
        X_train_scaled, X_test_scaled, y_train = prepare_regression_inputs(df_train, df_test, target_var, y_train)

        # 1. Замер времени обучения (Fit Time)
        t_fit_start = time.perf_counter()
        self._fit_regressor(model, X_train_scaled, y_train, df_test, target_var, dataset_name, mode=mode)
        t_fit_end = time.perf_counter()
        fit_time = t_fit_end - t_fit_start

        # 2. Замер времени прогноза (Predict Time)
        t_pred_start = time.perf_counter()
        raw_preds = model.predict(X_test_scaled)
        t_pred_end = time.perf_counter()
        predict_time = t_pred_end - t_pred_start

        final_preds = restore_regression_predictions(raw_preds, target_var)

        return final_preds, fit_time, predict_time, final_preds.size

    def run_standard_forecast_report(self, target_year, dataset_name):
        """Помесячная оценка с горизонтом один месяц и ежемесячным обучением."""
        self.mlp_diagnostics.fits.clear()
        print("\n" + "="*80)
        print(f" БЛОК 1: ПРОГНОЗ НА МЕСЯЦ ({dataset_name} | Период оценки: {target_year})")
        print(" Переобучение ежемесячно")
        print("="*80)

        t_load_start = time.perf_counter()
        df_full = self.load_data(dataset_name)
        t_load_end = time.perf_counter()
        print(f"  [Время загрузки данных и базовой предобработки: {t_load_end - t_load_start:.3f} сек]")

        if df_full.empty:
            print("  ОШИБКА: Данные не загружены.")
            return

        forecast_dates = pd.date_range(f'{target_year}-01-01', periods=12, freq='MS')
        history, forecast_rows = prepare_forecast_period(df_full, forecast_dates)
        eligible_forecasts = select_model_rows(forecast_rows)
        climatology = station_month_climatology(df_full, [target_year])
        for target_var in ['DTR', 'Precip']:
            train_rows = select_model_rows(history, target_var)
            print(f"\n---> Параметр: {target_var} <---")
            for model_name in self.model_classes.keys():
                print(f"\n  * Модель: {model_name}")
                
                total_fit_time = 0.0
                total_predict_time = 0.0
                total_samples = 0
                results = []

                # Walk-Forward по 12 месяцам года
                for forecast_date, df_train, df_pred in monthly_splits(train_rows, eligible_forecasts, forecast_dates):
                    if not self._can_predict(df_train, df_pred, model_name, target_var, dataset_name):
                        continue

                    preds, fit_t, pred_t, n_samp = self._timed_fit_predict(
                        df_train, df_pred, model_name, target_var, dataset_name
                    )

                    total_fit_time += fit_t
                    total_predict_time += pred_t
                    total_samples += n_samp

                    facts_slice = df_full[df_full['Date'] == forecast_date][['WMO', target_var]]
                    facts_map = dict(zip(facts_slice['WMO'], facts_slice[target_var]))
                    facts = df_pred['WMO'].map(facts_map).values

                    results.append(pd.DataFrame({
                        'Date': forecast_date, 'WMO': df_pred['WMO'].to_numpy(), 'Fact': facts, 'Forecast': preds,
                    }))

                predictions = pd.concat(results) if results else pd.DataFrame(columns=FORECAST_COLUMNS)
                metrics = calculate_climatology_comparison(predictions, climatology, target_var)
                
                speed_samples_sec = total_samples / total_predict_time if total_predict_time > 0 else 0.0
                total_time = total_fit_time + total_predict_time

                print(f"    - Время обучения (Fit Time):      {total_fit_time:.4f} сек")
                print(f"    - Время прогноза (Predict Time):  {total_predict_time:.4f} сек")
                print(f"    - Суммарное время (Total Time):   {total_time:.4f} сек")
                print(f"    - Скорость вычислений:            {speed_samples_sec:.2f} записей/сек ({total_predict_time/12:.4f} сек/месяц)")
                print(format_forecast_metrics(metrics, padding='    '))

                model_info = {
                    'Блок': f'Стандартный прогноз ({target_year})',
                    'Датасет': dataset_name,
                    'Параметр': target_var,
                    'Модель': model_name,
                }
                self.summary_table.append({
                    **model_info,
                    'Fit Time (с)': round(total_fit_time, 4),
                    'Predict Time (с)': round(total_predict_time, 4),
                    'Total Time (с)': round(total_time, 4),
                    'Скорость (зап/с)': round(speed_samples_sec, 2),
                })
                self.accuracy_table.extend({**model_info, **row} for row in metrics)
            self._save_mlp_diagnostics(forecast_filename(dataset_name, target_var, target_year))

    def run_validation_report(self, split_year, dataset_name, end_year=VALIDATION_END_YEAR):
        """Прогон и измерение метрик для долгосрочной валидации."""
        self.mlp_diagnostics.fits.clear()
        print("\n" + "="*80)
        print(f" БЛОК 2: ПРЯМОЙ ПРОГНОЗ НА 12 МЕСЯЦЕВ ({dataset_name} | Период оценки: {split_year}-{end_year})")
        print(" Переобучение ежегодно")
        print("="*80)

        t_load_start = time.perf_counter()
        df_full = self.load_data(dataset_name)
        t_load_end = time.perf_counter()
        print(f"  [Время загрузки данных и базовой предобработки: {t_load_end - t_load_start:.3f} сек]")

        if df_full.empty:
            print("  ОШИБКА: Данные не загружены.")
            return

        data_max_year = df_full['Year'].max()
        final_year = min(end_year, data_max_year)
        test_years = list(range(split_year, final_year + 1))

        if not test_years:
            print("  Нет годов для валидации.")
            return

        features = select_model_rows(prepare_annual_features(df_full, final_year))
        climatology = station_month_climatology(df_full, test_years)
        for target_var in ['DTR', 'Precip']:
            targets = annual_targets(df_full, features, target_var)
            print(f"\n---> Параметр: {target_var} <---")

            for model_name in self.model_classes.keys():
                print(f"\n  * Модель: {model_name}")
                
                total_fit_time = 0.0
                total_predict_time = 0.0
                total_samples = 0
                results = []
                splits = self.annual_model_splits(features, targets, test_years, model_name, target_var, dataset_name)
                for df_train, df_test, y_train, facts in splits:
                    preds, fit_t, pred_t, n_samp = self._timed_fit_predict(
                        df_train, df_test, model_name, target_var, dataset_name, y_train=y_train, mode=ANNUAL_FORECAST_MODE,
                    )

                    total_fit_time += fit_t
                    total_predict_time += pred_t
                    total_samples += n_samp

                    results.append(annual_prediction_rows(df_test, preds, facts, model_name, target_var))

                predictions = pd.concat(results) if results else pd.DataFrame(columns=FORECAST_COLUMNS)
                pairs = paired_predictions(predictions)
                if results:
                    print_monthly_errors(pairs)
                metrics = calculate_climatology_comparison(predictions, climatology, target_var)

                speed_samples_sec = total_samples / total_predict_time if total_predict_time > 0 else 0.0
                total_time = total_fit_time + total_predict_time

                print(f"    - Время обучения (Fit Time):      {total_fit_time:.4f} сек")
                print(f"    - Время прогноза (Predict Time):  {total_predict_time:.4f} сек")
                print(f"    - Суммарное время (Total Time):   {total_time:.4f} сек")
                print(f"    - Скорость вычислений:            {speed_samples_sec:.2f} записей/сек ({total_predict_time/len(test_years):.4f} сек/год)")
                print(f"    - Месячных прогнозов: {total_samples}, пар с фактом: {len(pairs)}")
                print(format_forecast_metrics(metrics, padding='    '))

                model_info = {
                    'Блок': f'Валидация ({split_year}-{final_year})',
                    'Датасет': dataset_name,
                    'Параметр': target_var,
                    'Модель': model_name,
                }
                self.summary_table.append({
                    **model_info,
                    'Fit Time (с)': round(total_fit_time, 4),
                    'Predict Time (с)': round(total_predict_time, 4),
                    'Total Time (с)': round(total_time, 4),
                    'Скорость (зап/с)': round(speed_samples_sec, 2),
                })
                self.accuracy_table.extend({**model_info, **row} for row in metrics)
            self._save_mlp_diagnostics(validation_filename(dataset_name, target_var, split_year, final_year))

    def print_summary_table(self):
        """Отрисовывает финальную сводную таблицу результатов."""
        print("\n" + "="*110)
        print("                  СВОДНАЯ СРАВНИТЕЛЬНАЯ ТАБЛИЦА ПРОИЗВОДИТЕЛЬНОСТИ И ТОЧНОСТИ МОДЕЛЕЙ")
        print("="*110)
        df_sum = pd.DataFrame(self.summary_table)
        if not df_sum.empty:
            print("ПРОИЗВОДИТЕЛЬНОСТЬ")
            print(df_sum.to_string(index=False))
            print("\nТОЧНОСТЬ ПРОГНОЗОВ")
            print(format_forecast_metrics(self.accuracy_table))
        print("="*110 + "\n")


def main():
    # Настройка дублирующего логгера
    logger = DualLogger(REPORT_FILE)
    old_stdout = sys.stdout
    sys.stdout = logger

    t_start_global = time.perf_counter()

    try:
        print("="*100)
        print("      ОТЧЕТ ПО ВРЕМЕНИ ОБУЧЕНИЯ, СКОРОСТИ ВЫЧИСЛЕНИЙ И ТОЧНОСТИ МОДЕЛЕЙ ПРОГНОЗИРОВАНИЯ")
        print("="*100)
        print(f"Дата и время генерации: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        print(f"Проект: {PROJECT_DIR}")
        print(f"Файл выгрузки: {REPORT_FILE}")

        reporter = ForecastReporter(base_dir=PROJECT_DIR, root_output_dir=RESULTS_DIR)

        # 1. СТАНДАРТНЫЙ ПРОГНОЗ
        reporter.run_standard_forecast_report(target_year=ANALYSIS_YEAR, dataset_name='METEO_RU')
        reporter.run_standard_forecast_report(target_year=ANALYSIS_YEAR, dataset_name='CRU_CEDA')

        # 2. ДОЛГОСРОЧНАЯ ВАЛИДАЦИЯ
        reporter.run_validation_report(split_year=VALIDATION_START_YEAR, dataset_name='METEO_RU', end_year=VALIDATION_END_YEAR)
        reporter.run_validation_report(split_year=VALIDATION_START_YEAR, dataset_name='CRU_CEDA', end_year=VALIDATION_END_YEAR)

        # 3. ИТОГОВАЯ СВОДНАЯ ТАБЛИЦА
        reporter.print_summary_table()

        t_end_global = time.perf_counter()
        total_glob_time = t_end_global - t_start_global
        print(f">>> ВСЕ РАСЧЕТЫ ЗАВЕРШЕНЫ УСПЕШНО ЗА {total_glob_time:.2f} сек ({total_glob_time/60:.2f} мин) <<<")
        print(f">>> Итоговый текстовый отчет сохранен в: {REPORT_FILE} <<<")

    finally:
        sys.stdout = old_stdout
        logger.close()


if __name__ == "__main__":
    main()
