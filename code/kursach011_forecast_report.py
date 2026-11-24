# -*- coding: utf-8 -*-
"""
Автономный скрипт для расширенного хронометрического и аналитического отчета.
Выполняет обучение и прогнозирование с помощью моделей Random Forest, K-Neighbors и MLP,
замеряет точное время обучения, прогноза, скорость обработки данных и точность (R2, MAE, RMSE).

Результаты НЕ сохраняются в CSV, а выгружаются в ЕДИНСТВЕННЫЙ итоговый файл:
forecast_report.txt в каталоге результатов рабочей версии CRU.
"""

import sys
import os
import time
from datetime import datetime
from pathlib import Path
import pandas as pd
import numpy as np
import warnings
from sklearn.preprocessing import StandardScaler
from kursach011_modules.project_settings import ANALYSIS_YEAR, VALIDATION_START_YEAR, VALIDATION_END_YEAR, results_dir

warnings.filterwarnings('ignore')

# --- КОНСТАНТЫ ПУТЕЙ ---
PROJECT_DIR = Path(__file__).resolve().parent.parent
RESULTS_DIR = results_dir(PROJECT_DIR)
REPORT_FILE = RESULTS_DIR / "forecast_report.txt"

# Добавляем путь к модулям проекта
sys.path.append(str(PROJECT_DIR / "code"))

from kursach011_modules.forecast_engine import BaseForecaster, BEST_PARAMS
from kursach011_modules.annual_forecast import annual_prediction_rows, annual_targets, prepare_annual_features, print_monthly_errors
from kursach011_modules.forecast_metrics import calculate_forecast_metrics, format_forecast_metrics, paired_predictions
from kursach011_modules.monthly_features import prepare_forecast_period, model_feature_columns, select_model_rows


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
        self.summary_table = []
        self.accuracy_table = []

    def _timed_fit_predict(self, df_train, df_test, model_name, target_var, dataset_name, *, y_train=None):
        """
        Идентичная обучающая логика _fit_predict_model из forecast_engine.py,
        но с раздельным хронометрированием этапов fit() и predict().
        """
        params = BEST_PARAMS[dataset_name][target_var][model_name]
        
        feature_cols = model_feature_columns(df_train)
        
        scaler = StandardScaler().fit(df_train[feature_cols])
        
        model_args = {'random_state': 42} if 'Random' in model_name or 'MLP' in model_name else {}
        if 'Random' in model_name or 'Neighbors' in model_name:
            model_args['n_jobs'] = -1
            
        model = self.model_classes[model_name](**params, **model_args)
        
        if y_train is None:
            y_train = df_train[target_var]
        if target_var == 'Precip':
            y_train = np.log1p(y_train)
        
        X_train_scaled = scaler.transform(df_train[feature_cols])
        X_test_scaled = scaler.transform(df_test[feature_cols])

        # 1. Замер времени обучения (Fit Time)
        t_fit_start = time.perf_counter()
        model.fit(X_train_scaled, y_train)
        t_fit_end = time.perf_counter()
        fit_time = t_fit_end - t_fit_start

        # 2. Замер времени прогноза (Predict Time)
        t_pred_start = time.perf_counter()
        raw_preds = model.predict(X_test_scaled)
        t_pred_end = time.perf_counter()
        predict_time = t_pred_end - t_pred_start

        final_preds = np.expm1(raw_preds) if target_var == 'Precip' else raw_preds
        final_preds = np.maximum(0, final_preds)

        return final_preds, fit_time, predict_time, final_preds.size

    def run_standard_forecast_report(self, target_year, dataset_name):
        """Помесячная оценка с горизонтом один месяц и ежемесячным обучением."""
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
                for forecast_date in forecast_dates:
                    df_train = train_rows.loc[train_rows['Date'] < forecast_date]
                    df_pred = eligible_forecasts.loc[eligible_forecasts['Date'] == forecast_date]
                    
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

                    results.append(pd.DataFrame({'Date': forecast_date, 'Fact': facts, 'Forecast': preds}))

                predictions = pd.concat(results) if results else pd.DataFrame(columns=['Date', 'Fact', 'Forecast'])
                metrics = calculate_forecast_metrics(paired_predictions(predictions), target_var)
                
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

    def run_validation_report(self, split_year, dataset_name, end_year=VALIDATION_END_YEAR):
        """Прогон и измерение метрик для долгосрочной валидации."""
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
                        df_train, df_test, model_name, target_var, dataset_name, y_train=y_train,
                    )

                    total_fit_time += fit_t
                    total_predict_time += pred_t
                    total_samples += n_samp

                    results.append(annual_prediction_rows(df_test, preds, facts, model_name, target_var))

                predictions = pd.concat(results) if results else pd.DataFrame(columns=['Date', 'Fact', 'Forecast'])
                pairs = paired_predictions(predictions)
                if results:
                    print_monthly_errors(pairs)
                metrics = calculate_forecast_metrics(pairs, target_var)

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
