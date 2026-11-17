# -*- coding: utf-8 -*-
"""
Автономный скрипт для расширенного хронометрического и аналитического отчета.
Выполняет обучение и прогнозирование с помощью моделей Random Forest, K-Neighbors и MLP,
замеряет точное время обучения, прогноза, скорость обработки данных и точность (R², MAE, RMSE).

Результаты НЕ сохраняются в CSV, а выгружаются в ЕДИНСТВЕННЫЙ итоговый файл:
output011_modules/forecast_report.txt (относительно корня проекта)
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
from sklearn.metrics import r2_score, mean_absolute_error, mean_squared_error

warnings.filterwarnings('ignore')

# --- КОНСТАНТЫ ПУТЕЙ ---
PROJECT_DIR = Path(__file__).resolve().parent.parent
RESULTS_DIR = PROJECT_DIR / "output011_modules"
REPORT_FILE = RESULTS_DIR / "forecast_report.txt"

# Добавляем путь к модулям проекта
sys.path.append(str(PROJECT_DIR / "code"))

from kursach011_modules.forecast_engine import BaseForecaster, BEST_PARAMS


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

    def _timed_fit_predict(self, df_train, df_test, model_name, target_var, dataset_name):
        """
        Идентичная обучающая логика _fit_predict_model из forecast_engine.py,
        но с раздельным хронометрированием этапов fit() и predict().
        """
        params = BEST_PARAMS[dataset_name][target_var][model_name]
        
        cols_exclude = ['Date', 'WMO', 'Year', 'Month', 'Quality', 'Tmin', 'Tavg', 'Tmax', 'Precip', 'DTR', 'Latitude', 'Longitude']
        feature_cols = [c for c in df_train.columns if c not in cols_exclude]
        
        scaler = StandardScaler().fit(df_train[feature_cols])
        
        model_args = {'random_state': 42} if 'Random' in model_name or 'MLP' in model_name else {}
        if 'Random' in model_name or 'Neighbors' in model_name:
            model_args['n_jobs'] = -1
            
        model = self.model_classes[model_name](**params, **model_args)
        
        y_train = np.log1p(df_train[target_var]) if target_var == 'Precip' else df_train[target_var]
        
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

        return final_preds, fit_time, predict_time, len(df_test)

    def run_standard_forecast_report(self, target_year, dataset_name):
        """Прогон и измерение метрик для Стандартного прогноза на 1 год."""
        print("\n" + "="*80)
        print(f" БЛОК 1: СТАНДАРТНЫЙ ПРОГНОЗ ({dataset_name} | Год прогноза: {target_year})")
        print("="*80)

        t_load_start = time.perf_counter()
        df_full = self.load_data(dataset_name)
        t_load_end = time.perf_counter()
        print(f"  [Время загрузки данных и базовой предобработки: {t_load_end - t_load_start:.3f} сек]")

        if df_full.empty:
            print("  ОШИБКА: Данные не загружены.")
            return

        for target_var in ['DTR', 'Precip']:
            print(f"\n---> Параметр: {target_var} <---")
            actuals = df_full[df_full['Year'] == target_year].groupby('Date')

            for model_name in self.model_classes.keys():
                print(f"\n  * Модель: {model_name}")
                
                total_fit_time = 0.0
                total_predict_time = 0.0
                total_samples = 0
                all_preds = []
                all_facts = []

                # Walk-Forward по 12 месяцам года
                for month_num in range(1, 13):
                    forecast_date = pd.to_datetime(f"{target_year}-{month_num:02d}-01")
                    
                    df_hist = df_full[df_full['Date'] < forecast_date].copy()
                    future = df_hist[['WMO', 'Latitude', 'Longitude']].drop_duplicates()
                    future['Date'] = forecast_date
                    for c in ['DTR', 'Precip']:
                        future[c] = np.nan
                    
                    df_combined = pd.concat([df_hist, future], ignore_index=True)
                    df_features = self.create_features(df_combined)
                    
                    df_train = df_features[df_features['Date'] < forecast_date].dropna().reset_index(drop=True)
                    df_pred = df_features[df_features['Date'] == forecast_date].fillna(0)
                    
                    if df_train.empty or df_pred.empty:
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

                    all_preds.extend(preds)
                    all_facts.extend(facts)

                # Расчет итоговых метрик по модели
                valid_mask = ~np.isnan(all_facts)
                clean_facts = np.array(all_facts)[valid_mask]
                clean_preds = np.array(all_preds)[valid_mask]

                r2 = r2_score(clean_facts, clean_preds) if len(clean_facts) > 0 else 0.0
                mae = mean_absolute_error(clean_facts, clean_preds) if len(clean_facts) > 0 else 0.0
                rmse = np.sqrt(mean_squared_error(clean_facts, clean_preds)) if len(clean_facts) > 0 else 0.0
                
                speed_samples_sec = total_samples / total_predict_time if total_predict_time > 0 else 0.0
                total_time = total_fit_time + total_predict_time

                print(f"    - Время обучения (Fit Time):      {total_fit_time:.4f} сек")
                print(f"    - Время прогноза (Predict Time):  {total_predict_time:.4f} сек")
                print(f"    - Суммарное время (Total Time):   {total_time:.4f} сек")
                print(f"    - Скорость вычислений:            {speed_samples_sec:.2f} записей/сек ({total_predict_time/12:.4f} сек/месяц)")
                print(f"    - Метрики точности:               R² = {r2:.4f} | MAE = {mae:.4f} | RMSE = {rmse:.4f}")

                self.summary_table.append({
                    'Блок': 'Стандартный прогноз',
                    'Датасет': dataset_name,
                    'Параметр': target_var,
                    'Модель': model_name,
                    'Fit Time (с)': round(total_fit_time, 4),
                    'Predict Time (с)': round(total_predict_time, 4),
                    'Total Time (с)': round(total_time, 4),
                    'Скорость (зап/с)': round(speed_samples_sec, 2),
                    'R²': round(r2, 4),
                    'MAE': round(mae, 4),
                    'RMSE': round(rmse, 4)
                })

    def run_validation_report(self, split_year, dataset_name, end_year=2020):
        """Прогон и измерение метрик для Долгосрочной валидации (1990-2020 гг., 31 год)."""
        print("\n" + "="*80)
        print(f" БЛОК 2: ДОЛГОСРОЧНАЯ ВАЛИДАЦИЯ ({dataset_name} | Период: {split_year}-{end_year})")
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

        for target_var in ['DTR', 'Precip']:
            print(f"\n---> Параметр: {target_var} <---")

            for model_name in self.model_classes.keys():
                print(f"\n  * Модель: {model_name}")
                
                total_fit_time = 0.0
                total_predict_time = 0.0
                total_samples = 0
                all_preds = []
                all_facts = []

                # Yearly Walk-Forward
                for year in test_years:
                    split_date = pd.to_datetime(f"{year}-01-01")
                    
                    df_current_slice = df_full[df_full['Year'] <= year].copy()
                    df_features = self.create_features(df_current_slice)
                    
                    df_train = df_features[df_features['Date'] < split_date].dropna().reset_index(drop=True)
                    df_test = df_features[df_features['Year'] == year].reset_index(drop=True).fillna(0)
                    
                    if df_train.empty or df_test.empty:
                        continue

                    preds, fit_t, pred_t, n_samp = self._timed_fit_predict(
                        df_train, df_test, model_name, target_var, dataset_name
                    )

                    total_fit_time += fit_t
                    total_predict_time += pred_t
                    total_samples += n_samp

                    all_preds.extend(preds)
                    all_facts.extend(df_test[target_var].values)

                # Расчет итоговых метрик по модели
                clean_facts = np.array(all_facts)
                clean_preds = np.array(all_preds)

                r2 = r2_score(clean_facts, clean_preds) if len(clean_facts) > 0 else 0.0
                mae = mean_absolute_error(clean_facts, clean_preds) if len(clean_facts) > 0 else 0.0
                rmse = np.sqrt(mean_squared_error(clean_facts, clean_preds)) if len(clean_facts) > 0 else 0.0

                speed_samples_sec = total_samples / total_predict_time if total_predict_time > 0 else 0.0
                total_time = total_fit_time + total_predict_time

                print(f"    - Время обучения (Fit Time):      {total_fit_time:.4f} сек")
                print(f"    - Время прогноза (Predict Time):  {total_predict_time:.4f} сек")
                print(f"    - Суммарное время (Total Time):   {total_time:.4f} сек")
                print(f"    - Скорость вычислений:            {speed_samples_sec:.2f} записей/сек ({total_predict_time/len(test_years):.4f} сек/год)")
                print(f"    - Метрики точности (1990-2020):   R² = {r2:.4f} | MAE = {mae:.4f} | RMSE = {rmse:.4f}")

                self.summary_table.append({
                    'Блок': 'Валидация (1990-2020)',
                    'Датасет': dataset_name,
                    'Параметр': target_var,
                    'Модель': model_name,
                    'Fit Time (с)': round(total_fit_time, 4),
                    'Predict Time (с)': round(total_predict_time, 4),
                    'Total Time (с)': round(total_time, 4),
                    'Скорость (зап/с)': round(speed_samples_sec, 2),
                    'R²': round(r2, 4),
                    'MAE': round(mae, 4),
                    'RMSE': round(rmse, 4)
                })

    def print_summary_table(self):
        """Отрисовывает финальную сводную таблицу результатов."""
        print("\n" + "="*110)
        print("                  СВОДНАЯ СРАВНИТЕЛЬНАЯ ТАБЛИЦА ПРОИЗВОДИТЕЛЬНОСТИ И ТОЧНОСТИ МОДЕЛЕЙ")
        print("="*110)
        df_sum = pd.DataFrame(self.summary_table)
        if not df_sum.empty:
            pd.set_option('display.max_columns', 15)
            pd.set_option('display.width', 1000)
            print(df_sum.to_string(index=False))
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
        reporter.run_standard_forecast_report(target_year=2024, dataset_name='METEO_RU')
        reporter.run_standard_forecast_report(target_year=2020, dataset_name='CRU_CEDA')

        # 2. ДОЛГОСРОЧНАЯ ВАЛИДАЦИЯ (1990-2020 гг.)
        reporter.run_validation_report(split_year=1990, dataset_name='METEO_RU', end_year=2020)
        reporter.run_validation_report(split_year=1990, dataset_name='CRU_CEDA', end_year=2020)

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
