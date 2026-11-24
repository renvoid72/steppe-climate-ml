# -*- coding: utf-8 -*-
"""
================================================================================
 Скрипт глубокого обучения (Deep Learning / LSTM) для климатического прогнозирования
 Проект: Исследование климатических показателей степной зоны РФ (Версия 011)
 Модель: Long Short-Term Memory (LSTM Neural Network на PyTorch)
 Входное окно состоит из последовательных календарных месяцев одной станции.
================================================================================
"""

import sys
import os
import time
from pathlib import Path
import warnings
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler

import torch
import torch.nn as nn
from torch.utils.data import TensorDataset, DataLoader

from kursach011_modules.annual_forecast import (
    ANNUAL_FORECAST_MODE, annual_prediction_rows, annual_splits, annual_targets, print_monthly_errors,
)
from kursach011_modules.forecast_engine import mark_prediction_source
from kursach011_modules.forecast_metrics import calculate_forecast_metrics, format_forecast_metrics, paired_predictions
from kursach011_modules.project_settings import (
    ANALYSIS_YEAR, VALIDATION_START_YEAR, VALIDATION_END_YEAR,
    cru_csv_path, forecast_filename, results_dir, validation_filename,
)
from prepare_cru import ensure_cru_csv

warnings.filterwarnings('ignore')

# --- СИСТЕМНЫЕ ПУТИ ---
PROJECT_DIR = Path(__file__).resolve().parent.parent
CODE_DIR = PROJECT_DIR / "code"
MODULES_DIR = CODE_DIR / "kursach011_modules"

RESULTS_DIR = results_dir(PROJECT_DIR) / "lstm"
STD_CSV_DIR = RESULTS_DIR / "standard_forecast" / "csv_data"
VAL_CSV_DIR = RESULTS_DIR / "validation_results" / "csv_data"
REPORT_FILE = RESULTS_DIR / "forecast_report.txt"

FORECAST_COLUMNS = ['Date', 'WMO', 'Latitude', 'Longitude', 'Forecast', 'Fact', 'Model', 'Target_Var']

# Подключение путей для импорта
for p in [str(CODE_DIR), str(MODULES_DIR), str(PROJECT_DIR)]:
    if p not in sys.path:
        sys.path.insert(0, p)

from kursach011_modules.meteo_data import load_monthly_meteo
from kursach011_modules.meteo_loader import SteppeClimateLoader


# ==============================================================================
# ЛОГГЕР ДЛЯ ДУБЛИРОВАНИЯ В КОНСОЛЬ И В ТЕКСТОВЫЙ ФАЙЛ
# ==============================================================================
class DualLogger:
    """Дублирует все выводы print одновременно в консоль и в файл отчета."""
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


# ==============================================================================
# АРХИТЕКТУРА ГЛУБОКОЙ НЕЙРОСЕТИ (PyTorch LSTM)
# ==============================================================================
class ClimateLSTM(nn.Module):
    """
    Глубокая рекуррентная нейросеть LSTM для временных рядов климатических показателей.
    Состоит из 2 слоев LSTM с Dropout и полносвязной регрессионной головки (MLP Head).
    """
    def __init__(self, input_dim, hidden_dim=64, num_layers=2, dropout=0.2, output_dim=1):
        super().__init__()
        self.lstm = nn.LSTM(
            input_size=input_dim,
            hidden_size=hidden_dim,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout if num_layers > 1 else 0.0
        )
        self.fc = nn.Sequential(
            nn.Linear(hidden_dim, 32),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(32, output_dim)
        )

    def forward(self, x):
        # x shape: (batch_size, seq_len, input_dim)
        lstm_out, _ = self.lstm(x)
        # Берем скрытое состояние последнего временного шага
        last_step = lstm_out[:, -1, :]
        out = self.fc(last_step)
        return out.squeeze(-1)


# ==============================================================================
# ОСНОВНОЙ ДВИЖОК ОБУЧЕНИЯ И ПРОГНОЗИРОВАНИЯ LSTM
# ==============================================================================
class LSTMFORECASTENGINE:
    def __init__(self, base_dir=PROJECT_DIR, lookback_window=12):
        self.base_dir = Path(base_dir)
        self.lookback = lookback_window # 12 месяцев истории
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.summary_table = []
        self.accuracy_table = []

    def load_data(self, dataset_name):
        """Загрузка и базовая подготовка климатических данных."""
        if dataset_name == 'METEO_RU':
            file_path = self.base_dir / "meteo_ru" / "all_steppe_data.txt"
            coords_path = self.base_dir / "meteo_ru" / "stations_coords.txt"
            
            if not file_path.exists() or not coords_path.exists():
                print(f"  ОШИБКА: Файлы данных не найдены в {self.base_dir}/meteo_ru")
                return pd.DataFrame()

            df_monthly = load_monthly_meteo(file_path, coords_path)
            
        elif dataset_name == 'CRU_CEDA':
            file_path = cru_csv_path(self.base_dir)
            if not file_path.exists():
                print(f"  ОШИБКА: Не найден файл {file_path}")
                return pd.DataFrame()
            df_monthly = pd.read_csv(file_path)
        else:
            raise ValueError(f"Неизвестный dataset_name: {dataset_name}")

        df_monthly['Date'] = pd.to_datetime(df_monthly[['Year', 'Month']].assign(day=1))
        df_monthly = df_monthly[df_monthly['Year'] >= 1960]
        required_columns = ['Latitude', 'Longitude']
        if dataset_name == 'CRU_CEDA':
            required_columns.extend(['DTR', 'Precip'])
        df_monthly = df_monthly.dropna(subset=required_columns)
        df_monthly = df_monthly.sort_values(by=['WMO', 'Date']).reset_index(drop=True)
        return df_monthly

    def _prepare_features_df(self, df):
        """Добавление гармонических сезонных признаков."""
        df = df.copy()
        df['Month_sin'] = np.sin(2 * np.pi * df['Date'].dt.month / 12)
        df['Month_cos'] = np.cos(2 * np.pi * df['Date'].dt.month / 12)
        return df

    def _build_sequences(self, df, target_var, *, annual=False):
        """Формирует входные окна. Для годового прогноза последнее значение — декабрь."""
        df = self._prepare_features_df(df.sort_values(['WMO', 'Date']))
        
        # Подготовка признаков: целевая + ко-переменная + календарь + координаты
        if target_var == 'DTR':
            t_col = 'DTR'
            aux_col = 'Precip'
            target_series = df[t_col].values
            aux_series = np.log1p(df[aux_col].values)
        else:
            t_col = 'Precip'
            aux_col = 'DTR'
            target_series = np.log1p(df[t_col].values)
            aux_series = df[aux_col].values

        features_raw = np.column_stack([
            target_series,
            aux_series,
            df['Month_sin'].values,
            df['Month_cos'].values,
            df['Latitude'].values,
            df['Longitude'].values
        ])

        dates_all = df['Date'].values
        months_all = (df['Date'].dt.year * 12 + df['Date'].dt.month).to_numpy()
        wmo_all = df['WMO'].to_numpy()
        lat_all = df['Latitude'].values
        lon_all = df['Longitude'].values

        X_list, y_list, date_list, wmo_list, lat_list, lon_list = [], [], [], [], [], []

        for w in np.unique(wmo_all):
            mask = (wmo_all == w)
            w_feats = features_raw[mask]
            w_dates = dates_all[mask]
            w_targets = df[target_var].values[mask]
            w_lats = lat_all[mask]
            w_lons = lon_all[mask]

            w_months = months_all[mask]
            if annual:
                # Последний месяц признаков — декабрь. Наличие будущего января не требуется.
                cutoffs = np.flatnonzero(w_months[self.lookback - 1:] % 12 == 0) + self.lookback
            else:
                if len(w_months) < self.lookback:
                    continue
                steps = np.lib.stride_tricks.sliding_window_view(np.diff(w_months) == 1, self.lookback - 1)
                cutoffs = np.flatnonzero(steps.all(axis=1)) + self.lookback

            for i in cutoffs:
                history = w_feats[i - self.lookback:i]
                if not np.isfinite(history).all():
                    continue
                if annual and not (np.diff(w_months[i - self.lookback:i]) == 1).all():
                    continue
                if not annual and i < len(w_months) and w_months[i] == w_months[i - 1] + 1:
                    date, coord_index = w_dates[i], i
                    target = w_targets[i] if np.isfinite(w_targets[i]) else np.nan
                else:
                    date = pd.Timestamp(w_dates[i - 1]) + pd.offsets.MonthBegin()
                    coord_index = i - 1
                    target = np.nan
                X_list.append(history)
                y_list.append(target)
                date_list.append(date)
                wmo_list.append(w)
                lat_list.append(w_lats[coord_index])
                lon_list.append(w_lons[coord_index])

        X_arr = np.array(X_list, dtype=np.float32).reshape(-1, self.lookback, features_raw.shape[1])
        y_arr = np.array(y_list, dtype=np.float32)
        date_arr = np.array(date_list, dtype=dates_all.dtype)
        wmo_arr = np.array(wmo_list, dtype=wmo_all.dtype)
        lat_arr = np.array(lat_list, dtype=np.float32)
        lon_arr = np.array(lon_list, dtype=np.float32)

        return X_arr, y_arr, date_arr, wmo_arr, lat_arr, lon_arr

    def _train_lstm_model(self, X_train, y_train, target_var, epochs=25, batch_size=256):
        """Обучение LSTM сети на тензорах истории."""
        torch.manual_seed(42)
        np.random.seed(42)

        N, seq_len, n_feat = X_train.shape

        # Масштабирование признаков (StandardScaler по 2D развертке)
        scaler = StandardScaler()
        X_train_reshaped = X_train.reshape(-1, n_feat)
        scaler.fit(X_train_reshaped)
        X_train_scaled = scaler.transform(X_train_reshaped).reshape(N, seq_len, n_feat)

        # Преобразование целевой переменной
        y_train_proc = np.log1p(y_train) if target_var == 'Precip' else y_train.copy()
        y_scaler = StandardScaler()
        y_matrix = y_train_proc.reshape(N, -1)
        y_train_scaled = y_scaler.fit_transform(y_matrix).reshape(y_train.shape)

        # Создание DataLoader
        tensor_x = torch.tensor(X_train_scaled, dtype=torch.float32)
        tensor_y = torch.tensor(y_train_scaled, dtype=torch.float32)
        dataset = TensorDataset(tensor_x, tensor_y)
        loader = DataLoader(dataset, batch_size=batch_size, shuffle=True, drop_last=False)

        # Модель
        model = ClimateLSTM(input_dim=n_feat, output_dim=y_matrix.shape[1]).to(self.device)
        criterion = nn.HuberLoss(delta=1.0)
        optimizer = torch.optim.AdamW(model.parameters(), lr=0.005, weight_decay=1e-4)
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)

        model.train()
        for epoch in range(epochs):
            for bx, by in loader:
                bx, by = bx.to(self.device), by.to(self.device)
                optimizer.zero_grad()
                out = model(bx)
                loss = criterion(out, by)
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
                optimizer.step()
            scheduler.step()

        return model, scaler, y_scaler

    def _predict_lstm(self, model, scaler, y_scaler, X_test, target_var):
        """Инференс обученной LSTM модели на тестовых окнах."""
        model.eval()
        N, seq_len, n_feat = X_test.shape
        X_test_reshaped = X_test.reshape(-1, n_feat)
        X_test_scaled = scaler.transform(X_test_reshaped).reshape(N, seq_len, n_feat)

        tensor_x = torch.tensor(X_test_scaled, dtype=torch.float32).to(self.device)
        with torch.no_grad():
            raw_out = model(tensor_x).cpu().numpy()

        # Обратная денормализация
        pred_scaled = y_scaler.inverse_transform(raw_out.reshape(N, -1)).reshape(raw_out.shape)
        if target_var == 'Precip':
            preds = np.expm1(pred_scaled)
        else:
            preds = pred_scaled

        return np.maximum(0.0, preds)

    # ==========================================================================
    # 1. СТАНДАРТНЫЙ ПРОГНОЗ (1 ГОД ПОШАГОВО WALK-FORWARD)
    # ==========================================================================
    def run_standard_forecast(self, target_year, dataset_name):
        print("\n" + "="*80)
        print(f" БЛОК 1: ПРОГНОЗ НА МЕСЯЦ ({dataset_name} | Период оценки: {target_year})")
        print(" Переобучение ежемесячно")
        print("="*80)

        t_load_start = time.perf_counter()
        df_full = self.load_data(dataset_name)
        t_load_end = time.perf_counter()
        print(f"  [Время загрузки данных и базовой предобработки: {t_load_end - t_load_start:.3f} сек]")

        if df_full.empty:
            return

        STD_CSV_DIR.mkdir(parents=True, exist_ok=True)
        for target_var in ['DTR', 'Precip']:
            fname = forecast_filename(dataset_name, target_var, target_year)
            print(f"\n---> Параметр: {target_var} <---")
            print(f"\n  * Модель: LSTM Neural Network")

            # Формируем все временные окна
            X_all, y_all, dates_all, wmo_all, lats_all, lons_all = self._build_sequences(df_full, target_var)
            has_fact = np.isfinite(y_all)

            all_res_rows = []
            total_fit_time = 0.0
            total_pred_time = 0.0
            total_samples = 0

            # Пошаговый помесячный Walk-Forward для 12 месяцев целевого года
            for month_num in range(1, 13):
                forecast_date = pd.to_datetime(f"{target_year}-{month_num:02d}-01")

                train_mask = (dates_all < forecast_date) & has_fact
                test_mask = (dates_all == forecast_date)

                if not np.any(train_mask) or not np.any(test_mask):
                    continue

                X_train, y_train = X_all[train_mask], y_all[train_mask]
                X_test, y_test = X_all[test_mask], y_all[test_mask]
                test_wmo = wmo_all[test_mask]
                test_lat = lats_all[test_mask]
                test_lon = lons_all[test_mask]

                # 1. Обучение сети
                t_fit_start = time.perf_counter()
                model, feat_scaler, y_scaler = self._train_lstm_model(X_train, y_train, target_var, epochs=20)
                t_fit_end = time.perf_counter()
                fit_t = t_fit_end - t_fit_start

                # 2. Инференс прогноза
                t_pred_start = time.perf_counter()
                # Сохраняем состав батча, по которому рассчитываются метрики.
                scored = has_fact[test_mask]
                preds = np.empty(len(y_test), dtype=np.float32)
                for batch in (scored, ~scored):
                    if batch.any():
                        preds[batch] = self._predict_lstm(
                            model, feat_scaler, y_scaler, X_test[batch], target_var,
                        ).reshape(-1)
                t_pred_end = time.perf_counter()
                pred_t = t_pred_end - t_pred_start

                total_fit_time += fit_t
                total_pred_time += pred_t
                total_samples += len(preds)

                # Сохранение результатов шага
                for i in range(len(preds)):
                    all_res_rows.append({
                        'Date': forecast_date.strftime('%Y-%m-%d'),
                        'WMO': test_wmo[i],
                        'Latitude': test_lat[i],
                        'Longitude': test_lon[i],
                        'Forecast': float(preds[i]),
                        'Fact': float(y_test[i]),
                        'Model': 'LSTM Neural Network',
                        'Target_Var': target_var
                    })

            df_out = pd.DataFrame(all_res_rows, columns=FORECAST_COLUMNS)
            pairs = paired_predictions(df_out)
            metrics = calculate_forecast_metrics(pairs, target_var)

            speed_samples_sec = total_samples / total_pred_time if total_pred_time > 0 else 0.0
            total_time = total_fit_time + total_pred_time

            print(f"    - Время обучения (Fit Time):      {total_fit_time:.4f} сек")
            print(f"    - Время прогноза (Predict Time):  {total_pred_time:.4f} сек")
            print(f"    - Суммарное время (Total Time):   {total_time:.4f} сек")
            print(f"    - Скорость вычислений:            {speed_samples_sec:.2f} записей/сек ({total_pred_time/12:.4f} сек/месяц)")
            print(f"    - Месячных прогнозов: {len(df_out)}, пар с фактом: {len(pairs)}")
            print(format_forecast_metrics(metrics, padding='    '))

            # Сохранение CSV
            df_out = df_out.round({'Forecast': 4, 'Fact': 4})
            mark_prediction_source(df_out, dataset_name)
            df_out.to_csv(STD_CSV_DIR / fname, index=False)

            model_info = {
                'Блок': f'Стандартный прогноз ({target_year})',
                'Датасет': dataset_name,
                'Параметр': target_var,
                'Модель': 'LSTM Neural Network',
            }
            self.summary_table.append({
                **model_info,
                'Fit Time (с)': round(total_fit_time, 4),
                'Predict Time (с)': round(total_pred_time, 4),
                'Total Time (с)': round(total_time, 4),
                'Скорость (зап/с)': round(speed_samples_sec, 2),
            })
            self.accuracy_table.extend({**model_info, **row} for row in metrics)

    # ==========================================================================
    # 2. ДОЛГОСРОЧНАЯ ВАЛИДАЦИЯ
    # ==========================================================================
    def run_validation(self, split_year, dataset_name, end_year=VALIDATION_END_YEAR):
        print("\n" + "="*80)
        print(f" БЛОК 2: ПРЯМОЙ ПРОГНОЗ НА 12 МЕСЯЦЕВ ({dataset_name} | Период оценки: {split_year}-{end_year})")
        print(" Переобучение ежегодно")
        print("="*80)

        t_load_start = time.perf_counter()
        df_full = self.load_data(dataset_name)
        t_load_end = time.perf_counter()
        print(f"  [Время загрузки данных и базовой предобработки: {t_load_end - t_load_start:.3f} сек]")

        if df_full.empty:
            return

        data_max_year = df_full['Year'].max()
        final_year = min(end_year, data_max_year)
        test_years = list(range(split_year, final_year + 1))
        if not test_years:
            return

        history = df_full.loc[df_full['Year'] < final_year]
        VAL_CSV_DIR.mkdir(parents=True, exist_ok=True)
        for target_var in ['DTR', 'Precip']:
            fname = validation_filename(dataset_name, target_var, split_year, final_year)
            print(f"\n---> Параметр: {target_var} <---")
            print(f"\n  * Модель: LSTM Neural Network")

            X_all, _, dates, wmo, lat, lon = self._build_sequences(history, target_var, annual=True)
            anchors = pd.DataFrame({'Date': dates, 'WMO': wmo, 'Latitude': lat, 'Longitude': lon})
            targets = annual_targets(df_full, anchors, target_var).astype(np.float32)

            results = []
            total_fit_time = 0.0
            total_pred_time = 0.0

            for train_mask, test_mask in annual_splits(anchors, targets, test_years):
                if not np.any(train_mask) or not np.any(test_mask):
                    continue

                t_fit_start = time.perf_counter()
                model, feat_scaler, y_scaler = self._train_lstm_model(
                    X_all[train_mask], targets[train_mask], target_var, epochs=15,
                )
                total_fit_time += time.perf_counter() - t_fit_start

                t_pred_start = time.perf_counter()
                preds = self._predict_lstm(model, feat_scaler, y_scaler, X_all[test_mask], target_var)
                total_pred_time += time.perf_counter() - t_pred_start
                results.append(annual_prediction_rows(
                    anchors.loc[test_mask], preds, targets[test_mask], 'LSTM Neural Network', target_var,
                ))

            df_out = pd.concat(results, ignore_index=True) if results else pd.DataFrame(columns=FORECAST_COLUMNS)
            pairs = paired_predictions(df_out)
            if not df_out.empty:
                print_monthly_errors(pairs)
            metrics = calculate_forecast_metrics(pairs, target_var)

            speed_samples_sec = len(df_out) / total_pred_time if total_pred_time > 0 else 0.0
            total_time = total_fit_time + total_pred_time

            print(f"    - Время обучения (Fit Time):      {total_fit_time:.4f} сек")
            print(f"    - Время прогноза (Predict Time):  {total_pred_time:.4f} сек")
            print(f"    - Суммарное время (Total Time):   {total_time:.4f} сек")
            print(f"    - Скорость вычислений:            {speed_samples_sec:.2f} записей/сек ({total_pred_time/len(test_years):.4f} сек/год)")
            print(f"    - Месячных прогнозов: {len(df_out)}, пар с фактом: {len(pairs)}")
            print(format_forecast_metrics(metrics, padding='    '))

            # Сохранение CSV
            df_out = df_out.reindex(columns=FORECAST_COLUMNS).round({'Forecast': 4, 'Fact': 4})
            df_out['Forecast_Mode'] = ANNUAL_FORECAST_MODE
            mark_prediction_source(df_out, dataset_name)
            df_out.to_csv(VAL_CSV_DIR / fname, index=False)

            model_info = {
                'Блок': f'Валидация ({split_year}-{final_year})',
                'Датасет': dataset_name,
                'Параметр': target_var,
                'Модель': 'LSTM Neural Network',
            }
            self.summary_table.append({
                **model_info,
                'Fit Time (с)': round(total_fit_time, 4),
                'Predict Time (с)': round(total_pred_time, 4),
                'Total Time (с)': round(total_time, 4),
                'Скорость (зап/с)': round(speed_samples_sec, 2),
            })
            self.accuracy_table.extend({**model_info, **row} for row in metrics)

    def print_final_summary(self):
        """Печать и сохранение сводной таблицы производительности и точности."""
        print("\n" + "="*115)
        print("                           СВОДНАЯ ТАБЛИЦА РЕЗУЛЬТАТОВ ИСПЫТАНИЙ LSTM")
        print("="*115)
        df_summary = pd.DataFrame(self.summary_table)
        if not df_summary.empty:
            print("ПРОИЗВОДИТЕЛЬНОСТЬ")
            print(df_summary.to_string(index=False))
            print("\nТОЧНОСТЬ ПРОГНОЗОВ")
            print(format_forecast_metrics(self.accuracy_table))
        print("="*115)


# ==============================================================================
# ТОЧКА ВХОДА (MAIN)
# ==============================================================================
def main():
    # Информация об аппаратном устройстве (только в консоль!)
    if torch.cuda.is_available():
        gpu_name = torch.cuda.get_device_name(0)
        device_info = f"CUDA (GPU: {gpu_name})"
    else:
        cpu_count = os.cpu_count()
        device_info = f"CPU ({cpu_count} ядер/потоков)"

    sys.stdout.write(f"\n>>> [ИНФОРМАЦИЯ О СИСТЕМЕ] Вычисления выполняются на устройстве: {device_info} <<<\n\n")
    sys.stdout.flush()

    # Направляем поток вывода в DualLogger (файл + консоль)
    logger = DualLogger(REPORT_FILE)
    sys.stdout = logger

    t_global_start = time.perf_counter()

    print("="*100)
    print("      ОТЧЕТ ПО ВРЕМЕНИ ОБУЧЕНИЯ, СКОРОСТИ ВЫЧИСЛЕНИЙ И ТОЧНОСТИ МОДЕЛЕЙ ПРОГНОЗИРОВАНИЯ (LSTM)")
    print("="*100)

    # 1. Проверка ETL (исходных данных)
    meteo_file = PROJECT_DIR / "meteo_ru" / "all_steppe_data.txt"
    if not meteo_file.exists():
        SteppeClimateLoader(PROJECT_DIR).run()

    ensure_cru_csv(PROJECT_DIR)

    # 2. Инициализация движка
    engine = LSTMFORECASTENGINE(base_dir=PROJECT_DIR, lookback_window=12)

    # 3. Стандартный прогноз на следующий месяц
    engine.run_standard_forecast(target_year=ANALYSIS_YEAR, dataset_name='METEO_RU')
    engine.run_standard_forecast(target_year=ANALYSIS_YEAR, dataset_name='CRU_CEDA')

    # 4. Долгосрочная валидация
    engine.run_validation(split_year=VALIDATION_START_YEAR, dataset_name='METEO_RU', end_year=VALIDATION_END_YEAR)
    engine.run_validation(split_year=VALIDATION_START_YEAR, dataset_name='CRU_CEDA', end_year=VALIDATION_END_YEAR)

    # 5. Итоговая сводка
    engine.print_final_summary()

    t_global_end = time.perf_counter()
    print(f"\n  [Суммарное время выполнения всех расчетов: {t_global_end - t_global_start:.3f} сек]")
    print(f">>> ВСЕ РАСЧЕТЫ УСПЕШНО ЗАВЕРШЕНЫ! Результаты: {RESULTS_DIR} <<<")
    print(f">>> Отчет сохранен в файл: {REPORT_FILE} <<<\n")

    logger.close()
    sys.stdout = logger.terminal


if __name__ == "__main__":
    main()
