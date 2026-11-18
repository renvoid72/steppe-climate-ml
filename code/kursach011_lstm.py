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
from sklearn.metrics import r2_score, mean_absolute_error, mean_squared_error

import torch
import torch.nn as nn
from torch.utils.data import TensorDataset, DataLoader

warnings.filterwarnings('ignore')

# --- СИСТЕМНЫЕ ПУТИ ---
PROJECT_DIR = Path(__file__).resolve().parent.parent
CODE_DIR = PROJECT_DIR / "code"
MODULES_DIR = CODE_DIR / "kursach011_modules"

RESULTS_DIR = PROJECT_DIR / "output011_modules" / "lstm"
STD_CSV_DIR = RESULTS_DIR / "standard_forecast" / "csv_data"
VAL_CSV_DIR = RESULTS_DIR / "validation_results" / "csv_data"
REPORT_FILE = RESULTS_DIR / "forecast_report.txt"

FORECAST_COLUMNS = ['Date', 'WMO', 'Latitude', 'Longitude', 'Forecast', 'Fact', 'Model', 'Target_Var']

# Подключение путей для импорта
for p in [str(CODE_DIR), str(MODULES_DIR), str(PROJECT_DIR)]:
    if p not in sys.path:
        sys.path.insert(0, p)

from kursach011_modules.meteo_loader import SteppeClimateLoader
from kursach011_modules.cru_loader import CruDataLoader


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
    def __init__(self, input_dim, hidden_dim=64, num_layers=2, dropout=0.2):
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
            nn.Linear(32, 1)
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

    def load_data(self, dataset_name):
        """Загрузка и базовая подготовка климатических данных."""
        if dataset_name == 'METEO_RU':
            file_path = self.base_dir / "meteo_ru" / "all_steppe_data.txt"
            coords_path = self.base_dir / "meteo_ru" / "stations_coords.txt"
            
            if not file_path.exists() or not coords_path.exists():
                print(f"  ОШИБКА: Файлы данных не найдены в {self.base_dir}/meteo_ru")
                return pd.DataFrame()

            cols = ['WMO', 'Year', 'Month', 'Day', 'Quality', 'Tmin', 'Tavg', 'Tmax', 'Precip']
            df = pd.read_csv(file_path, sep=r'\s+', names=cols, na_values=[-999.9])
            df['DTR'] = df['Tmax'] - df['Tmin']
            
            coords = pd.read_csv(coords_path, sep=r'\s+')
            lat_map = dict(zip(coords['WMO'], coords['Lat']))
            lon_map = dict(zip(coords['WMO'], coords['Lon']))
            
            df['Latitude'] = df['WMO'].map(lat_map)
            df['Longitude'] = df['WMO'].map(lon_map)
            
            df_monthly = df.groupby(['WMO', 'Year', 'Month', 'Latitude', 'Longitude']).agg(
                DTR=('DTR', 'mean'), 
                Precip=('Precip', 'sum')
            ).reset_index()
            
        elif dataset_name == 'CRU_CEDA':
            file_path = self.base_dir / "cru ceda" / "cru_aligned_data.csv"
            if not file_path.exists():
                print(f"  ОШИБКА: Не найден файл {file_path}")
                return pd.DataFrame()
            df_monthly = pd.read_csv(file_path)
        else:
            raise ValueError(f"Неизвестный dataset_name: {dataset_name}")

        df_monthly['Date'] = pd.to_datetime(df_monthly[['Year', 'Month']].assign(day=1))
        df_monthly = df_monthly[df_monthly['Year'] >= 1960]
        df_monthly = df_monthly.dropna(subset=['DTR', 'Precip', 'Latitude', 'Longitude'])
        df_monthly = df_monthly.sort_values(by=['WMO', 'Date']).reset_index(drop=True)
        return df_monthly

    def _prepare_features_df(self, df):
        """Добавление гармонических сезонных признаков."""
        df = df.copy()
        df['Month_sin'] = np.sin(2 * np.pi * df['Date'].dt.month / 12)
        df['Month_cos'] = np.cos(2 * np.pi * df['Date'].dt.month / 12)
        return df

    def _build_sequences(self, df, target_var):
        """Формирует входные окна из непрерывных месяцев перед целевой датой."""
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

            n_records = len(w_feats)
            if n_records <= self.lookback:
                continue

            # Проверяем все переходы от начала окна до целевого месяца.
            consecutive_months = np.diff(months_all[mask]) == 1
            steps = np.lib.stride_tricks.sliding_window_view(consecutive_months, self.lookback)
            for i in np.flatnonzero(steps.all(axis=1)) + self.lookback:
                X_list.append(w_feats[i - self.lookback:i])
                y_list.append(w_targets[i])
                date_list.append(w_dates[i])
                wmo_list.append(w)
                lat_list.append(w_lats[i])
                lon_list.append(w_lons[i])

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
        y_train_scaled = y_scaler.fit_transform(y_train_proc.reshape(-1, 1)).flatten()

        # Создание DataLoader
        tensor_x = torch.tensor(X_train_scaled, dtype=torch.float32)
        tensor_y = torch.tensor(y_train_scaled, dtype=torch.float32)
        dataset = TensorDataset(tensor_x, tensor_y)
        loader = DataLoader(dataset, batch_size=batch_size, shuffle=True, drop_last=False)

        # Модель
        model = ClimateLSTM(input_dim=n_feat, hidden_dim=64, num_layers=2, dropout=0.2).to(self.device)
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
        pred_scaled = y_scaler.inverse_transform(raw_out.reshape(-1, 1)).flatten()
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
        print(f" БЛОК 1: СТАНДАРТНЫЙ ПРОГНОЗ ({dataset_name} | Год прогноза: {target_year})")
        print("="*80)

        t_load_start = time.perf_counter()
        df_full = self.load_data(dataset_name)
        t_load_end = time.perf_counter()
        print(f"  [Время загрузки данных и базовой предобработки: {t_load_end - t_load_start:.3f} сек]")

        if df_full.empty:
            return

        STD_CSV_DIR.mkdir(parents=True, exist_ok=True)
        for target_var in ['DTR', 'Precip']:
            fname = f"{dataset_name}_{target_var}_{target_year}.csv"
            print(f"\n---> Параметр: {target_var} <---")
            print(f"\n  * Модель: LSTM Neural Network")

            # Формируем все временные окна
            X_all, y_all, dates_all, wmo_all, lats_all, lons_all = self._build_sequences(df_full, target_var)

            all_res_rows = []
            total_fit_time = 0.0
            total_pred_time = 0.0
            total_samples = 0
            all_preds = []
            all_facts = []

            # Пошаговый помесячный Walk-Forward для 12 месяцев целевого года
            for month_num in range(1, 13):
                forecast_date = pd.to_datetime(f"{target_year}-{month_num:02d}-01")

                train_mask = (dates_all < forecast_date)
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
                preds = self._predict_lstm(model, feat_scaler, y_scaler, X_test, target_var)
                t_pred_end = time.perf_counter()
                pred_t = t_pred_end - t_pred_start

                total_fit_time += fit_t
                total_pred_time += pred_t
                total_samples += len(preds)

                all_preds.extend(preds)
                all_facts.extend(y_test)

                # Сохранение результатов шага
                for i in range(len(preds)):
                    all_res_rows.append({
                        'Date': forecast_date.strftime('%Y-%m-%d'),
                        'WMO': test_wmo[i],
                        'Latitude': test_lat[i],
                        'Longitude': test_lon[i],
                        'Forecast': round(float(preds[i]), 4),
                        'Fact': round(float(y_test[i]), 4),
                        'Model': 'LSTM Neural Network',
                        'Target_Var': target_var
                    })

            # Метрики
            clean_facts = np.array(all_facts)
            clean_preds = np.array(all_preds)
            r2 = r2_score(clean_facts, clean_preds) if clean_facts.size > 1 else np.nan
            mae = mean_absolute_error(clean_facts, clean_preds) if clean_facts.size else np.nan
            rmse = np.sqrt(mean_squared_error(clean_facts, clean_preds)) if clean_facts.size else np.nan

            speed_samples_sec = total_samples / total_pred_time if total_pred_time > 0 else 0.0
            total_time = total_fit_time + total_pred_time

            print(f"    - Время обучения (Fit Time):      {total_fit_time:.4f} сек")
            print(f"    - Время прогноза (Predict Time):  {total_pred_time:.4f} сек")
            print(f"    - Суммарное время (Total Time):   {total_time:.4f} сек")
            print(f"    - Скорость вычислений:            {speed_samples_sec:.2f} записей/сек ({total_pred_time/12:.4f} сек/месяц)")
            print(f"    - Метрики точности:               R² = {r2:.4f} | MAE = {mae:.4f} | RMSE = {rmse:.4f}")

            # Сохранение CSV
            df_out = pd.DataFrame(all_res_rows, columns=FORECAST_COLUMNS)
            df_out.to_csv(STD_CSV_DIR / fname, index=False)

            self.summary_table.append({
                'Блок': 'Стандартный прогноз',
                'Датасет': dataset_name,
                'Параметр': target_var,
                'Модель': 'LSTM Neural Network',
                'Fit Time (с)': round(total_fit_time, 4),
                'Predict Time (с)': round(total_pred_time, 4),
                'Total Time (с)': round(total_time, 4),
                'Скорость (зап/с)': round(speed_samples_sec, 2),
                'R²': round(r2, 4),
                'MAE': round(mae, 4),
                'RMSE': round(rmse, 4)
            })

    # ==========================================================================
    # 2. ДОЛГОСРОЧНАЯ ВАЛИДАЦИЯ (1990-2020 BACKTESTING)
    # ==========================================================================
    def run_validation(self, split_year, dataset_name, end_year=2020):
        print("\n" + "="*80)
        print(f" БЛОК 2: ДОЛГОСРОЧНАЯ ВАЛИДАЦИЯ ({dataset_name} | Период: {split_year}-{end_year})")
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

        VAL_CSV_DIR.mkdir(parents=True, exist_ok=True)
        for target_var in ['DTR', 'Precip']:
            fname = f"Validation_{dataset_name}_{target_var}_{split_year}-{final_year}.csv"
            print(f"\n---> Параметр: {target_var} <---")
            print(f"\n  * Модель: LSTM Neural Network")

            # Формируем все окна
            X_all, y_all, dates_all, wmo_all, lats_all, lons_all = self._build_sequences(df_full, target_var)
            dates_dt = pd.to_datetime(dates_all)

            all_res_rows = []
            total_fit_time = 0.0
            total_pred_time = 0.0
            total_samples = 0
            all_preds = []
            all_facts = []

            # Годовой Expanding Walk-Forward
            for year in test_years:
                split_date = pd.to_datetime(f"{year}-01-01")
                next_year_date = pd.to_datetime(f"{year + 1}-01-01")

                train_mask = (dates_dt < split_date)
                test_mask = (dates_dt >= split_date) & (dates_dt < next_year_date)

                if not np.any(train_mask) or not np.any(test_mask):
                    continue

                X_train, y_train = X_all[train_mask], y_all[train_mask]
                X_test, y_test = X_all[test_mask], y_all[test_mask]
                test_dates = dates_all[test_mask]
                test_wmo = wmo_all[test_mask]
                test_lat = lats_all[test_mask]
                test_lon = lons_all[test_mask]

                # 1. Обучение сети
                t_fit_start = time.perf_counter()
                model, feat_scaler, y_scaler = self._train_lstm_model(X_train, y_train, target_var, epochs=15)
                t_fit_end = time.perf_counter()
                fit_t = t_fit_end - t_fit_start

                # 2. Инференс
                t_pred_start = time.perf_counter()
                preds = self._predict_lstm(model, feat_scaler, y_scaler, X_test, target_var)
                t_pred_end = time.perf_counter()
                pred_t = t_pred_end - t_pred_start

                total_fit_time += fit_t
                total_pred_time += pred_t
                total_samples += len(preds)

                all_preds.extend(preds)
                all_facts.extend(y_test)

                # Запись строк
                for i in range(len(preds)):
                    d_val = pd.to_datetime(test_dates[i]).strftime('%Y-%m-%d')
                    all_res_rows.append({
                        'Date': d_val,
                        'WMO': test_wmo[i],
                        'Latitude': test_lat[i],
                        'Longitude': test_lon[i],
                        'Forecast': round(float(preds[i]), 4),
                        'Fact': round(float(y_test[i]), 4),
                        'Model': 'LSTM Neural Network',
                        'Target_Var': target_var
                    })

            # Итоговые метрики за 30 лет
            clean_facts = np.array(all_facts)
            clean_preds = np.array(all_preds)
            r2 = r2_score(clean_facts, clean_preds) if clean_facts.size > 1 else np.nan
            mae = mean_absolute_error(clean_facts, clean_preds) if clean_facts.size else np.nan
            rmse = np.sqrt(mean_squared_error(clean_facts, clean_preds)) if clean_facts.size else np.nan

            speed_samples_sec = total_samples / total_pred_time if total_pred_time > 0 else 0.0
            total_time = total_fit_time + total_pred_time

            print(f"    - Время обучения (Fit Time):      {total_fit_time:.4f} сек")
            print(f"    - Время прогноза (Predict Time):  {total_pred_time:.4f} сек")
            print(f"    - Суммарное время (Total Time):   {total_time:.4f} сек")
            print(f"    - Скорость вычислений:            {speed_samples_sec:.2f} записей/сек ({total_pred_time/(len(test_years)*12):.4f} сек/месяц)")
            print(f"    - Метрики точности:               R² = {r2:.4f} | MAE = {mae:.4f} | RMSE = {rmse:.4f}")

            # Сохранение CSV
            df_out = pd.DataFrame(all_res_rows, columns=FORECAST_COLUMNS)
            df_out.to_csv(VAL_CSV_DIR / fname, index=False)

            self.summary_table.append({
                'Блок': 'Долгосрочная валидация',
                'Датасет': dataset_name,
                'Параметр': target_var,
                'Модель': 'LSTM Neural Network',
                'Fit Time (с)': round(total_fit_time, 4),
                'Predict Time (с)': round(total_pred_time, 4),
                'Total Time (с)': round(total_time, 4),
                'Скорость (зап/с)': round(speed_samples_sec, 2),
                'R²': round(r2, 4),
                'MAE': round(mae, 4),
                'RMSE': round(rmse, 4)
            })

    def print_final_summary(self):
        """Печать и сохранение сводной таблицы производительности и точности."""
        print("\n" + "="*115)
        print("                           СВОДНАЯ ТАБЛИЦА РЕЗУЛЬТАТОВ ИСПЫТАНИЙ LSTM")
        print("="*115)
        df_summary = pd.DataFrame(self.summary_table)
        print(df_summary.to_string(index=False))
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

    cru_file = PROJECT_DIR / "cru ceda" / "cru_aligned_data.csv"
    if not cru_file.exists():
        CruDataLoader(PROJECT_DIR).run()

    # 2. Инициализация движка
    engine = LSTMFORECASTENGINE(base_dir=PROJECT_DIR, lookback_window=12)

    # 3. Стандартный прогноз на 1 год
    engine.run_standard_forecast(target_year=2024, dataset_name='METEO_RU')
    engine.run_standard_forecast(target_year=2020, dataset_name='CRU_CEDA')

    # 4. Долгосрочная валидация 1990-2020
    engine.run_validation(split_year=1990, dataset_name='METEO_RU', end_year=2020)
    engine.run_validation(split_year=1990, dataset_name='CRU_CEDA', end_year=2020)

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
