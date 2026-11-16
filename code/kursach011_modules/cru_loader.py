# -*- coding: utf-8 -*-
"""
Модуль обработки данных CRU CEDA.
Задачи:
1. Читает координаты станций, полученные на этапе Meteo ETL.
2. Извлекает данные из NetCDF файлов CRU (температура и осадки) для этих точек.
3. Формирует единый CSV файл, синхронизированный со станциями.
"""
import pandas as pd
import xarray as xr
from pathlib import Path
import warnings

# Отключаем предупреждения о датах (cftime)
warnings.filterwarnings('ignore')

class CruDataLoader:
    def __init__(self, base_dir):
        """
        Инициализация путей.
        base_dir: Корневая папка проекта
        """
        self.base_dir = Path(base_dir)
        
        # 1. Откуда берем координаты станций (результат работы Meteo Loader)
        self.coords_file = self.base_dir / "метео ру" / "stations_coords.txt"
        
        # 2. Где лежат исходные карты CRU (NetCDF)
        self.cru_dtr_path = self.base_dir / "cru ceda" / "temp dtr full rus stepp.nc"
        self.cru_pre_path = self.base_dir / "cru ceda" / "precip full rus stepp.nc"
        
        # 3. Куда сохраним результат (Таблица)
        self.out_file = self.base_dir / "cru ceda" / "cru_aligned_data.csv"
        
        self.stations_df = None
        self.final_df = pd.DataFrame()

    def load_coordinates(self):
        """Читает список станций, для которых нужно добыть данные"""
        print("\n[CRU] Шаг 1: Чтение координат станций...")
        
        if not self.coords_file.exists():
            raise FileNotFoundError(f"Файл координат не найден: {self.coords_file}\nСначала запустите Meteo Loader!")
            
        # Читаем txt с пробелами как разделителями
        self.stations_df = pd.read_csv(self.coords_file, sep=r'\s+')
        print(f"  -> Найдено {len(self.stations_df)} станций для обработки.")

    def extract_data(self):
        """
        Самая главная часть: 'Бурение' карт NetCDF.
        Для каждой точки (Lat/Lon) извлекаем временной ряд.
        """
        print("[CRU] Шаг 2: Извлечение данных из NetCDF (это может занять время)...")
        
        if not self.cru_dtr_path.exists() or not self.cru_pre_path.exists():
            raise FileNotFoundError("Не найдены файлы .nc в папке 'cru ceda'. Проверьте имена файлов!")

        # Открываем "лениво" (без загрузки всего файла в память)
        try:
            ds_dtr = xr.open_dataset(self.cru_dtr_path)
            ds_pre = xr.open_dataset(self.cru_pre_path)
        except Exception as e:
            raise RuntimeError(f"Ошибка открытия NetCDF: {e}")

        # Проверяем имена переменных внутри файлов (обычно это 'dtr' и 'pre')
        # Если имена другие, скрипт может упасть, но обычно в CRU TS они такие.
        var_dtr = 'dtr' if 'dtr' in ds_dtr else list(ds_dtr.data_vars)[0]
        var_pre = 'pre' if 'pre' in ds_pre else list(ds_pre.data_vars)[0]
        
        cru_data_list = []
        total = len(self.stations_df)
        
        # Проходим по каждой станции
        for i, row in self.stations_df.iterrows():
            wmo_id = int(row['WMO'])
            lat = row['Lat']
            lon = row['Lon']
            
            # Красивый вывод прогресса
            if i % 10 == 0:
                print(f"  -> Обработка станции {i+1}/{total}...", end='\r')

            try:
                # .sel(..., method='nearest') находит ближайший пиксель сетки к координатам станции
                dtr_series = ds_dtr[var_dtr].sel(lat=lat, lon=lon, method='nearest').to_pandas()
                pre_series = ds_pre[var_pre].sel(lat=lat, lon=lon, method='nearest').to_pandas()
                
                # Создаем мини-таблицу для одной станции
                df_station = pd.DataFrame({
                    'DTR': dtr_series,
                    'Precip': pre_series
                }).reset_index()
                
                # Переименовываем колонку с датой (xarray может назвать её 'time')
                df_station.rename(columns={'time': 'Date', 'index': 'Date'}, inplace=True)
                
                # Добавляем метаданные (чтобы знать, чьи это данные)
                df_station['WMO'] = wmo_id
                df_station['Latitude'] = lat
                df_station['Longitude'] = lon
                
                # Извлекаем Год и Месяц для удобства
                df_station['Year'] = df_station['Date'].dt.year
                df_station['Month'] = df_station['Date'].dt.month
                
                # Убираем пустые строки (если станция попала в океан или данные кончились)
                df_station = df_station.dropna(subset=['DTR', 'Precip'])
                
                cru_data_list.append(df_station)
                
            except Exception as e:
                print(f"\n  ! Ошибка для станции {wmo_id}: {e}")

        # Собираем всё в одну большую таблицу
        if cru_data_list:
            self.final_df = pd.concat(cru_data_list)
            print(f"\n  -> Успешно! Собрано {len(self.final_df)} строк данных.")
        else:
            print("\n  -> Ошибка: Не удалось извлечь данные ни для одной станции.")

    def save_data(self):
        """Сохраняет результат в CSV"""
        print("[CRU] Шаг 3: Сохранение...")
        
        if not self.final_df.empty:
            # Сортировка для порядка
            self.final_df = self.final_df.sort_values(by=['WMO', 'Year', 'Month'])
            
            # Сохраняем в CSV с заголовком (header=True) и запятыми
            self.final_df.to_csv(self.out_file, index=False)
            print(f"  -> Файл готов: {self.out_file}")
            print("  Теперь этот файл можно использовать в прогнозе вместо тяжелых NetCDF.")
        else:
            print("  -> Нечего сохранять.")

    def run(self):
        """Запуск модуля"""
        print("-" * 60)
        print("ЗАПУСК МОДУЛЯ ОБРАБОТКИ CRU CEDA")
        self.load_coordinates()
        self.extract_data()
        self.save_data()
        print("-" * 60)

# Для тестирования (если запускать этот файл отдельно)
if __name__ == "__main__":
    PROJECT_DIR = r"C:\Users\defaultuser0\Desktop\ucheba\3 kurs\kursach\proga i files"
    loader = CruDataLoader(PROJECT_DIR)
    loader.run()