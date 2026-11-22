# -*- coding: utf-8 -*-
"""
Модуль сбора данных (ETL).
Версия: 2.6 (Fix: Гарантированное заполнение WMO ID)

ИЗМЕНЕНИЯ:
- Изменен способ создания DataFrame: теперь используется словарь (dict).
  Это гарантирует, что ID станции (WMO) пропишется в КАЖДОЙ строке
  и не потеряется при сохранении.
"""
import matplotlib
matplotlib.use('Agg')

import geopandas as gpd
import pandas as pd
import numpy as np
from meteostat import Stations, Daily
from shapely.geometry import Point
from datetime import datetime
from pathlib import Path
import warnings
import os
import matplotlib.pyplot as plt
import time 

warnings.filterwarnings('ignore')

class SteppeClimateLoader:
    def __init__(self, base_dir):
        self.base_dir = Path(base_dir)
        
        # --- НАСТРОЙКИ ПУТЕЙ ---
        self.path_eco = self.base_dir / "maps_ecoregions" / "Ecoregions2017.shp"
        if not self.path_eco.exists():
            self.path_eco = self.base_dir / "maps" / "Ecoregions2017.shp"

        self.path_countries = self.base_dir / "maps_countries" / "ne_10m_admin_0_countries.shp"
        if not self.path_countries.exists():
             self.path_countries = self.base_dir / "maps" / "ne_10m_admin_0_countries.shp"
             
        self.out_dir = self.base_dir / "meteo_ru"
        self.out_dir.mkdir(parents=True, exist_ok=True)
        
        self.file_data = self.out_dir / "all_steppe_data.txt"      
        self.file_coords = self.out_dir / "stations_coords.txt"    
        self.file_map_png = self.out_dir / "stations_final_map.png"
        self.file_map_tiff = self.out_dir / "stations_final_map.tif"

        self.start = datetime(1960, 1, 1)
        self.end = datetime(2024, 12, 31)

        self.steppe_mask = None
        self.target_stations = None
        self.final_data = pd.DataFrame()
        self.final_meta = pd.DataFrame()

    def step_1_create_mask(self):
        print("\n[ШАГ 1] Гео-анализ: Создание маски степей...")
        
        if not self.path_eco.exists() or not self.path_countries.exists():
            raise FileNotFoundError("Не найдены файлы карт!")

        print("  - Загрузка карт...")
        eco_gdf = gpd.read_file(self.path_eco)
        countries = gpd.read_file(self.path_countries)
        
        biome_col = 'BIOME_NUM' if 'BIOME_NUM' in eco_gdf.columns else 'BIOME'
        steppes = eco_gdf[eco_gdf[biome_col] == 8]
        russia = countries[countries['NAME'] == 'Russia']
        
        print("  - Обрезка степей по границам РФ...")
        if steppes.crs != russia.crs:
            steppes = steppes.to_crs(russia.crs)
            
        self.steppe_mask = gpd.overlay(steppes, russia, how='intersection')
        print("  -> Маска готова.")

    def step_2_find_stations(self):
        print("\n[ШАГ 2] Поиск метеостанций...")
        stations = Stations()
        stations = stations.region('RU')
        stations = stations.inventory('daily', (pd.to_datetime('1980-01-01'), pd.to_datetime('2010-01-01')))
        df_stations = stations.fetch()
        
        geometry = [Point(xy) for xy in zip(df_stations['longitude'], df_stations['latitude'])]
        gdf_stations = gpd.GeoDataFrame(df_stations, geometry=geometry, crs=self.steppe_mask.crs)
        
        print("  - Фильтрация станций по маске степей...")
        stations_in_steppe = gpd.sjoin(gdf_stations, self.steppe_mask, how="inner", predicate="intersects")
        
        stations_in_steppe = stations_in_steppe[~stations_in_steppe.index.duplicated(keep='first')]
        self.target_stations = stations_in_steppe[stations_in_steppe.index.str.len() == 5]
        
        print(f"  -> Найдено {len(self.target_stations)} станций в целевой зоне.")

    def step_3_download_data(self):
        print("\n[ШАГ 3] Скачивание и обработка данных...")
        
        all_dfs = []
        valid_meta = []
        
        total = len(self.target_stations)
        count = 0
        MAX_RETRIES = 3
        
        for wmo_id, row in self.target_stations.iterrows():
            count += 1
            print(f"  [{count}/{total}] {row['name']} ({wmo_id})...", end=" ", flush=True)
            
            df = pd.DataFrame()
            download_success = False

            for attempt in range(MAX_RETRIES):
                try:
                    data = Daily(wmo_id, self.start, self.end)
                    df = data.fetch()
                    download_success = True
                    break 
                except Exception as e:
                    if attempt < MAX_RETRIES - 1:
                        time.sleep(1)
                    else:
                        print(f" -> Ошибка сети: {e}")
            
            if not download_success:
                print(" -> ПРОПУСК")
                continue

            try:
                if df.empty or len(df) < 3650:
                    print("Мало данных -> Skip")
                    continue
                
                has_dtr = df['tmin'].notna() & df['tmax'].notna()
                usable_days = df['tavg'].notna() | df['prcp'].notna() | has_dtr
                df = df.loc[usable_days].reset_index()
                
                # ===> ИСПРАВЛЕНИЕ: Создаем через словарь <===
                # Это гарантирует, что WMO растянется на все строки
                data_dict = {
                    'WMO': int(wmo_id),
                    'Year': df['time'].dt.year,
                    'Month': df['time'].dt.month,
                    'Day': df['time'].dt.day,
                    'Quality': 0,
                    'Tmin': df['tmin'].fillna(-999.9),
                    'Tavg': df['tavg'].fillna(-999.9),
                    'Tmax': df['tmax'].fillna(-999.9),
                    'Precip': df['prcp']
                }
                
                clean = pd.DataFrame(data_dict)
                
                # Округление
                cols_round = ['Tmin', 'Tavg', 'Tmax', 'Precip']
                clean[cols_round] = clean[cols_round].round(1)
                
                # Типы данных (для надежности)
                clean = clean.astype({'WMO': int, 'Year': int, 'Month': int, 'Day': int, 'Quality': int})

                if not clean.empty:
                    all_dfs.append(clean)
                    valid_meta.append({
                        'WMO': int(wmo_id),
                        'Name': row['name'],
                        'Lat': round(row['latitude'], 4),
                        'Lon': round(row['longitude'], 4)
                    })
                    print("ОК")
                else:
                    print("Пусто после очистки -> Skip")

            except Exception as e:
                print(f"Ошибка обработки: {e}")
                
        self.final_data = pd.concat(all_dfs) if all_dfs else pd.DataFrame()
        self.final_meta = pd.DataFrame(valid_meta)

    def step_4_save(self):
        print("\n[ШАГ 4] Сохранение...")
        
        if self.final_data.empty:
            print("ОШИБКА: Нет данных для сохранения.")
            return

        # Гарантируем порядок колонок
        cols_order = ['WMO', 'Year', 'Month', 'Day', 'Quality', 'Tmin', 'Tavg', 'Tmax', 'Precip']
        self.final_data = self.final_data[cols_order]
        
        self.final_data = self.final_data.sort_values(by=['WMO', 'Year', 'Month', 'Day'])
        
        # Сохранение
        self.final_data.to_csv(self.file_data, sep=' ', index=False, header=False, na_rep='-999.9')
        self.final_meta.to_csv(self.file_coords, sep=' ', index=False)
        
        # ПРОВЕРКА: Выводим первые 5 строк в консоль, чтобы ты увидел, что WMO на месте
        print("\n--- ПРОВЕРКА (Первые 5 строк) ---")
        print(self.final_data.head().to_string(index=False, header=False))
        print("---------------------------------\n")
        
        print(f"  -> Данные сохранены: {self.file_data}")
        print(f"  -> Координаты сохранены: {self.file_coords}")
        print(f"  -> Всего записей: {len(self.final_data)}")
        print(f"  -> Всего станций: {len(self.final_meta)}")

    def step_5_visualize(self):
        print("\n[ШАГ 5] Генерация карты с РЕАЛЬНЫМ рельефом (ETOPO1)...")
        if self.final_meta.empty: return

        try:
            import rioxarray
            from matplotlib.colors import LightSource, LinearSegmentedColormap
            HAS_RASTER = True
        except ImportError:
            HAS_RASTER = False
            print("  ! rioxarray не найден.")

        # Путь к ETOPO1
        relief_path = self.base_dir / "maps_relief" / "ETOPO1_Ice_g_geotiff.tif"
        
        gdf_final = gpd.GeoDataFrame(
            self.final_meta, 
            geometry=gpd.points_from_xy(self.final_meta.Lon, self.final_meta.Lat),
            crs="EPSG:4326"
        )

        fig, ax = plt.subplots(figsize=(20, 8))

        # 1. РЕЛЬЕФ (ETOPO1)
        if HAS_RASTER and relief_path.exists():
            try:
                print("  - Загрузка и обработка ETOPO1...")
                # Открываем
                rds = rioxarray.open_rasterio(relief_path, chunks=True, masked=True)
                rds.rio.write_crs("EPSG:4326", inplace=True)
                
                # Обрезаем СРАЗУ по bounding box
                minx, miny, maxx, maxy = self.steppe_mask.total_bounds
                rds_small = rds.rio.clip_box(minx=minx-5, miny=miny-5, maxx=maxx+5, maxy=maxy+5)
                
                # Теперь обрезаем точно по маске степей
                rds_clipped = rds_small.rio.clip(self.steppe_mask.geometry, self.steppe_mask.crs, drop=True)
                
                # Превращаем в numpy массив высот
                elev = rds_clipped.isel(band=0).values
                
                # --- ИСПРАВЛЕНИЕ: СВЕТЛАЯ "СТЕПНАЯ" ПАЛИТРА ---
                # Цвета: Светло-зеленый -> Пшеничный -> Песочный -> Коричневый -> Белый
                # Это сделает карту намного светлее и читаемее.
                colors = [
                    "#A8D08D",  # Светлый травяной (вместо темно-зеленого)
                    "#D8E6AD",  # Бледно-салатовый
                    "#F5F5DC",  # Бежевый (Beige)
                    "#EEDC82",  # Пшеничный/Песочный
                    "#D2B48C",  # Загар (Tan)
                    "#8D6E63",  # Светло-коричневый (Горы)
                    "#FFFFFF"   # Снег (Вершины)
                ]
                
                # Создаем карту цветов
                cmap_land = LinearSegmentedColormap.from_list("steppe_nature", colors)

                # --- 3D ЭФФЕКТ ---
                ls = LightSource(azdeg=315, altdeg=45)
                
                # Настраиваем диапазоны высот
                vmin, vmax = -50, 3500 
                
                # Заменяем NaN на 0 временно
                elev_filled = np.nan_to_num(elev, nan=vmin)
                
                # Создаем затененное изображение
                rgb = ls.shade(elev_filled, cmap=cmap_land, vmin=vmin, vmax=vmax, vert_exag=100, blend_mode='soft')
                
                # Если shade вернул RGBA, отрезаем альфу
                if rgb.ndim == 3 and rgb.shape[2] == 4:
                    rgb = rgb[:, :, :3]
                
                # Маска прозрачности (где нет данных)
                mask = np.isnan(elev)
                alpha = np.where(mask, 0, 1) # 0 - прозрачно
                
                # Собираем итоговый RGBA
                rgba = np.dstack((rgb, alpha))
                
                # Рисуем
                extent = [
                    rds_clipped.x.min().item(), rds_clipped.x.max().item(),
                    rds_clipped.y.min().item(), rds_clipped.y.max().item()
                ]
                
                ax.imshow(rgba, extent=extent, zorder=1)
                
                # === ЛЕГЕНДА ===
                sm = plt.cm.ScalarMappable(cmap=cmap_land, norm=plt.Normalize(vmin=0, vmax=vmax))
                sm.set_array([])
                cbar = plt.colorbar(sm, ax=ax, fraction=0.015, pad=0.02)
                cbar.set_label('Высота над уровнем моря, м', fontsize=12)
                
            except Exception as e:
                print(f"  ! Ошибка ETOPO1: {e}")
                self.steppe_mask.plot(ax=ax, color='wheat', alpha=0.5, zorder=1) # wheat светлее orange
        else:
            if not relief_path.exists():
                print(f"  ! Файл ETOPO1 не найден: {relief_path}")
            self.steppe_mask.plot(ax=ax, color='wheat', alpha=0.5, zorder=1)

        # 3. ГРАНИЦЫ СТЕПЕЙ (Коричневый контур)
        self.steppe_mask.plot(ax=ax, color='none', edgecolor='#8B4513', linewidth=0.6, zorder=2)
        
        # 4. СТАНЦИИ
        # ИСПРАВЛЕНИЕ: Добавляем edgecolor='white', чтобы черные точки были видны на любом фоне
        gdf_final.plot(
            ax=ax, 
            color='black',       # Сами точки черные
            edgecolor='white',   # Белая обводка для контраста
            linewidth=0.8,       # Толщина обводки
            markersize=25,       # Чуть крупнее
            label='Метеостанции', 
            zorder=3
        )

        minx, miny, maxx, maxy = self.steppe_mask.total_bounds
        ax.set_xlim(minx - 2, maxx + 2)
        ax.set_ylim(miny - 1, maxy + 1)

        plt.title(f"Итоговый набор данных: {len(gdf_final)} станций (1960-2024)", fontsize=20, fontweight='bold')
        plt.xlabel("Долгота", fontsize=12)
        plt.ylabel("Широта", fontsize=12)
        plt.legend(loc='upper right', fontsize=12)
        
        ax.grid(True, linestyle=':', alpha=0.4, color='black')

        # Сохраняем PNG (для быстрого просмотра)
        plt.savefig(self.file_map_png, dpi=300, bbox_inches='tight')
        
        # Сохраняем TIFF 
        plt.savefig(
            self.file_map_tiff, 
            dpi=300, 
            bbox_inches='tight', 
            format='tiff', 
            pil_kwargs={'compression': 'tiff_lzw'}
        )
        plt.close()
        print(f"  -> Карты сохранены: .png и .tif")
    def run(self):
        print("=== ЗАПУСК СИСТЕМЫ СБОРА ДАННЫХ (ETL) ===")
        start_time = time.time()
        try:
            self.step_1_create_mask()
            self.step_2_find_stations()
            self.step_3_download_data()
            self.step_4_save()
            self.step_5_visualize()
            print(f"\n=== ГОТОВО (Время: {time.time() - start_time:.1f} сек) ===")
        except Exception as e:
            print(f"\n!!! КРИТИЧЕСКАЯ ОШИБКА: {e}")

if __name__ == "__main__":
    PROJECT_DIR = Path(__file__).resolve().parent.parent.parent
    loader = SteppeClimateLoader(PROJECT_DIR)
    loader.run()
