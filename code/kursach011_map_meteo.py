# -*- coding: utf-8 -*-
from pathlib import Path
import sys
import pandas as pd

# Добавляем путь к модулям
current_dir = Path(__file__).resolve().parent
sys.path.append(str(current_dir))

from kursach011_modules.meteo_loader import SteppeClimateLoader

# Корень проекта
PROJECT_DIR = Path(__file__).resolve().parent.parent

def main():
    print(">>> ЗАПУСК ТОЛЬКО ГЕНЕРАЦИИ КАРТЫ СТАНЦИЙ <<<\n")
    
    # Инициализируем класс
    loader = SteppeClimateLoader(PROJECT_DIR)
    
    # 1. Создаем маску (нужна для фона)
    loader.step_1_create_mask()
    
    # 2. Вместо скачивания (step_2 и step_3) просто читаем готовый файл координат
    print("  Чтение сохраненных координат...")
    if loader.file_coords.exists():
        # Читаем файл stations_coords.txt (разделитель пробел)
        loader.final_meta = pd.read_csv(loader.file_coords, sep=r'\s+')
        print(f"  -> Загружено {len(loader.final_meta)} станций из файла.")
        
        # 3. Запускаем только визуализацию
        loader.step_5_visualize()
        
        print("\n>>> ГОТОВО! Проверь файл stations_final_map.tif и .png <<<")
    else:
        print(f"ОШИБКА: Файл координат не найден: {loader.file_coords}")
        print("Сначала нужно хотя бы один раз прогнать полную загрузку.")

if __name__ == "__main__":
    main()