# -*- coding: utf-8 -*-
from pathlib import Path
import sys

current_dir = Path(__file__).resolve().parent
sys.path.append(str(current_dir))

from kursach011_modules.meteo_loader import SteppeClimateLoader
from kursach011_modules.cru_loader import CruDataLoader
from kursach011_modules.forecast_engine import StandardForecaster, LongTermValidator
from kursach011_modules.visualizer import StandardForecastVisualizer, ValidationVisualizer

PROJECT_DIR = Path(__file__).resolve().parent.parent
RESULTS_DIR = PROJECT_DIR / "output011_modules"

def main():
    print("\n>>> СТАРТ ИНФОРМАЦИОННОЙ СИСТЕМЫ <<<\n")
    
    # 1. ETL (Если нужно)
    meteo_file = Path(PROJECT_DIR) / "meteo_ru" / "all_steppe_data.txt"
    if not meteo_file.exists():
        SteppeClimateLoader(PROJECT_DIR).run()
    
    cru_file = Path(PROJECT_DIR) / "cru ceda" / "cru_aligned_data.csv"
    if not cru_file.exists():
        CruDataLoader(PROJECT_DIR).run()
    
    # 2. ПРОГНОЗЫ
    print("\n" + "="*60)
    print("БЛОК 1: РАСЧЕТ ПРОГНОЗОВ")
    print("="*60)
    
    # Standard Forecast
    forecaster = StandardForecaster(base_dir=PROJECT_DIR, root_output_dir=RESULTS_DIR)
    forecaster.run(target_year=2024, dataset_name='METEO_RU')
    forecaster.run(target_year=2020, dataset_name='CRU_CEDA')
    
    # Validation
    validator = LongTermValidator(base_dir=PROJECT_DIR, root_output_dir=RESULTS_DIR)
    validator.run(split_year=1990, dataset_name='METEO_RU', end_year=2020)
    validator.run(split_year=1990, dataset_name='CRU_CEDA', end_year=2020)
    
    # 3. ВИЗУАЛИЗАЦИЯ
    print("\n" + "="*60)
    print("БЛОК 2: ГЕНЕРАЦИЯ ОТЧЕТОВ И КАРТ")
    print("="*60)
    
    # Viz Standard
    viz_std = StandardForecastVisualizer(base_dir=PROJECT_DIR, results_dir=RESULTS_DIR)
    viz_std.run()
    
    # Viz Validation
    viz_val = ValidationVisualizer(base_dir=PROJECT_DIR, results_dir=RESULTS_DIR)
    viz_val.run()

    print(f"\n>>> ГОТОВО! Все результаты в папке: {RESULTS_DIR} <<<")

if __name__ == "__main__":
    main()