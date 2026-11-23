# -*- coding: utf-8 -*-
from pathlib import Path
import sys

current_dir = Path(__file__).resolve().parent
sys.path.append(str(current_dir))

from kursach011_modules.meteo_loader import SteppeClimateLoader
from kursach011_modules.forecast_engine import StandardForecaster, LongTermValidator
from kursach011_modules.project_settings import ANALYSIS_YEAR, VALIDATION_START_YEAR, VALIDATION_END_YEAR, results_dir
from kursach011_modules.visualizer import StandardForecastVisualizer, ValidationVisualizer
from prepare_cru import ensure_cru_csv

PROJECT_DIR = Path(__file__).resolve().parent.parent
RESULTS_DIR = results_dir(PROJECT_DIR)

def main():
    print("\n>>> СТАРТ ИНФОРМАЦИОННОЙ СИСТЕМЫ <<<\n")
    
    # 1. ETL (Если нужно)
    meteo_file = Path(PROJECT_DIR) / "meteo_ru" / "all_steppe_data.txt"
    if not meteo_file.exists():
        SteppeClimateLoader(PROJECT_DIR).run()
    
    ensure_cru_csv(PROJECT_DIR)
    
    # 2. ПРОГНОЗЫ
    print("\n" + "="*60)
    print("БЛОК 1: РАСЧЕТ ПРОГНОЗОВ")
    print("="*60)
    
    # Standard Forecast
    forecaster = StandardForecaster(base_dir=PROJECT_DIR, root_output_dir=RESULTS_DIR)
    forecaster.run(target_year=ANALYSIS_YEAR, dataset_name='METEO_RU')
    forecaster.run(target_year=ANALYSIS_YEAR, dataset_name='CRU_CEDA')
    
    # Validation
    validator = LongTermValidator(base_dir=PROJECT_DIR, root_output_dir=RESULTS_DIR)
    validator.run(split_year=VALIDATION_START_YEAR, dataset_name='METEO_RU', end_year=VALIDATION_END_YEAR)
    validator.run(split_year=VALIDATION_START_YEAR, dataset_name='CRU_CEDA', end_year=VALIDATION_END_YEAR)
    
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
