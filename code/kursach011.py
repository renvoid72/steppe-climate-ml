# -*- coding: utf-8 -*-
import argparse
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

def main(argv=None):
    parser = argparse.ArgumentParser(description='Рассчитывает прогнозы, валидацию и карты.')
    parser.add_argument('--force', action='store_true', help='Пересчитать прогнозы и валидацию при наличии готовых CSV.')
    args = parser.parse_args(argv)

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
    datasets = ('METEO_RU', 'CRU_CEDA')
    forecaster = StandardForecaster(base_dir=PROJECT_DIR, root_output_dir=RESULTS_DIR)
    for dataset in datasets:
        forecaster.run(target_year=ANALYSIS_YEAR, dataset_name=dataset, force=args.force)
        forecaster.run(target_year=ANALYSIS_YEAR - 1, dataset_name=dataset, target_month=12, force=args.force)
    
    # Validation
    validator = LongTermValidator(base_dir=PROJECT_DIR, root_output_dir=RESULTS_DIR)
    validation_end_years = {
        dataset: validator.run(VALIDATION_START_YEAR, dataset, VALIDATION_END_YEAR, force=args.force)
        for dataset in datasets
    }
    
    # 3. ВИЗУАЛИЗАЦИЯ
    print("\n" + "="*60)
    print("БЛОК 2: ГЕНЕРАЦИЯ ОТЧЕТОВ И КАРТ")
    print("="*60)
    
    # Viz Standard
    viz_std = StandardForecastVisualizer(base_dir=PROJECT_DIR, results_dir=RESULTS_DIR)
    viz_std.run()
    
    # Viz Validation
    viz_val = ValidationVisualizer(base_dir=PROJECT_DIR, results_dir=RESULTS_DIR)
    viz_val.run(end_year_by_dataset=validation_end_years)

    print(f"\n>>> ГОТОВО! Все результаты в папке: {RESULTS_DIR} <<<")

if __name__ == "__main__":
    main()
