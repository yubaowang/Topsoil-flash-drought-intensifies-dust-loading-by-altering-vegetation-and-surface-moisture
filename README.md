# Topsoil-flash-drought-intensifies-dust-loading-by-altering-vegetation-and-surface-moisture

# Code Description

This repository contains the core scripts used for soil moisture percentile calculation, topsoil flash drought identification, and Random Forest modelling.

## Files

### `Calculating_Soil_Moisture_percentiles.ipynb`

This notebook calculates soil moisture percentiles. It is used to derive percentile-based thresholds from soil moisture data, which provide the basis for identifying anomalously dry soil moisture conditions.

### `Topsoil_flash_drought_identification.ipynb`

This notebook identifies topsoil flash drought events. Based on the soil moisture percentile information, it detects events characterized by rapid soil moisture depletion and sustained dry conditions.

### `RF_Model_main.py`

This script contains the main workflow for the Random Forest modelling analysis. It is responsible for data loading, preprocessing, model training, cross-validation, prediction, and model interpretation.

### `RF_workers.py`

This script provides auxiliary worker functions used by `RF_Model_main.py`. It supports parallel or distributed computation tasks, including model-related calculations and interpretation procedures.
