# Predict Customer Churn

- Project **Predict Customer Churn** of ML DevOps Engineer Nanodegree Udacity

## Project Description
This project implements a machine learning pipeline to identify credit card customers who are most likely to churn. 


## Files and Data Description
* **churn_library.py**: The core library containing functions for data processing and model training.
* **churn_script_logging_and_tests.py**: The test suite and logging script to verify library functionality.
* **data/**: Contains the `bank_data.csv` dataset.
* **images/**: 
    * **eda/**: Stores plots from exploratory data analysis.
    * **results/**: Stores model evaluation plots (ROC curves, classification reports).
* **models/**: Stores serialized model files (`.pkl`).
* **logs/**: Contains `churn_library.log` generated during execution.


## Running the Project
To install dependencies from requirements.txt:
```bash
pip install -r requirements.txt
```
To run the main library and generate models/plots:
```bash
python churn_library.py
```
To run the tests:
```bash
pytest churn_script_logging_and_tests.py
```
or
```bash
python -m pytest churn_script_logging_and_tests.py
```
when in a venv





