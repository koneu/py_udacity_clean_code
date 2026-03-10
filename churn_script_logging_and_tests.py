"""
Test suite for the churn_library.py module.

Execution:
    pytest churn_script_logging_and_tests.py

Author: koneu
Creation Data: 9/3/2026
"""
import os
import logging
import pytest

import numpy as np
from sklearn.ensemble import RandomForestClassifier

import churn_library as cls

# changed config to avoid pytest taking over logging config
os.makedirs('./logs', exist_ok=True)
logger = logging.getLogger('churn_tests')
logger.setLevel(logging.INFO)
handler = logging.FileHandler('./logs/churn_library.log', mode='w')
handler.setFormatter(logging.Formatter(
    '%(name)s - %(levelname)s - %(message)s'))
logger.addHandler(handler)


def test_import():
    '''
    test data import - this example is completed for you to assist with the other test functions
    '''
    try:
        df = cls.import_data("./data/bank_data.csv")
        logger.info("Testing import_data: SUCCESS")
    except FileNotFoundError as err:
        logger.error("Testing import_eda: The file wasn't found")
        raise err

    try:
        assert df.shape[0] > 0
        assert df.shape[1] > 0
    except AssertionError as err:
        logger.error(
            "Testing import_data: The file doesn't appear to have rows and columns")
        raise err


@pytest.fixture(scope="module")
def dataframe():
    '''
    Fixture to provide the dataframe to all test
    '''
    df = cls.import_data("./data/bank_data.csv")
    # Ensure Churn column exists as it's required for EDA
    df['Churn'] = df['Attrition_Flag'].apply(
        lambda val: 0 if val == "Existing Customer" else 1
    )
    return df


def test_eda(dataframe, monkeypatch, tmp_path):
    '''
    test perform_eda with folder isolation and data validation
    '''
    # create test environment
    test_eda_dir = tmp_path / "eda_test"
    test_eda_dir.mkdir()
    monkeypatch.setattr(cls, "EDA_IMAGE_DIR", str(test_eda_dir))

    try:
        # --- execute code ---
        cls.perform_eda(dataframe)

        # validation
        expected_files = [
            "Churn_hist.png",
            "Customer_Age.png",
            "normalize_Marital_Status.png",
            "histplot_Total_Trans_Ct.png",
            "heatmap_corr.png"
        ]

        for file_name in expected_files:
            file_path = test_eda_dir / file_name
            assert file_path.exists()
            assert file_path.stat().st_size > 0

        logger.info(
            "SUCCESS: test_eda - Isolated run with data validation passed.")

    except AssertionError as err:
        logger.error(f"FAILURE: test_eda - {err}")
        raise err


def test_encoder_helper(dataframe):
    '''
    test encoder helper
    '''
    # create test environment
    cat_columns = ['Gender', 'Education_Level']
    response = 'Churn'
    try:
        # --- execute code ---
        encoded_df = cls.encoder_helper(dataframe, cat_columns, response)

        # validation
        # checkl response behaviour
        for col in cat_columns:
            encoded_col_name = f"{col}_{response}"
            assert encoded_col_name in encoded_df.columns
            assert encoded_df[encoded_col_name].dtype in [float, np.float64]

        # check random value
        test_category = 'F'
        actual_mean = dataframe[dataframe['Gender']
                                == test_category][response].mean()
        encoded_val = encoded_df[encoded_df['Gender'] ==
                                 test_category][f"Gender_{response}"].iloc[0]

        assert np.isclose(actual_mean, encoded_val)

        logger.info("SUCCESS: test_encoder_helper")

    except AssertionError as err:
        logger.error(
            "FAILURE: test_encoder_helper - Math mismatch or missing column")
        raise err


def test_perform_feature_engineering(dataframe):
    '''
    test perform_feature_engineering
    '''
    # create test environment
    cat_columns = [
        'Gender', 'Education_Level', 'Marital_Status',
        'Income_Category', 'Card_Category'
    ]
    encoded_df = cls.encoder_helper(dataframe, cat_columns, response='Churn')

    try:
        # --- execute code ---
        X_train, X_test, y_train, y_test = cls.perform_feature_engineering(
            encoded_df)

        # validation - no lost data
        assert len(X_train) + len(X_test) == len(dataframe)
        assert len(y_train) + len(y_test) == len(dataframe)
        assert list(X_train.columns) == list(X_test.columns)
        assert X_train.shape[0] == y_train.shape[0]
        assert X_test.shape[0] == y_test.shape[0]

        # validation No Leaks
        assert 'Churn' not in X_train.columns

        logger.info("SUCCESS: test_perform_feature_engineering")

    except AssertionError as err:
        logger.error(
            "FAILURE: test_perform_feature_engineering - Split or alignment error")
        raise err


def test_train_models(dataframe, monkeypatch, tmp_path):
    '''
    test train_models - verifies model saving and result image generation
    '''
    # create test environment
    # - image paths
    t_model_dir = tmp_path / "models"
    t_model_img_dir = tmp_path / "model_imgs"

    for folder in [t_model_dir, t_model_img_dir]:
        folder.mkdir()

    monkeypatch.setattr(cls, "MODEL_DIR", str(t_model_dir))
    monkeypatch.setattr(cls, "MODEL_IMAGE_DIR", str(t_model_img_dir))

    # - input data
    cat_columns = ['Gender', 'Education_Level', 'Marital_Status',
                   'Income_Category', 'Card_Category']
    encoded_df = cls.encoder_helper(dataframe, cat_columns, response='Churn')
    X_train, X_test, y_train, y_test = cls.perform_feature_engineering(
        encoded_df)

    try:
        # --- execute code ---
        cls.train_models(X_train, X_test, y_train, y_test)

        # validation models exist
        assert (t_model_dir / "rfc_model.pkl").exists()
        assert (t_model_dir / "logistic_model.pkl").exists()

        # validation plots exist
        assert (t_model_img_dir / "roc_lrc_plot.png").exists()
        assert (t_model_img_dir / "roc_rfc_disp.png").exists()

        logger.info("SUCCESS: test_train_models")
    except Exception as err:
        logger.error("FAILURE: test_train_models - %s", err)
        raise err


def test_classification_report_image(dataframe, monkeypatch, tmp_path):
    '''
    test classification_report_image - verifying plot generation from predictions
    '''
    # create test environment
    # - image paths
    t_report_dir = tmp_path / "reports"
    t_report_dir.mkdir()
    monkeypatch.setattr(cls, "REPORT_IMAGE_DIR", str(t_report_dir))

    # - fake input data
    y_train = [0, 1, 0, 1]
    y_test = [0, 1, 0, 1]
    y_train_preds_lr = [0, 1, 0, 0]
    y_test_preds_lr = [0, 1, 1, 0]
    y_train_preds_rf = [0, 1, 0, 1]
    y_test_preds_rf = [0, 1, 0, 1]

    try:
        # --- execute code ---
        cls.classification_report_image(
            y_train,
            y_test,
            y_train_preds_lr,
            y_train_preds_rf,
            y_test_preds_lr,
            y_test_preds_rf
        )

        # validation plots exist
        assert (t_report_dir / "forest_train.png").exists()
        assert (t_report_dir / "regression_train.png").exists()

        logger.info("SUCCESS: test_classification_report_image")
    except Exception as err:
        logger.error("FAILURE: test_classification_report_image - %s", err)
        raise err


def test_feature_importance_plot(dataframe, monkeypatch, tmp_path):
    '''
    test feature_importance_plot - verifying importance visualization
    '''
    # create test environment
    # - image paths
    t_importance_dir = tmp_path / "importance"
    t_importance_dir.mkdir()
    monkeypatch.setattr(cls, "IMPORTANCE_IMAGE_DIR", str(t_importance_dir))

    # - input data
    cat_columns = ['Gender', 'Education_Level', 'Marital_Status',
                   'Income_Category', 'Card_Category']
    encoded_df = cls.encoder_helper(dataframe, cat_columns, response='Churn')
    X_train, X_test, y_train, y_test = cls.perform_feature_engineering(
        encoded_df)

    # - tiny test model on tiny test data
    model = RandomForestClassifier(n_estimators=10)
    model.fit(X_train, y_train)

    try:
        # --- execute code ---
        cls.feature_importance_plot(model, X_test, "feature_importance_rfc")

        # validation plots exist
        assert (t_importance_dir / "feature_importance_rfc.png").exists()

        logger.info("SUCCESS: test_feature_importance_plot")
    except Exception as err:
        logger.error("FAILURE: test_feature_importance_plot - %s", err)
        raise err


if __name__ == "__main__":
    pass
