"""
A library of functions to predict customer churn using credit card customer data.

Author: koneu
Creation Data: 9/3/2026
"""

# import libraries
import os
import argparse
from pathlib import Path

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import joblib
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import GridSearchCV, train_test_split
from sklearn.metrics import RocCurveDisplay, classification_report

os.environ['QT_QPA_PLATFORM'] = 'offscreen'

EDA_IMAGE_DIR = os.path.join("images", "eda")
IMPORTANCE_IMAGE_DIR = os.path.join("images", "results")
MODEL_IMAGE_DIR = os.path.join("images", "results")
REPORT_IMAGE_DIR = os.path.join("images", "results")
MODEL_DIR = os.path.join("models")


def import_data(pth: str) -> pd.DataFrame:
    '''
    returns dataframe for the csv found at pth

    input:
            pth: a path to the csv
    output:
            df: pandas dataframe
    '''
    try:
        return pd.read_csv(pth, sep=',', engine='python')

    except FileNotFoundError:
        print(f"Error: The file at {pth} was not found.")

    return None


def perform_eda(df: pd.DataFrame):
    '''
    perform eda on df and save figures to images folder
    input:
            df: pandas dataframe

    output:
            None
    '''

    print("--- Shape & Info ---")
    print(df.info())

    print("\n--- Statistical Summary ---")
    print(df.describe())

    print("\n--- Missing Values ---")
    print(df.isnull().sum())

    print("\n--- Correlations ---")
    print(df.corr(numeric_only=True))

    pth = EDA_IMAGE_DIR
    plt.figure(figsize=(20, 10))
    df['Churn'].hist()
    plt.savefig(
        os.path.join(pth, "Churn_hist.png"),
        bbox_inches='tight',
        dpi=300)
    plt.close()

    plt.figure(figsize=(20, 10))
    df['Customer_Age'].hist()
    plt.savefig(
        os.path.join(pth, "Customer_Age.png"),
        bbox_inches='tight',
        dpi=300)
    plt.close()

    plt.figure(figsize=(20, 10))
    df.Marital_Status.value_counts('normalize').plot(kind='bar')
    plt.savefig(
        os.path.join(pth, "normalize_Marital_Status.png"),
        bbox_inches='tight',
        dpi=300)
    plt.close()

    plt.figure(figsize=(20, 10))
    sns.histplot(df['Total_Trans_Ct'], stat='density', kde=True)
    plt.savefig(
        os.path.join(pth, "histplot_Total_Trans_Ct.png"),
        bbox_inches='tight',
        dpi=300)
    plt.close()

    plt.figure(figsize=(20, 10))
    sns.heatmap(
        df.corr(
            numeric_only=True),
        annot=False,
        cmap='Dark2_r',
        linewidths=2)
    plt.savefig(
        os.path.join(pth, "heatmap_corr.png"),
        bbox_inches='tight',
        dpi=300)
    plt.close()


def encoder_helper(
        df: pd.DataFrame, category_lst: list[str], response: str = None) -> pd.DataFrame:
    '''
    helper function to turn each categorical column into a new column with
    propotion of churn for each category - associated with cell 15 from the notebook

    input:
            df: pandas dataframe
            category_lst: list of columns that contain categorical features
            response: string of response name [optional argument that could be used for naming variables or index y column]

    output:
            df: pandas dataframe with new columns for
    '''
    target = response if response else 'Churn'

    for category in category_lst:
        group_means = df.groupby(category)[target].mean()
        df[f'{category}_Churn'] = df[category].map(group_means)

    return df


def perform_feature_engineering(df: pd.DataFrame, response: str = None):
    '''
    input:
              df: pandas dataframe
              response: string of response name [optional argument that could be used for naming variables or index y column]

    output:
              X_train: X training data
              X_test: X testing data
              y_train: y training data
              y_test: y testing data
    '''
    target = response if response else 'Churn'

    keep_cols = [
        'Customer_Age', 'Dependent_count', 'Months_on_book',
        'Total_Relationship_Count', 'Months_Inactive_12_mon',
        'Contacts_Count_12_mon', 'Credit_Limit', 'Total_Revolving_Bal',
        'Avg_Open_To_Buy', 'Total_Amt_Chng_Q4_Q1', 'Total_Trans_Amt',
        'Total_Trans_Ct', 'Total_Ct_Chng_Q4_Q1', 'Avg_Utilization_Ratio',
        'Gender_Churn', 'Education_Level_Churn', 'Marital_Status_Churn',
        'Income_Category_Churn', 'Card_Category_Churn'
    ]

    y = df[target]
    X = df[keep_cols]

    return train_test_split(X, y, test_size=0.3, random_state=42)


def classification_report_image(y_train,
                                y_test,
                                y_train_preds_lr,
                                y_train_preds_rf,
                                y_test_preds_lr,
                                y_test_preds_rf):
    '''
    produces classification report for training and testing results and stores report as image
    in images folder
    input:
            y_train: training response values
            y_test:  test response values
            y_train_preds_lr: training predictions from logistic regression
            y_train_preds_rf: training predictions from random forest
            y_test_preds_lr: test predictions from logistic regression
            y_test_preds_rf: test predictions from random forest

    output:
             None
    '''
    pth = REPORT_IMAGE_DIR

    plt.figure(figsize=(5, 5))
    # plt.text(0.01, 0.05, str(model.summary()), {'fontsize': 12}) old approach
    plt.text(0.01, 1.25, str('Random Forest Test'), {
             'fontsize': 10}, fontproperties='monospace')
    plt.text(0.01, 0.05, str(classification_report(y_test, y_test_preds_rf)), {
             'fontsize': 10}, fontproperties='monospace')
    plt.text(0.01, 0.6, str('Random Forest Train'), {
             'fontsize': 10}, fontproperties='monospace')
    plt.text(0.01, 0.7, str(classification_report(y_train, y_train_preds_rf)), {
             'fontsize': 10}, fontproperties='monospace')
    plt.axis('off')
    plt.savefig(
        os.path.join(pth, "forest_train.png"),
        bbox_inches='tight',
        dpi=300)
    plt.close()

    plt.figure(figsize=(5, 5))
    plt.text(0.01, 1.25, str('Logistic Regression Train'),
             {'fontsize': 10}, fontproperties='monospace')
    plt.text(0.01, 0.05, str(classification_report(y_train, y_train_preds_lr)), {
             'fontsize': 10}, fontproperties='monospace')
    plt.text(0.01, 0.6, str('Logistic Regression Test'), {
             'fontsize': 10}, fontproperties='monospace')
    plt.text(0.01, 0.7, str(classification_report(y_test, y_test_preds_lr)), {
             'fontsize': 10}, fontproperties='monospace')
    plt.axis('off')
    plt.savefig(
        os.path.join(pth, "regression_train.png"),
        bbox_inches='tight',
        dpi=300)
    plt.close()


def feature_importance_plot(model, X_data, output_pth):
    '''
    creates and stores the feature importances in pth
    input:
            model: model object containing feature_importances_
            X_data: pandas dataframe of X values
            output_pth: path to store the figure

    output:
             None
    '''
    # Handel different input model
    actual_model = model.best_estimator_ if hasattr(
        model, 'best_estimator_') else model

    if hasattr(actual_model, 'feature_importances_'):
        importances = actual_model.feature_importances_
        title = "Feature Importance (Random Forest)"
    elif hasattr(actual_model, 'coef_'):
        importances = np.abs(actual_model.coef_[0])
        title = "Feature Importance (Logistic Regression - Absolute Coefficients)"
    else:
        print("Model type not supported for importance plotting.")
        return
    # Sort feature importances in descending order
    indices = np.argsort(importances)[::-1]
    # Rearrange feature names so they match the sorted feature importances
    names = [X_data.columns[i] for i in indices]

    # create and save plot
    plt.figure(figsize=(20, 5))
    plt.title(title)
    plt.ylabel('Importance/Weight')
    plt.bar(range(X_data.shape[1]), importances[indices])
    plt.xticks(range(X_data.shape[1]), names, rotation=90)

    pth = IMPORTANCE_IMAGE_DIR
    plt.savefig(os.path.join(pth, output_pth), bbox_inches='tight')
    plt.close()


def train_models(X_train, X_test, y_train, y_test):
    '''
    train, store model results: images + scores, and store models
    input:
              X_train: X training data
              X_test: X testing data
              y_train: y training data
              y_test: y testing data
    output:
              models
    '''

    rfc = RandomForestClassifier(random_state=42)
    # Use a different solver if the default 'lbfgs' fails to converge
    # Reference:
    # https://scikit-learn.org/stable/modules/linear_model.html#logistic-regression
    lrc = LogisticRegression(solver='lbfgs', max_iter=3000)

    param_grid = {
        'n_estimators': [200, 500],
        'max_features': ['auto', 'sqrt'],
        'max_depth': [4, 5, 100],
        'criterion': ['gini', 'entropy']
    }

    cv_rfc = GridSearchCV(estimator=rfc, param_grid=param_grid, cv=5)
    cv_rfc.fit(X_train, y_train)

    lrc.fit(X_train, y_train)

    y_train_preds_rf = cv_rfc.best_estimator_.predict(X_train)
    y_test_preds_rf = cv_rfc.best_estimator_.predict(X_test)

    y_train_preds_lr = lrc.predict(X_train)
    y_test_preds_lr = lrc.predict(X_test)

    # scores
    print('random forest results')
    print('test results')
    print(classification_report(y_test, y_test_preds_rf))
    print('train results')
    print(classification_report(y_train, y_train_preds_rf))

    print('logistic regression results')
    print('test results')
    print(classification_report(y_test, y_test_preds_lr))
    print('train results')
    print(classification_report(y_train, y_train_preds_lr))

    pth = MODEL_IMAGE_DIR

    plt.figure(figsize=(15, 8))
    lrc_plot = RocCurveDisplay.from_estimator(lrc, X_test, y_test)
    plt.savefig(
        os.path.join(pth, "roc_lrc_plot.png"),
        bbox_inches='tight',
        dpi=300)
    plt.close()

    plt.figure(figsize=(15, 8))
    ax = plt.gca()
    rfc_disp = RocCurveDisplay.from_estimator(
        cv_rfc.best_estimator_, X_test, y_test, ax=ax, alpha=0.8)
    lrc_plot.plot(ax=ax, alpha=0.8)
    plt.savefig(
        os.path.join(pth, "roc_rfc_disp.png"),
        bbox_inches='tight',
        dpi=300)
    plt.close()

    pth = MODEL_DIR
    joblib.dump(cv_rfc.best_estimator_, os.path.join(pth, "rfc_model.pkl"))
    joblib.dump(lrc, os.path.join(pth, "logistic_model.pkl"))

    return cv_rfc.best_estimator_, lrc


if __name__ == "__main__":
    # input parser
    parser = argparse.ArgumentParser(description="churn library")
    parser.add_argument(
        "-fc",
        "--force_clean",
        action="store_true",
        help="Force recalculation of model")
    args = parser.parse_args()

    # make sure all required paths exist
    required_dirs = [
        EDA_IMAGE_DIR,
        IMPORTANCE_IMAGE_DIR,
        MODEL_IMAGE_DIR,
        REPORT_IMAGE_DIR,
        MODEL_DIR
    ]

    # Create them all at once
    for directory in required_dirs:
        Path(directory).mkdir(parents=True, exist_ok=True)

    # business logic
    dataframe = import_data('data/bank_data.csv')
    dataframe['Churn'] = dataframe['Attrition_Flag'].apply(
        lambda val: 0 if val == "Existing Customer" else 1)
    perform_eda(dataframe)

    cat_columns = [
        'Gender',
        'Education_Level',
        'Marital_Status',
        'Income_Category',
        'Card_Category'
    ]
    dataframe = encoder_helper(dataframe, cat_columns, response='Churn')

    X_train, X_test, y_train, y_test = perform_feature_engineering(
        dataframe, response='Churn')

    rfc_path = os.path.join(MODEL_DIR, "rfc_model.pkl")
    lr_path = os.path.join(MODEL_DIR, "logistic_model.pkl")
    if args.force_clean or not os.path.exists(
            rfc_path) or not os.path.exists(lr_path):
        rfc_model, lr_model = train_models(X_train, X_test, y_train, y_test)
    else:
        rfc_model = joblib.load(rfc_path)
        lr_model = joblib.load(lr_path)

    # Generate predictions for the report
    print("Generating predictions...")
    y_train_preds_rf = rfc_model.predict(X_train)
    y_test_preds_rf = rfc_model.predict(X_test)

    y_train_preds_lr = lr_model.predict(X_train)
    y_test_preds_lr = lr_model.predict(X_test)

    # generate plots
    feature_importance_plot(rfc_model, X_train, 'feature_importance_rfc.png')
    classification_report_image(
        y_train,
        y_test,
        y_train_preds_lr,
        y_train_preds_rf,
        y_test_preds_lr,
        y_test_preds_rf
    )
