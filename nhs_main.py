import numpy as np
import pandas as pd
from sklearn.linear_model import LinearRegression, Lasso, Ridge
from sklearn import tree
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.neural_network import MLPRegressor
from sklearn.metrics import mean_squared_error, mean_absolute_error
import lightgbm as lgb
import xgboost as xgb
from xgboost.sklearn import XGBRegressor
from lightgbm.sklearn import LGBMRegressor

def printResults(model, test_x, test_y, val_x, val_y, train_x, train_y, name):
    model_pred_val_y = model.predict(val_x)
    model_pred_test_y = model.predict(test_x)
    print(f"{name} TST SCORE:", model.score(test_x, test_y))
    print(f"{name} VAL SCORE:", model.score(val_x, val_y))
    print(f"{name} TRA SCORE:", model.score(train_x, train_y))
    print(f"{name} TST RMSE:", mean_squared_error(test_y, model_pred_test_y, squared=False))
    print(f"{name} VAL RMSE:", mean_squared_error(val_y, model_pred_val_y, squared=False))
    print(f"{name} TST MAE:", mean_absolute_error(test_y, model_pred_test_y))
    print(f"{name} VAL MAE:", mean_absolute_error(val_y, model_pred_val_y))

def main():
    # from practice info drop rurality, atypical characteristics and practice type (not in 15-16) and quarter used (not in 18-19)
    # includecols15 = ['NHSEnglandRegionCode','CCGCode','PracticeCloseDate','ContractType','DispensingPractice','QuarterUsedForPatientData','NumberOfRegisteredPatientsLastKnownFigure','NumberOfWeightedPatientsLastKnownFigure','AveragePaymentPerRegisteredPatient_£','AveragePaymentPerWeightedPatient_£','TotalNHSPaymentsToGeneralPractice_£','DeductionsForPensionsLeviesAndPrescriptionChargeIncome_£','TotalNHSPaymentsToGeneralPracticeMinusDeductions_£']
    includecols15 = ['NHSEnglandRegionCode','DispensingPractice','ContractType','NumberOfRegisteredPatientsLastKnownFigure','NumberOfWeightedPatientsLastKnownFigure','TotalNHSPaymentsToGeneralPractice_£']
    includecols17 = ['NHS England (Region, local office) Code','CCG Code','Practice Code','Practice Postcode','Practice Open Date','Practice Close Date','Contract Type','Dispensing Practice','Quarter used for patient data','Number of Registered Patients (Last Known Figure)','Number of Weighted Patients (Last Known Figure)','Average payments per registered patient','Average payments per weighted patient','Total NHS Payments to General Practice',"Deductions for Pensions, Levies and Prescription Charge Income",'Total NHS Payments to General Practice Minus Deductions']
    includecols18 = ['NHS England and NHS Improvement (Region, local office) Code','CCG Code','Practice Code','Practice Postcode','Practice Open Date','Practice Close Date','Contract Type','Dispensing Practice','Average Number of Registered Patients','Average Number of Weighted Patients','Average payments per registered patient','Average payments per weighted patient','Total NHS Payments to General Practice',"Deductions for Pensions, Levies and Prescription Charge Income",'Total NHS Payments to General Practice Minus Deductions']
    
    df15 = pd.read_csv('../nhs-data/nhspaymentsgp-15-16-csv.csv', encoding = "ISO-8859-1")[includecols15]
    df15 = df15[df15.NHSEnglandRegionCode != 'Q73']

    df16 = pd.read_csv('../nhs-data/nhspaymentsgp-16-17-csv.csv', encoding = "ISO-8859-1")
    df16 = df16.rename(columns={
        'NHSEnglandRegionLocalOfficeCode': 'NHSEnglandRegionCode',
        'AveragePaymentsPerRegisteredPatient_£': 'AveragePaymentPerRegisteredPatient_£',
        'AveragePaymentsPerWeightedPatient_£': 'AveragePaymentPerWeightedPatient_£',
        'NumberofWeightedPatientsLastKnownFigure': 'NumberOfWeightedPatientsLastKnownFigure',
        'NumberofRegisteredPatientsLastKnownFigure': 'NumberOfRegisteredPatientsLastKnownFigure'
        })
    df16 = df16[includecols15].dropna()
    df16 = df16[df16.DispensingPractice != 'UNKNOWN']
    df16 = df16[df16.ContractType != 'UNKNOWN']
    df16 = df16[df16.NHSEnglandRegionCode != 'Q83']
    df16 = df16[df16.NHSEnglandRegionCode != 'Q84']
    
    df17 = pd.read_csv('../nhs-data/nhspaymentsgp-17-18-csv.csv', encoding = "ISO-8859-1")[includecols17]
    df17 = df17.rename(columns={
        'NHS England (Region, local office) Code': 'NHSEnglandRegionCode',
        'Contract Type': 'ContractType',
        'Dispensing Practice': 'DispensingPractice',
        'Number of Registered Patients (Last Known Figure)': 'NumberOfRegisteredPatientsLastKnownFigure',
        'Number of Weighted Patients (Last Known Figure)': 'NumberOfWeightedPatientsLastKnownFigure',
        'Total NHS Payments to General Practice': 'TotalNHSPaymentsToGeneralPractice_£'
    })
    df17 = df17[includecols15].dropna()
    df17 = df17[df17.DispensingPractice != 'UNKNOWN']
    df17 = df17[df17.ContractType != 'UNKNOWN']
    df17 = df17[df17.NHSEnglandRegionCode != 'Q83']
    df17 = df17[df17.NHSEnglandRegionCode != 'Q84']

    df18 = pd.read_csv('../nhs-data/nhspaymentsgp-18-19-csv.csv', encoding = "ISO-8859-1")[includecols18]

    train_y = df15['TotalNHSPaymentsToGeneralPractice_£']
    train_x = df15.drop(columns=['TotalNHSPaymentsToGeneralPractice_£'])

    train_x = pd.get_dummies(train_x)
    print(train_x)
    print(train_x.head(1).T)

    val_y = df16['TotalNHSPaymentsToGeneralPractice_£']
    val_x = df16.drop(columns=['TotalNHSPaymentsToGeneralPractice_£'])

    val_x = pd.get_dummies(val_x)
    print(val_x)
    print(val_x.head(1).T)

    test_y = df17['TotalNHSPaymentsToGeneralPractice_£']
    test_x = df17.drop(columns=['TotalNHSPaymentsToGeneralPractice_£'])

    test_x = pd.get_dummies(test_x)
    print(test_x)
    print(test_x.head(1).T)

    print(df15.shape)
    print(df16.shape)
    print(df17.shape)
    print(df18.shape)

    # LINEAR REGRESSION
    linreg = LinearRegression().fit(train_x, train_y)
    printResults(linreg, test_x, test_y, val_x, val_y, train_x, train_y, "LINEAR REGRESSION")

    # LASSO
    lasso = Lasso(alpha=100, max_iter=1000).fit(train_x, train_y)
    printResults(lasso, test_x, test_y, val_x, val_y, train_x, train_y, "LASSO")

    # RIDGE REGRESSION
    ridge = Ridge(alpha=0.5).fit(train_x, train_y)
    printResults(ridge, test_x, test_y, val_x, val_y, train_x, train_y, "RIDGE")

    # DECISION TREE
    dtr = tree.DecisionTreeRegressor().fit(train_x, train_y)
    printResults(dtr, test_x, test_y, val_x, val_y, train_x, train_y, "DECISION TREE")

    # RANDOM FOREST
    rf = RandomForestRegressor(n_estimators=200,
                               ).fit(train_x, train_y)
    printResults(rf, test_x, test_y, val_x, val_y, train_x, train_y, "RANDOM FOREST")

    # GRADIENT BOOSTING
    gb = GradientBoostingRegressor(n_estimators=200, 
                                   learning_rate=0.07, 
                                   max_depth=6,
                                   subsample=0.8,
                                   ).fit(train_x, train_y)
    printResults(gb, test_x, test_y, val_x, val_y, train_x, train_y, "GRADIENT BOOSTING")
    print(gb.feature_importances_)

    # XGBOOST
    xgboost = XGBRegressor(n_extimators=100, 
                           learning_rate=0.1, 
                           max_depth=7,
                           subsample=0.8,
                           colsample_bylevel=0.9,
                           ).fit(train_x, train_y)
    printResults(xgboost, test_x, test_y, val_x, val_y, train_x, train_y, "XGBOOST")
    print(xgboost.feature_importances_)

    # LIGHTGBM
    lightgbm = LGBMRegressor(n_estimators=300,
                             learning_rate=0.12,
                             max_depth=10,
                             num_leaves=30,
                             min_child_samples=1,
                             subsample_for_bin=4750,
                             reg_alpha=2.7,
                             ).fit(train_x, train_y)
    printResults(lightgbm, test_x, test_y, val_x, val_y, train_x, train_y, "LIGHTGBM")

    # MLP
    mlp = MLPRegressor(hidden_layer_sizes=(10,),
                       solver='lbfgs',
                       max_iter=1500,
                       learning_rate_init=0.005,
                       alpha=0.0035
                       ).fit(train_x, train_y)
    printResults(mlp, test_x, test_y, val_x, val_y, train_x, train_y, "MLP")
    
main()