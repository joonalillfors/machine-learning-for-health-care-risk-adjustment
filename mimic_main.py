import numpy as np
import pandas as pd
from datetime import datetime
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression

def getData():
    admissions = pd.read_csv('../data/ADMISSIONS.csv').drop(columns="ROW_ID")
    patients = pd.read_csv('../data/PATIENTS.csv').drop(columns="ROW_ID").set_index('SUBJECT_ID')
    diagnoses = pd.read_csv('../data/DIAGNOSES_ICD.csv').drop(columns="ROW_ID")

    # LEFT JOIN admissions and patients
    patients_admissions_join = admissions.join(patients, on="SUBJECT_ID")
    
    # Gather all ICD9 diagnoses to a single array for each hospital admission
    grouped_icd = diagnoses.groupby(['HADM_ID'], sort=False)
    grouped_icd = grouped_icd.agg({"ICD9_CODE": lambda x: list(x.unique())})
    grouped_icd['ICD9_CODE'] = grouped_icd['ICD9_CODE'].apply(np.array)

    # Assemble return dataframe
    stayData = patients_admissions_join.join(grouped_icd, on="HADM_ID")

    # Convert timestamps
    stayData['ADMITTIME'] = stayData['ADMITTIME'].apply(lambda x: datetime.strptime(x, '%Y-%m-%d %H:%M:%S'))
    stayData['DISCHTIME'] = stayData['DISCHTIME'].apply(lambda x: datetime.strptime(x, '%Y-%m-%d %H:%M:%S'))
    # stayData['EDREGTIME'] = stayData['EDREGTIME'].apply(lambda x: datetime.strptime(x, '%Y-%m-%d %H:%M:%S'))
    # stayData['EDOUTTIME'] = stayData['EDOUTTIME'].apply(lambda x: datetime.strptime(x, '%Y-%m-%d %H:%M:%S'))
    stayData['DOB'] = stayData['DOB'].apply(lambda x: datetime.strptime(x, '%Y-%m-%d %H:%M:%S'))

    # Convert timestamps into hours/years
    stayData['DOB'] = (stayData['ADMITTIME'] - stayData['DOB']).apply(lambda x: x.to_timedelta64().astype('timedelta64[Y]').astype(np.int64))
    stayData['DISCHTIME'] = (stayData['DISCHTIME'] - stayData['ADMITTIME']).apply(lambda x: x.to_timedelta64().astype('timedelta64[h]').astype(np.int64))

    print(admissions)
    print(patients)
    print(diagnoses)
    print(patients_admissions_join)
    print(grouped_icd)
    print(stayData.head(1).T)
    print(stayData)

    return train_test_split(stayData, test_size=0.16)

def main():
    train, test = getData()

    Y_train = train['DISCHTIME']
    X_train = train.drop(columns=['DISCHTIME'])
    
    Y_test = test['DISCHTIME']
    X_test = test.drop(columns=['DISCHTIME'])

    print(X_train.shape, Y_train.shape)

    lin = LinearRegression().fit(X_train, Y_train)
    lin_score = lin.score(X_test, Y_test)
    print(lin_score)


main()