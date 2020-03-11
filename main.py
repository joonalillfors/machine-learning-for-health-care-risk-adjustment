import numpy as np
import pandas as pd

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

    print(admissions)
    print(patients)
    print(diagnoses)
    print(patients_admissions_join)
    print(grouped_icd)
    print(stayData.head(3).T)
    print(stayData.tail(3).T)
    return stayData

def main():
    stayData = getData()

main()