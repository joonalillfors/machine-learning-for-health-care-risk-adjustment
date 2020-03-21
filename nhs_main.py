import numpy as np
import pandas as pd
from sklearn.linear_model import LinearRegression
from sklearn import tree
from sklearn.ensemble import RandomForestRegressor
import lightgbm as lgb

def main():
    # from practice info drop rurality, atypical characteristics and practice type (not in 15-16) and quarter used (not in 18-19)
    # indcludecols15 = ['NHSEnglandRegionCode','CCGCode','PracticeCloseDate','ContractType','DispensingPractice','QuarterUsedForPatientData','NumberOfRegisteredPatientsLastKnownFigure','NumberOfWeightedPatientsLastKnownFigure','AveragePaymentPerRegisteredPatient_£','AveragePaymentPerWeightedPatient_£','TotalNHSPaymentsToGeneralPractice_£','DeductionsForPensionsLeviesAndPrescriptionChargeIncome_£','TotalNHSPaymentsToGeneralPracticeMinusDeductions_£']
    indcludecols15 = ['NHSEnglandRegionCode','ContractType','DispensingPractice','QuarterUsedForPatientData','NumberOfRegisteredPatientsLastKnownFigure','NumberOfWeightedPatientsLastKnownFigure','TotalNHSPaymentsToGeneralPractice_£']
    indcludecols16 = ['NHSEnglandRegionLocalOfficeCode','ContractType','DispensingPractice','QuarterUsedForPatientData','NumberofRegisteredPatientsLastKnownFigure','NumberofWeightedPatientsLastKnownFigure','AveragePaymentsPerRegisteredPatient_£','AveragePaymentsPerWeightedPatient_£','TotalNHSPaymentsToGeneralPractice_£','DeductionsForPensionsLeviesAndPrescriptionChargeIncome_£','TotalNHSPaymentsToGeneralPracticeMinusDeductions_£']
    indcludecols17 = ['NHS England (Region, local office) Code','CCG Code','Practice Code','Practice Postcode','Practice Open Date','Practice Close Date','Contract Type','Dispensing Practice','Quarter used for patient data','Number of Registered Patients (Last Known Figure)','Number of Weighted Patients (Last Known Figure)','Average payments per registered patient','Average payments per weighted patient','Total NHS Payments to General Practice',"Deductions for Pensions, Levies and Prescription Charge Income",'Total NHS Payments to General Practice Minus Deductions']
    indcludecols18 = ['NHS England and NHS Improvement (Region, local office) Code','CCG Code','Practice Code','Practice Postcode','Practice Open Date','Practice Close Date','Contract Type','Dispensing Practice','Average Number of Registered Patients','Average Number of Weighted Patients','Average payments per registered patient','Average payments per weighted patient','Total NHS Payments to General Practice',"Deductions for Pensions, Levies and Prescription Charge Income",'Total NHS Payments to General Practice Minus Deductions']

    excludecols15 = ['GlobalSum_£','MPIGCorrectionFactor_£','PremisesPayments_£','Seniority_£','DoctorsRetainerSchemePayments_£','TotalLocumAllowances_£','ProlongedStudyLeave_£','Appraisal_AppraiserCostsInRespectOfLocums_£','PCOAdminOther_£','TotalQOFPayments_£','Alcohol_£','ChildhoodVaccinationAndImmunisationScheme_£','ExtendedHoursAccess_£','FacilitatingTimelyDiagnosisAndSupportForPeopleWithDementia_£','ImprovingPatientOnlineAccess_£','InfluenzaAndPneumococcalImmunisations_£','LearningDisabilities_£','MinorSurgery_£','PatientParticipation_£','RemoteCareMonitoring_£','RiskProfilingAndCaseManagement_£','RotavirusAndShinglesImmunisation_£','ServicesForViolentPatients_£','UnplannedAdmissions_£','OutOfAreaInHoursUrgentCare_£','Meningitis_£','NationalEnhancedServices_£','LocalEnhancedServices_NHAIS_£','LocalEnhancedServices_ISFE_£','InformationManagementAndTechnology_£','BalanceOfPMSExpenditure_£','NonDESItemPneumococcalVaccineChildhoodImmunisationMainProgramme_£','PrescribingFeePayments_£','DispensingFeePayments_£','ReimbursementOfDrugs_£','OtherPayments_£']
    excludecols16 = ['GlobalSum_£','MPIGCorrectionfactor_£','PremisesPayments_£','Seniority_£','DoctorsRetainerSchemePayments_£','TotalLocumAllowances_£','ProlongedStudyLeave_£','AppraisalAppraiserCostsinRespectofLocums_£','PCOAdminOther_£','TotalQOFPayments_£','ChildhoodVaccinationAndImmunisationScheme_£','ExtendedHoursAccess_£','FacilitatingTimelyDiagnosisAndSupportForPeopleWithDementia_£','InfluenzaandPneumococcalImmunisations_£','LearningDisabilities_£','MinorSurgery_£','RotavirusAndShinglesImmunisation_£','ServicesForViolentPatients_£','UnplannedAdmissions_£','OutOfAreaInHoursUrgentCare_£','Meningitis_£','TotalNationalEnhancedServices_£','TotalLocalEnhancedServices_£','GeneralPracticeForwardView_£','InformationManagementandTechnology_£','BalanceofPMSExpenditure_£','NonDESItemPneumococcalVaccineChildhoodImmunisationMainProgramme_£','PrescribingFeePayments_£','DispensingFeePayments_£','ReimbursementOfDrugs_£','OtherPayments_£']
    excludecols17 = ['Global Sum','MPIG Correction factor','Premises Payments','Seniority','Doctors Retainer Scheme Payments','Total Locum Allowances','Prolonged Study Leave','Appraisal Costs ','PCO Admin Other','Total QOF Payments','Childhood Vaccination and Immunisation Scheme','Extended Hours Access','Facilitating Timely Diagnosis and Support for People with Dementia','Influenza and Pneumococcal Immunisations','Learning Disabilities','Minor Surgery','Rotavirus and Shingles Immunisation','Services for Violent Patients','Unplanned Admissions','Out Of Area in Hours Urgent Care','Meningitis','Pertussis','Total Local Enhanced Services','General Practice Forward View','Information Management and Technology','Balance of PMS Expenditure','Non DES Item Pneumococcal Vaccine, Childhood Immunisation Main Programme','Prescribing Fee Payments','Dispensing Fee Payments','Reimbursement of Drugs','Other Payments1']
    excludecols18 = ['Global Sum','MPIG Correction factor','Balance of PMS Expenditure','Total QOF Payments','Childhood Vaccination and Immunisation Scheme','Extended Hours Access','Influenza and Pneumococcal Immunisations','Learning Disabilities','Meningitis','Minor Surgery','Out Of Area in Hours Urgent Care','Pertussis','Rotavirus and Shingles Immunisation','Services for Violent Patients','Total Local Incentive Schemes','Premises Payments','Seniority','Doctors Retainer Scheme Payments','Total Locum Allowances','Appraisal Costs ','Prolonged Study Leave','PCO Admin Other','Information Management and Technology','Non DES Item Pneumococcal Vaccine, Childhood Immunisation Main Programme','General Practice Forward View','Prescribing Fee Payments','Dispensing Fee Payments','Reimbursement of Drugs','Other Payments1']
    
    df15 = pd.read_csv('../nhs-data/nhspaymentsgp-15-16-csv.csv', encoding = "ISO-8859-1")[indcludecols15]
    df15 = df15[df15.NHSEnglandRegionCode != 'Q73']

    df16 = pd.read_csv('../nhs-data/nhspaymentsgp-16-17-csv.csv', encoding = "ISO-8859-1")
    df16 = df16.rename(columns={
        'NHSEnglandRegionLocalOfficeCode': 'NHSEnglandRegionCode',
        'AveragePaymentsPerRegisteredPatient_£': 'AveragePaymentPerRegisteredPatient_£',
        'AveragePaymentsPerWeightedPatient_£': 'AveragePaymentPerWeightedPatient_£',
        'NumberofWeightedPatientsLastKnownFigure': 'NumberOfWeightedPatientsLastKnownFigure',
        'NumberofRegisteredPatientsLastKnownFigure': 'NumberOfRegisteredPatientsLastKnownFigure'
        })
    df16 = df16[indcludecols15].dropna()
    df16 = df16[df16.DispensingPractice != 'UNKNOWN']
    df16 = df16[df16.ContractType != 'UNKNOWN']
    df16 = df16[df16.NHSEnglandRegionCode != 'Q83']
    df16 = df16[df16.NHSEnglandRegionCode != 'Q84']
    
    df17 = pd.read_csv('../nhs-data/nhspaymentsgp-17-18-csv.csv', encoding = "ISO-8859-1")[indcludecols17]
    
    df18 = pd.read_csv('../nhs-data/nhspaymentsgp-18-19-csv.csv', encoding = "ISO-8859-1")[indcludecols18]

    train_y = df15['TotalNHSPaymentsToGeneralPractice_£']
    train_x = df15.drop(columns=['TotalNHSPaymentsToGeneralPractice_£'])

    train_x = pd.get_dummies(train_x)
    print(train_x)
    print(train_x.head(1).T)

    test_y = df16['TotalNHSPaymentsToGeneralPractice_£']
    test_x = df16.drop(columns=['TotalNHSPaymentsToGeneralPractice_£'])

    test_x = pd.get_dummies(test_x)
    print(test_x)
    print(test_x.head(1).T)
    
    print(df15.shape)
    print(df16.shape)
    print(df17.shape)
    print(df18.shape)

    linreg = LinearRegression().fit(train_x, train_y)
    print("LINEAR REGRESSION SCORE:",linreg.score(test_x, test_y))
    print("LINEAR REGRESSION SCORE:", linreg.score(train_x, train_y))

    dtr = tree.DecisionTreeRegressor().fit(train_x, train_y)
    print("DECISION TREE SCORE:",dtr.score(test_x, test_y))
    print("DECISION TREE SCORE:", dtr.score(train_x, train_y))

    rf = RandomForestRegressor(n_estimators=30).fit(train_x, train_y)
    print("RANDOM FOREST SCORE:",rf.score(test_x, test_y))
    print("RANDOM FOREST SCORE:", rf.score(train_x, train_y))

main()