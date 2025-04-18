from ucimlrepo import fetch_ucirepo 
import pandas as pd
import numpy as np
from scipy.stats import skew, kurtosis

# Fetch dataset 
breast_cancer_wisconsin_diagnostic = fetch_ucirepo(id=17) 

#Data (as pandas dataframes) 
X = breast_cancer_wisconsin_diagnostic.data.features 
y = breast_cancer_wisconsin_diagnostic.data.targets 

# oncatenate features and targets into a single dataframe
dataframe = pd.concat([X, y], axis=1)

#print(dataframe)

def separa_dataframe(DataFrame,value):
    dataframe_separado = dataframe[dataframe['Diagnosis'] == value]

    return dataframe_separado

dataframe_benigno = separa_dataframe(dataframe, 'B')
dataframe_maligno = separa_dataframe(dataframe, 'M')

print(dataframe_benigno)

print(dataframe_maligno)