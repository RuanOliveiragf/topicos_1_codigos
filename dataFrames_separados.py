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

# Listar todas as colunas
todas_as_colunas = dataframe.columns.tolist()

# Encontrar os índices das colunas de início e fim
inicio = todas_as_colunas.index('radius2')
fim = todas_as_colunas.index('fractal_dimension3') + 1  # Inclui 'Diagnosis' na seleção

# Selecionar as colunas a serem deletadas
colunas_para_deletar = todas_as_colunas[inicio:fim]

# Deletar as colunas do DataFrame
dataframe = dataframe.drop(colunas_para_deletar, axis=1)

# Exibir o DataFrame modificado
#print(dataframe.head())

#print(dataframe)

def separa_dataframe(DataFrame,value):
    dataframe_separado = dataframe[dataframe['Diagnosis'] == value]

    return dataframe_separado

dataframe_benigno = separa_dataframe(dataframe, 'B')
dataframe_maligno = separa_dataframe(dataframe, 'M')

print(dataframe_benigno)

print(dataframe_maligno)
