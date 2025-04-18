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
fim = todas_as_colunas.index('Diagnosis') + 1  # Inclui 'Diagnosis' na seleção

# Selecionar as colunas a serem deletadas
colunas_para_deletar = todas_as_colunas[inicio:fim]

# Deletar as colunas do DataFrame
dataframe = dataframe.drop(colunas_para_deletar, axis=1)

# Exibir o DataFrame modificado
print(dataframe.head())

#------------------------------------CALCULO DAS MEDIDAS ESTATISTICAS------------------------------------#
def calcular_media_por_atributo(dataframe):
    medias = {}
    for coluna in dataframe.select_dtypes(include='number').columns:
        medias[coluna] = dataframe[coluna].mean()
    return medias

def calcular_moda_por_atributo(dataframe):
    modas = {}
    for coluna in dataframe.select_dtypes(include='number').columns:
        modas[coluna] = dataframe[coluna].mode().tolist()  # Convertendo para lista para melhor visualização
    return modas

def calcular_frequencia_por_atributo(dataframe):
    frequencias = {}
    for coluna in dataframe.select_dtypes(include='number').columns:
        frequencias[coluna] = dataframe[coluna].value_counts().to_dict()
    return frequencias

def calcular_mediana_por_atributo(dataframe):
    medianas = {}
    for coluna in dataframe.select_dtypes(include='number').columns:
        medianas[coluna] = dataframe[coluna].median()
    return medianas

def calcular_desvio_padrao_por_atributo(dataframe):
    desvios = {}
    for coluna in dataframe.select_dtypes(include='number').columns:
        desvios[coluna] = dataframe[coluna].std()
    return desvios

def calcular_obliquidade_por_atributo(dataframe):
    obliquidades = {}
    for coluna in dataframe.select_dtypes(include='number').columns:
        obliquidades[coluna] = skew(dataframe[coluna].dropna())  # dropna para lidar com valores faltantes
    return obliquidades

def calcular_quartis_por_atributo(dataframe):
    quartis = {}
    for coluna in dataframe.select_dtypes(include='number').columns:
        quartis[coluna] = {
            'Q1': dataframe[coluna].quantile(0.25),
            'Q3': dataframe[coluna].quantile(0.75)
        }
    return quartis

def calcular_curtose_por_atributo(dataframe):
    curtoses = {}
    for coluna in dataframe.select_dtypes(include='number').columns:
        curtoses[coluna] = kurtosis(dataframe[coluna].dropna())  # dropna para lidar com valores faltantes
    return curtoses

# Funções de cálculo já estão definidas no seu código

# Calcular todas as estatísticas
media_por_atributo = calcular_media_por_atributo(dataframe)
moda_por_atributo = calcular_moda_por_atributo(dataframe)
mediana_por_atributo = calcular_mediana_por_atributo(dataframe)
desvio_padrao_por_atributo = calcular_desvio_padrao_por_atributo(dataframe)
obliquidade_por_atributo = calcular_obliquidade_por_atributo(dataframe)
quartis_por_atributo = calcular_quartis_por_atributo(dataframe)
curtose_por_atributo = calcular_curtose_por_atributo(dataframe)

# Criar um DataFrame a partir das estatísticas calculadas
estatisticas_df = pd.DataFrame({
    'Média': pd.Series(media_por_atributo),
    'Mediana': pd.Series(mediana_por_atributo),
    'Desvio Padrão': pd.Series(desvio_padrao_por_atributo),
    'Obliquidade': pd.Series(obliquidade_por_atributo),
    'Curtose': pd.Series(curtose_por_atributo),
    'Moda': pd.Series({key: value[0] if value else np.nan for key, value in moda_por_atributo.items()}),  # Usa o primeiro valor da moda
    'Quartil 1': pd.Series({key: value['Q1'] for key, value in quartis_por_atributo.items()}),
    'Quartil 3': pd.Series({key: value['Q3'] for key, value in quartis_por_atributo.items()})
})

# Exibir o DataFrame de estatísticas
print(estatisticas_df)
