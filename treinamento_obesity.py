import pandas as pd
import numpy as np
import math
import pickle
from sklearn.preprocessing import MinMaxScaler
from sklearn.cluster import KMeans
from scipy.spatial.distance import cdist


#carregando dados
caminho_arquivo = "ObesityDataSet_raw_and_data_sinthetic.csv"
dados = pd.read_csv(caminho_arquivo)

#preenchendo NaN
cols_numericas = dados.select_dtypes(include=[np.number]).columns
dados[cols_numericas] = dados[cols_numericas].fillna(dados[cols_numericas].mean())

cols_categoricas = dados.select_dtypes(exclude=[np.number]).columns
for col in cols_categoricas:
    dados[col] = dados[col].fillna(dados[col].mode()[0])

# transformando os dados categóricos
dados_codificados = pd.get_dummies(dados, drop_first=True)

scaler = MinMaxScaler()

#salvando o normalizador
normalizador = scaler.fit(dados_codificados)
pickle.dump(normalizador, open("normalizador_obesidade.pkl", "wb"))

dados_norm_array = normalizador.transform(dados_codificados)
dados_norm = pd.DataFrame(dados_norm_array, columns=dados_codificados.columns)

#primeiro teste com 15 apenas.
K = range(1, 15)
distortions = []

for i in K:
    # Use n_init='auto' ou 10 para suprimir avisos em versões mais novas do sklearn
    cluster_model_temp = KMeans(n_clusters=i, random_state=42, n_init=10).fit(dados_norm)
    distortions.append(
        sum(
            np.min(
                cdist(dados_norm, cluster_model_temp.cluster_centers_, 'euclidean'), axis=1
            ) / dados_norm.shape[0]
        )
    )

#matemática pra achar o cotovelo da reta
x0 = K[0]
y0 = distortions[0]
xn = K[-1]
yn = distortions[-1]
distances = []

for i in range(len(distortions)):
    x = K[i]
    y = distortions[i]
    numerador = abs((yn-y0)*x - (xn-x0)*y + xn*y0 - yn*x0)
    denominador = math.sqrt((yn-y0)**2 + (xn-x0)**2)
    distances.append(numerador/denominador)


numero_clusters_otimo = K[distances.index(np.max(distances))]
print(f"Número de clusters ótimo •-> {numero_clusters_otimo}")

cluster_model_final = KMeans(
    n_clusters=numero_clusters_otimo,
    random_state=42,
    n_init=10
).fit(dados_norm)

pickle.dump(cluster_model_final, open('cluster_model_obesidade_final.pkl', 'wb'))
print("::: success!")
