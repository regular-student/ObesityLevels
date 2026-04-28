import pandas as pd
import pickle
import numpy as np

#carregando os dados
caminho_arquivo = "ObesityDataSet_raw_and_data_sinthetic.csv"
dados = pd.read_csv(caminho_arquivo)
cluster_model = pickle.load(open("cluster_model_obesidade_final.pkl", "rb"))

dados['Cluster'] = cluster_model.labels_

cols_numericas = dados.select_dtypes(include=[np.number]).columns.drop('Cluster')
cols_categoricas = dados.select_dtypes(exclude=[np.number]).columns

desc_num = dados.groupby('Cluster')[cols_numericas].mean()

desc_cat = dados.groupby('Cluster')[cols_categoricas].agg(lambda x: x.mode()[0])


#agrupa os dados por cluster
descricao_clusters = pd.concat([desc_num, desc_cat], axis=1)

colunas_chave = ['Age', 'Weight', 'Height', 'FAF', 'FCVC', 'Gender', 'MTRANS', 'NObeyesdad']

print("\t• Perfil médio numérico de cada cluster\n")
print(descricao_clusters[colunas_chave].round(2))

descricao_clusters.to_csv("descricao_clusters_obesidade.csv")
print("\nArquivo salvo")
