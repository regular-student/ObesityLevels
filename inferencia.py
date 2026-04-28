import pandas as pd
import pickle

normalizador = pickle.load(open("normalizador_obesidade.pkl", "rb"))
cluster_obesidade = pickle.load(open("cluster_model_obesidade_final.pkl", 'rb'))

colunas = [
    'Gender', 'Age', 'Height', 'Weight', 'family_history_with_overweight',
    'FAVC', 'FCVC', 'NCP', 'CAEC', 'SMOKE', 'CH2O', 'SCC', 'FAF', 'TUE',
    'CALC', 'MTRANS', 'NObeyesdad'
]

# novo paciente
novo_paciente = [
    ['Female', 22.0, 1.65, 60.0, 'yes', 'yes', 2.0, 3.0, 'Sometimes', 'no', 2.0, 'no', 1.0, 1.0, 'no', 'Walking', 'Normal_Weight']
]

#converte o novo paciente pra um dataframe
novo_paciente_df = pd.DataFrame(novo_paciente, columns=colunas)

novo_paciente_codificado = pd.get_dummies(novo_paciente_df)

colunas_treinamento = normalizador.feature_names_in_
novo_paciente_codificado = novo_paciente_codificado.reindex(columns=colunas_treinamento, fill_value=0)

#normaliza os dados do paciente
novo_paciente_norm_array = normalizador.transform(novo_paciente_codificado)

novo_paciente_norm = pd.DataFrame(novo_paciente_norm_array, columns=colunas_treinamento)

cluster_predict = cluster_obesidade.predict(novo_paciente_norm)

print(f"O paciente pertence ao cluster {cluster_predict[0]}")
