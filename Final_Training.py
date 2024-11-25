import pandas as pd
import numpy as np
from sklearn.metrics import r2_score, mean_squared_error, mean_absolute_percentage_error, mean_absolute_error
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split 
from sklearn.preprocessing import StandardScaler
import keras
from keras import layers
import sys
from sklearn.decomposition import PCA
from math import sqrt

def index(txt):
    ''' Retorna o índice de determinada coluna de entrada
    visando utilização da matriz de entradas em numpy. '''
    global col
    for i in range(len(col)):
        if col[i][0] == txt:
            return i
    print(f"Coluna não encontrada. Value = {txt}.")
    sys.exit()

nro_epocas = 50
col = []

# Importação e aleatorização do dataset
df = pd.read_csv("quantum.csv")
#df.drop(["index", "dist_au", "alpha_symm", "kQ_rate"], axis=1, inplace=True)
#df.drop(["index", "kQ_rate"], axis=1, inplace=True)     # Testando PCA
df.drop(["index", "kQ_rate"], axis=1, inplace=True)
df = df.sample(frac=1).reset_index(drop=True)

# Verificar se não há linhas vazias.
assert df.isnull().values.any() == False, "Valor nulo encontrado!"
#df.info()  
#print(df.describe())

# Criando "dicionário" para número dos índices.
cont = 0
for txt in df.columns:
    col.append([txt, cont])
    cont += 1
col.pop(-1)

# Separação dos dados de entrada e de saída
df = df.to_numpy()
entradas = df[:, :-1]
saida = df[:, -1]
nro_linhas, nro_colunas = entradas.shape[0], entradas.shape[1]  # Reavaliar se isso é necessário.

# Normalização dos dados
sc = StandardScaler()   # Responsável pela normalização dos dados: z = (x - u) / s
                        #   Onde:  x = valor_atual, u = média, s = desvio padrão
entradas = sc.fit_transform(entradas)
saida = sc.fit_transform(saida.reshape(-1, 1))

# Implementando o PCA
entradas_PCA = np.array([])
pca = PCA(n_components=1)
dist_au_width_2_au = np.column_stack((entradas[:, index('width_2_au')], entradas[:, index('dist_au')]))
alpha_symm_slope = np.column_stack((entradas[:, index('alpha_symm')], entradas[:, index('slope')]))
pca1 = pca.fit_transform(dist_au_width_2_au.reshape(-1, 2))
pca2 = pca.fit_transform(alpha_symm_slope.reshape(-1, 2))
flag = 0
for txt in col:
    if txt[0] != 'width_2_au' and txt[0] != 'dist_au' and txt[0] != 'alpha_symm' and txt[0] != 'slope':
        if flag == 0:
            entradas_PCA = entradas[:, index(txt[0])]
            flag = 1
        else:
            entradas_PCA = np.column_stack((entradas_PCA, entradas[:, index(txt[0])]))
entradas_PCA = np.column_stack((entradas_PCA, pca1))
entradas_PCA = np.column_stack((entradas_PCA, pca2))
nro_colunas = entradas_PCA.shape[1]

# Igualando as matrizes para não precisar alterar o código.
entradas = entradas_PCA

# Divisão dos conjuntos trinamento/validação/teste.
pct_train, pct_val, pct_test = 80, 10, 10
pct_train, pct_val, pct_test = pct_train / 100, pct_val / 100, pct_test / 100
assert pct_train + pct_val + pct_test == 1, "Erro na separação dos conjuntos."
x_train, x_test, y_train, y_test = train_test_split(entradas, saida, test_size = pct_test)#, random_state=42)
x_train, x_val, y_train, y_val = train_test_split(x_train, y_train, test_size = pct_val / (1 - pct_test))

# Manipulação das entradas da rede p/ grid search
for i in range(1, 11):
        for j in range(1, 11):
            pass
estrutura = [i, j]

model = keras.models.load_model("melhor_modelo_500_epocas")
#model = keras.Sequential(name='quantum')
#model.add(layers.Dense(10, activation='relu', input_shape=(nro_colunas,)))
#model.add(layers.Dense(10, activation='relu'))
#model.add(layers.Dense(1))
#model.summary()
#model.compile(keras.optimizers.Adam(0.001), loss=keras.losses.MeanSquaredError(), metrics=[keras.metrics.MeanAbsoluteError()])

#history = model.fit(x_train, y_train, batch_size=64, epochs=nro_epocas, validation_data=(x_val, y_val))

#y_pred = model.predict(x_test)
#r_2 = r2_score(y_test, y_pred)

y_pred = model.predict(x_test)
r_2 = r2_score(y_test, y_pred)
mse = mean_squared_error(y_test, y_pred)
rmse = sqrt(mse)
mae = mean_absolute_error(y_test, y_pred)
mape = mean_absolute_percentage_error(y_test, y_pred)
plt.figure(0, figsize=(6, 6), dpi=160)
comparacao = np.arange(-16, 2, 0.1)
plt.plot(y_test, y_pred, 'ro', label = "Predição do modelo")
plt.plot(comparacao, comparacao, 'b', label = "Estimativa ideal")
plt.xlabel("Taxa de reação")
plt.ylabel("Taxa de reação estimada")
plt.legend(loc='lower right')
plt.savefig(f"final_teste_prediction.png", bbox_inches='tight')
plt.grid()

#print("Características da rede:\n")
print(f"Resultados do modelo:\n\tR^2 = {r_2}\n\tMSE = {mse}\n\tRMSE = {rmse}\n\tMAE = {mae}\n\tMAPE = {mape}\n")

plt.show()

