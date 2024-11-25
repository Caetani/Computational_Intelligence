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
from time import time

t_0_0 = time()

def index(txt):
    ''' Retorna o índice de determinada coluna de entrada
    visando utilização da matriz de entradas em numpy. '''
    global col
    for i in range(len(col)):
        if col[i][0] == txt:
            return i
    print(f"Coluna não encontrada. Value = {txt}.")
    sys.exit()

def fill(Xo, Xf, d_X):
    arr = []
    for x in range(Xo, Xf+1, d_X):
        arr.append(x)
    return arr

nro_epocas = 5
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
pct_train, pct_val, pct_test = 60, 20, 20
pct_train, pct_val, pct_test = pct_train / 100, pct_val / 100, pct_test / 100
assert pct_train + pct_val + pct_test == 1, "Erro na separação dos conjuntos."
x_train, x_test, y_train, y_test = train_test_split(entradas, saida, test_size = pct_test)#, random_state=42)
x_train, x_val, y_train, y_val = train_test_split(x_train, y_train, test_size = pct_val / (1 - pct_test))

#n1, n2 = fill(1, 2, 1), fill(1, 2, 1)
#learn_rate = [0.001]
#act_func = ['sigmoid', 'tanh']
#batch = [10000]   # Valores muito baixos exigem demasiado poder de computação.

n1, n2 = fill(1, 10, 2), fill(1, 10, 2)
learn_rate = [0.001, 0.0001]
act_func = ['sigmoid', 'tanh', 'relu']
batch = [150]

possib = []
for x1 in n1:
    for x2 in n2:
        for l_r in learn_rate:
            for a_f in act_func:
                for b_s in batch:
                    possib.append([x1, x2, l_r, a_f, b_s])

# Manipulação das entradas da rede p/ grid search
contador = 0
for i in range(len(possib)):
    print(f"\nIteração {contador+1}/{len(possib)}\t")
    to = time()
    x1, x2 = possib[contador][0], possib[contador][1]
    l_r = possib[contador][2]
    func = possib[contador][3]
    b_s = possib[contador][4]

    model = keras.Sequential(name='quantum')
    model.add(layers.Dense(x1, activation=func, input_shape=(nro_colunas,)))
    model.add(layers.Dense(x2, activation=func))
    model.add(layers.Dense(1))
    model.summary()
    model.compile(keras.optimizers.Adam(l_r), loss=keras.losses.MeanSquaredError(), metrics=[keras.metrics.MeanAbsoluteError()])

    history = model.fit(x_train, y_train, batch_size=b_s, epochs=nro_epocas, validation_data=(x_val, y_val))

    y_pred = model.predict(x_val)
    r_2 = r2_score(y_val, y_pred)
    mse = mean_squared_error(y_val, y_pred)
    rmse = sqrt(mse)
    mae = mean_absolute_error(y_val, y_pred)
    mape = mean_absolute_percentage_error(y_val, y_pred)

    possib[contador].append(r_2)
    possib[contador].append(mse)
    possib[contador].append(rmse)
    possib[contador].append(mae)
    possib[contador].append(mape)

    tf = time() - to
    contador += 1
    print(f"Tempo decorrido = {tf} sec.")
    
data = pd.DataFrame(possib, columns=['n0', 'n1', 'learning_rate', 'activation_function', 'batch_size',
                                    'R^2', 'MSE', 'RMSE', 'MAE', 'MAPE'])

data.to_excel('resultados.xlsx', engine='xlsxwriter', index=False, header=True)
print(f"\nTempo total = {time() - t_0_0} sec.")
