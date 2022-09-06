# +---------------------------------------------------------------------------+
# |                 MACHINE LEARNING - LIENAR REGRESSION                      |
# +---------------------------------------------------------------------------+
# | @author: Juliano Sarnes Longo                                             |
# | @date: 04/09/2022                                                         |
# +---------------------------------------------------------------------------+
import pandas as pd
from xgboost import XGBRegressor
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
import matplotlib.pyplot as plt

# carregando os dados diamonds.csv
diamond_df = pd.read_csv('./dataset/diamonds.csv', index_col=0)
#print(diamond_df.head(20)) #mostra as 20 primeiras linhas do dataset.

# Verificações diversas.
# O dataset possui 53940 observações e 10 variáveis.
print("""Não existem dados faltantes.""")
print(diamond_df.shape)

# Não existem dados faltantes.
print("""Não existem dados faltantes.""")
print(diamond_df.isna().sum())

# Tipos de dados: todas as variáveis são numéricas exceto: cut, color e clarity.
print("""Tipos de dados: todas as variáveis são numéricas exceto: cut, color e clarity.""")
print(diamond_df.dtypes)

# Entendimento sobre as informações, a partir de algumas pesquisas é possível compreender
# que cada diamente tem um classificação de CORTE, COR e CLARESA em uma escala. Para cada
# CUT a escala (do mais alto para o mais baixo) é IDEAL, PREMIUM, VERY GOOD, GOOD e FAIR.
# Devido essas categorias podemos chamar essas variáveis de CATEGÓRICAS ORDINAIS. Sendo
# assim, temos que codificá-las de acordo com a escala, para que consigamos capturar
# adequadamente a informação em forma numérica.
# A maneira mais simples para codificar as variáveis categóricas é através do mapeamento.

# Codificação da variável categórica ordinal CUT.
cut_mapping = {'Fair': 0, 'Good': 1, 'Very Goog': 2, 'Premium': 3, 'Ideal': 4}
diamond_df.cut = diamond_df.cut.map(cut_mapping)

# Codificação da variável categórica ordinal COLOR.
color_mapping = {'J': 0, 'I': 1, 'H': 2, 'G': 3, 'F': 4, 'E': 5, 'D': 6}
diamond_df.color = diamond_df.color.map(color_mapping)

# Codificação da variável categórica ordinal CLARITY.
clarity_mapping = {'I1': 0, 'SI2': 1, 'SI1': 2, 'VS2': 3, 'VS1': 4, 'VVS2': 5, 'VVS1': 6, 'IF': 7}
diamond_df.clarity = diamond_df.clarity.map(clarity_mapping)

# Agora temos que eliminar quaisquer valores discrepante das dimensões X, Y e Z.
diamond_df = diamond_df.drop(diamond_df[diamond_df["x"]==0].index)
diamond_df = diamond_df.drop(diamond_df[diamond_df["y"]==0].index)
diamond_df = diamond_df.drop(diamond_df[diamond_df["z"]==0].index)

# Agora vamos reduzir o conjunto de dados para o percentil 99 com base em algumas variáveis
# diferentes com a finalidade de eliminarmos os valores discrepantes.
diamond_df = diamond_df[diamond_df['depth'] < diamond_df['depth'].quantile(0.99)]
diamond_df = diamond_df[diamond_df['table'] < diamond_df['table'].quantile(0.99)]
diamond_df = diamond_df[diamond_df['x'] < diamond_df['x'].quantile(0.99)]
diamond_df = diamond_df[diamond_df['y'] < diamond_df['y'].quantile(0.99)]
diamond_df = diamond_df[diamond_df['z'] < diamond_df['z'].quantile(0.99)]

# Agora que os dados foram preparados para o modelo vamos olhar o mapa de calor da correlação
# para identificar quais recursos que podemos ver que influenciem o preço e como:
diamond_plot = sns.heatmap(diamond_df.corr(), vmin = -1, vmax = +1, annot = True, cmap = 'coolwarm')
plt.show()

# Agora vamos criar as bases de treinamento, teste e validação.
model_df = diamond_df.copy()
X = model_df.drop(['price'], axis=1)
y = model_df['price']
X_train_val, X_test, y_train_val, y_test = train_test_split(X, y, test_size=0.2, random_state=0)
X_train, X_val, y_train, y_val = train_test_split(X_train_val, y_train_val, test_size=0.25, random_state=0)

# Para fins de agilidade vamos criar um GridSearchCV em um XGBoost porém em um cenário real não é
# recomendado esse tipo de abordagem.
xgb1 = XGBRegressor()
parameters = {
      'objective':['reg:squarederror'],
      'learning_rate': [.0001, 0.001, .01],
      'max_depth': [3, 5, 7],
      'min_child_weight': [3,5,7],
      'subsample': [0.1,0.5,1.0],
      'colsample_bytree': [0.1, 0.5, 1.0],
      'n_estimators': [500]
}

xgb_grid = GridSearchCV(xgb1, parameters, cv = 3, n_jobs = -1, verbose=0)
xgb_grid.fit(X_train, y_train)
xgb_cv = (xgb_grid.best_estimator_)

eval_set = [(X_train, y_train),
            (X_val, y_val)]

fit_model = xgb_cv.fit(
    X_train,
    y_train,
    eval_set=eval_set,
    eval_metric='mae',
    early_stopping_rounds=50,
    verbose=False)

#print("MAE:", mean_absolute_error(y_val, fit_model.predict(X_val)))
#print("MSE:", mean_squared_error(y_val, fit_model.predict(X_val)))
#print("R2:", r2_score(y_val, fit_model.predict(X_val)))

# Já que nosso modelo está aceitável vamos salvá-lo
#fit_model.save_model('xgb_model.json')