import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error
from sklearn.ensemble import RandomForestRegressor

# Carregando o dataset
home_data = pd.read_csv('melb_data.csv')

# Removendo linhas com valores nulos
home_data = home_data.dropna(axis=0)

# Selecionando o target
y = home_data['Price']

# Selecionando as features
selected_features = ['Rooms', 'Bathroom', 'Landsize', 'Lattitude', 'Longtitude']
X = home_data[selected_features]

# Dividindo em dados de treino e validação
train_X, val_X, train_y, val_y = train_test_split(X, y, random_state=1)

# Criando e treinando o modelo
home_model = RandomForestRegressor(random_state=1)
home_model.fit(train_X, train_y)

# Fazendo as previsões
home_pred = home_model.predict(val_X)

# Calculando o MAE da random forest

val_mae = mean_absolute_error(home_pred, val_y)
# print("Home Model Val. MAE: ", val_mae)

# Calculando o MAE em cima dos dados de treino
train_pred = home_model.predict(train_X)
train_mae = mean_absolute_error(train_pred, train_y)

# print("MAE treino:", train_mae)
# print("MAE validação:", val_mae)

for max_depth in [5, 10, 15, 20, 50, 75, 100, None]:
    model = RandomForestRegressor(max_depth=max_depth, random_state=1)
    model.fit(train_X, train_y)
    mae_treino = mean_absolute_error(train_y, model.predict(train_X))
    mae_val = mean_absolute_error(val_y, model.predict(val_X))
    print(f"max_depth={str(max_depth):>4} | treino: {mae_treino:,.0f} | validação: {mae_val:,.0f}")

