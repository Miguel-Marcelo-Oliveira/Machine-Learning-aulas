import pandas as pd
from sklearn.tree import DecisionTreeRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error

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

# Criando e treinando o modelo apenas com dados de treino
home_model = DecisionTreeRegressor(random_state=1)
home_model.fit(train_X, train_y)

# Fazendo previsões nos dados de validação (nunca vistos pelo modelo)
val_predictions = home_model.predict(val_X)

# Calculando o erro médio absoluto (MAE)
def get_mae(max_leaf_nodes, train_X, val_X, train_y, val_y):
    home_model = DecisionTreeRegressor(max_leaf_nodes=max_leaf_nodes, random_state=0)
    home_model.fit(train_X, train_y)
    preds_val = home_model.predict(val_X)
    mae = mean_absolute_error(preds_val, val_y)
    return(mae)

# comparando MAE com diferentes números máximos de folhas
for max_leaf_nodes in [5, 50, 100, 500, 5000]:
    my_mae = get_mae(max_leaf_nodes, train_X, val_X, train_y, val_y)
    print("Max leaf nodes: %d  \t\t Mean Absolute Error:  %d" %(max_leaf_nodes, my_mae))
