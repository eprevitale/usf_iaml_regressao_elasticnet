import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.linear_model import ElasticNet
from sklearn.metrics import mean_squared_error
from sklearn.preprocessing import StandardScaler, LabelEncoder

# Carregar o dataset
url = 'dataset-2003-2019.csv'  # Substitua pelo caminho correto do dataset
df = pd.read_csv(url,sep=";",encoding="utf-8")

# Exemplo de preparação dos dados
# Supondo que queremos prever o número de gols do time da casa ('home_score')
# e que temos as seguintes variáveis independentes: 'away_score', 'home_team_position', 'away_team_position'

for obj in df['team']:
    obj = obj.strip()

# Definir variáveis dependente e independentes
X = df.drop(columns=["position"])
y = df['position']

# Dividir os dados em conjuntos de treino e teste
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Padronizar os dados
encoder = LabelEncoder()
X['team'] = encoder.fit_transform(X['team'])
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)


# Treinar o modelo Elastic Net
elastic_net = ElasticNet(alpha=1.0, l1_ratio=0.5, random_state=42)
elastic_net.fit(X_train_scaled, y_train)

# Fazer previsões
y_pred = elastic_net.predict(X_test_scaled)

# Avaliar o modelo
mse = mean_squared_error(y_test, y_pred)
print(f'Erro Quadrático Médio: {mse}')
