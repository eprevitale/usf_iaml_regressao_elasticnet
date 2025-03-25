import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import ElasticNet
from sklearn.metrics import mean_absolute_error, mean_squared_error, root_mean_squared_error, r2_score
from sklearn.preprocessing import StandardScaler, LabelEncoder

url = 'dataset-2003-2019.csv'
df = pd.read_csv(url,sep=";",encoding="utf-8")


for obj in df['team']:
    obj = obj.strip()
encoder = LabelEncoder()
for col in df.columns:
    df[col] = encoder.fit_transform(df[col])


X = df.drop(columns=["position"])
y = df['position']


X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)


scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)


elastic_net = ElasticNet(alpha=1.0, l1_ratio=0.5, random_state=42)
elastic_net.fit(X_train_scaled, y_train)


y_pred = elastic_net.predict(X_test_scaled)


mae = mean_absolute_error(y_test, y_pred)
print(f'Erro Absoluto Médio: {mae}')


mse = mean_squared_error(y_test, y_pred)
print(f'Erro Quadrático Médio: {mse}')


rmse = root_mean_squared_error(y_test, y_pred)
print(f'Raíz do Erro Quadrático Médio: {rmse}')


r2 = r2_score(y_test, y_pred)
print(f'Coeficiente de Determinação: {r2}')
