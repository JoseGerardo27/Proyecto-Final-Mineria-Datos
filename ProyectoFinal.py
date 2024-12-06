import os
import pandas as pd
from sklearn.model_selection import train_test_split, cross_val_score, GridSearchCV
from sklearn.preprocessing import MinMaxScaler
import matplotlib.pyplot as plt
import seaborn as sns

# Imprimir la ruta de trabajo actual
print("Ruta de trabajo actual:", os.getcwd())

# Ruta al directorio donde están los archivos CSV
directorio_data = './archive/'  

# Lista de nombres de archivos CSV que quieres cargar
archivos = [
    'Batting.csv',
    'Fielding.csv',
    'Pitching.csv',
    'AllstarFull.csv',
    'HallOfFame.csv',
    'Master.csv',
    'Teams.csv'
]

# Intentar cargar los archivos en DataFrames de pandas
try:
    batting_df = pd.read_csv(os.path.join(directorio_data, 'Batting.csv'))
    fielding_df = pd.read_csv(os.path.join(directorio_data, 'Fielding.csv'))
    pitching_df = pd.read_csv(os.path.join(directorio_data, 'Pitching.csv'))
    allstar_full_df = pd.read_csv(os.path.join(directorio_data, 'AllstarFull.csv'))
    hall_of_fame_df = pd.read_csv(os.path.join(directorio_data, 'HallOfFame.csv'))
    master_df = pd.read_csv(os.path.join(directorio_data, 'Master.csv'))
    teams_df = pd.read_csv(os.path.join(directorio_data, 'Teams.csv'))

    print("Archivos cargados exitosamente.")
except FileNotFoundError as e:
    print(f"Error: {e}")
except pd.errors.EmptyDataError:
    print("Error: Un archivo está vacío.")
except pd.errors.ParserError:
    print("Error: Error al analizar un archivo CSV.")
except Exception as e:
    print(f"Error inesperado: {e}")

# 1.- EXPLORACIÓN Y ANÁLISIS PRELIMINAR
# Imprimir los nombres de las tablas y sus primeras filas con encabezado
print("Tabla: Batting")
print(batting_df.head())

print("\nTabla: Fielding")
print(fielding_df.head())

print("\nTabla: Pitching")
print(pitching_df.head())

print("\nTabla: AllstarFull")
print(allstar_full_df.head())

print("\nTabla: HallOfFame")
print(hall_of_fame_df.head())

print("\nTabla: Master")
print(master_df.head())

print("\nTabla: Teams")
print(teams_df.head())

# Descripción estadística de los DataFrames
print("Tabla: Batting")
print(batting_df.describe())
print("\nTabla: Fielding")
print(fielding_df.describe())
print("\nTabla: Pitching")
print(pitching_df.describe())

# 2.- PREPROCESAMIENTO DE DATOS
# Verificar valores nulos
print(batting_df.isnull().sum())
print(fielding_df.isnull().sum())

# Imputar o eliminar valores nulos
batting_df = batting_df.dropna()  # Eliminar filas con valores nulos

# Convertir columnas a tipos de datos apropiados si es necesario
batting_df['yearID'] = batting_df['yearID'].astype(int)

# Calcular la variable OBP (Porcentaje de embasado)
batting_df['OBP'] = (batting_df['H'] + batting_df['BB'] + batting_df['HBP']) / (batting_df['AB'] + batting_df['BB'] + batting_df['HBP'] + batting_df['SF'])

# Eliminar filas donde OBP sea NaN
batting_df = batting_df.dropna(subset=['OBP'])

# Normalización (ejemplo usando MinMaxScaler)
scaler = MinMaxScaler()
batting_df[['H', 'HR', 'RBI']] = scaler.fit_transform(batting_df[['H', 'HR', 'RBI']])

# 3.- DIVISIÓN DE DATOS
X = batting_df[['yearID', 'H', 'HR', 'RBI']]  # Variables independientes
y = batting_df['OBP']  # Variable dependiente (ejemplo: porcentaje de embasado)

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)


from sklearn.linear_model import LinearRegression, Ridge, Lasso
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.metrics import mean_squared_error, r2_score

# Inicializar los modelos
models = {
    'Linear Regression': LinearRegression(),
    'Ridge Regression': Ridge(alpha=1.0),  # Puedes ajustar alpha para regularización
    'Lasso Regression': Lasso(alpha=0.1),  # Puedes ajustar alpha para regularización
    'Decision Tree': DecisionTreeRegressor(),
    'Random Forest': RandomForestRegressor(n_estimators=100, random_state=42),
    'Gradient Boosting': GradientBoostingRegressor(n_estimators=100, learning_rate=0.1, max_depth=3, random_state=42)
}

# Entrenar y evaluar los modelos
results = {}
cross_val_results = {}

for name, model in models.items():
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)

    mse = mean_squared_error(y_test, y_pred)
    r2 = r2_score(y_test, y_pred)
    
    # Almacenar resultados
    results[name] = {'MSE': mse, 'R^2': r2}

    # Validación cruzada
    cross_val_scores = cross_val_score(model, X, y, cv=5, scoring='r2')
    cross_val_results[name] = cross_val_scores
    
    print(f"Modelo: {name}")
    print(f"MSE: {mse}")
    print(f"R^2: {r2}")
    print(f"R^2 promedio (CV): {cross_val_scores.mean()}")
    print("-------------------------")

# 4.- VISUALIZACIÓN DE RESULTADOS
# Comparación de MSE y R^2
mse_values = [results[name]['MSE'] for name in results]
r2_values = [results[name]['R^2'] for name in results]

# Gráfico de MSE
plt.figure(figsize=(12, 6))
plt.bar(results.keys(), mse_values, color='skyblue')
plt.title('Comparación de MSE entre Modelos')
plt.xlabel('Modelo')
plt.ylabel('MSE')
plt.xticks(rotation=45)
plt.show()

# Gráfico de R^2
plt.figure(figsize=(12, 6))
plt.bar(results.keys(), r2_values, color='lightgreen')
plt.title('Comparación de R^2 entre Modelos')
plt.xlabel('Modelo')
plt.ylabel('R^2')
plt.xticks(rotation=45)
plt.show()

# Boxplot de distribución de R^2 en validación cruzada
plt.figure(figsize=(12, 6))
data = [(name, score) for name, scores in cross_val_results.items() for score in scores]
df = pd.DataFrame(data, columns=['Modelo', 'R^2'])
sns.boxplot(x='Modelo', y='R^2', data=df)
plt.title('Distribución de R^2 en Validación Cruzada por Modelo')
plt.xticks(rotation=45)
plt.show()
