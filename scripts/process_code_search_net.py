import json
import pandas as pd
from utils.data_utils import load_data, preprocess_data, split_data

# Cargar configuración
with open('config/config.json') as f:
    config = json.load(f)

# Cargar y preprocesar datos de CodeSearchNet
code_search_net_path = 'data/raw/code_search_net/python/code_search_net_python.json'
code_search_net_data = load_data(code_search_net_path)

# Procesar datos
processed_data = preprocess_data(code_search_net_data)

# Guardar datos procesados
processed_data_path = 'data/processed/code_search_net_processed.csv'
processed_data.to_csv(processed_data_path, index=False)

# Dividir datos en conjuntos de entrenamiento y prueba
X_train, X_test, y_train, y_test = split_data(processed_data, test_size=config['test_size'], random_state=config['random_state'])

# Continuar con el entrenamiento del modelo
# Aquí puedes llamar a tu script de entrenamiento o continuar con el flujo de trabajo
# Por ejemplo: from scripts.train import train_model
# train_model(X_train, y_train, X_test, y_test)
