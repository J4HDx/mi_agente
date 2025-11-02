import json
import pandas as pd
from utils.data_utils import load_data, preprocess_data, split_data
from models.model_v1 import create_model

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

# Crear modelo
model = create_model(input_shape=(X_train.shape[1],))

# Entrenar modelo
model.fit(X_train, y_train, epochs=config['epochs'], batch_size=config['batch_size'], validation_split=0.2)

# Guardar modelo
model.save('models/model_v1.h5')
