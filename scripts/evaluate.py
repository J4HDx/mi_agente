import json
import tensorflow as tf
from utils.data_utils import load_data, preprocess_data, split_data

# Cargar configuración
with open('config/config.json') as f:
    config = json.load(f)

# Cargar y preprocesar datos
data = load_data(config['data_path'])
data = preprocess_data(data)

# Dividir datos en conjuntos de entrenamiento y prueba
X_train, X_test, y_train, y_test = split_data(data, test_size=config['test_size'], random_state=config['random_state'])

# Cargar modelo
model = tf.keras.models.load_model('models/model_v1.h5')

# Evaluar modelo
loss, accuracy = model.evaluate(X_test, y_test)
print(f'Loss: {loss}, Accuracy: {accuracy}')
