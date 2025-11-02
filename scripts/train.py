import os
import json
import pandas as pd
from datasets import load_dataset
from utils.data_utils import load_data, preprocess_data, split_data
from models.model_v1 import create_model

# -----------------------------
# 1️⃣ Cargar configuración
# -----------------------------
with open('config/config.json') as f:
    config = json.load(f)

# -----------------------------
# 2️⃣ Descargar dataset si no existe
# -----------------------------
raw_dir = 'data/raw/code_search_net/python'
os.makedirs(raw_dir, exist_ok=True)
csv_path = os.path.join(raw_dir, 'code_search_net_python.csv')

if not os.path.exists(csv_path):
    print("📥 Descargando dataset CodeSearchNet (python)...")
    dataset = load_dataset("code_search_net", "python")
    df = dataset["train"].to_pandas()
    df.to_csv(csv_path, index=False)
    print(f"✅ Dataset guardado en {csv_path}")

# -----------------------------
# 3️⃣ Cargar y preprocesar datos
# -----------------------------
print("📖 Cargando dataset...")
code_search_net_data = load_data(csv_path)

print("⚙️ Procesando dataset...")
processed_data = preprocess_data(code_search_net_data)

# Guardar datos procesados
processed_data_path = 'data/processed/code_search_net_processed.csv'
os.makedirs('data/processed', exist_ok=True)
processed_data.to_csv(processed_data_path, index=False)
print(f"✅ Datos procesados guardados en {processed_data_path}")

# -----------------------------
# 4️⃣ Dividir dataset en train/test
# -----------------------------
X_train, X_test, y_train, y_test = split_data(
    processed_data,
    test_size=config['test_size'],
    random_state=config['random_state']
)
print(f"✅ Dataset dividido: Train={len(X_train)}, Test={len(X_test)}")

# -----------------------------
# 5️⃣ Crear y entrenar modelo
# -----------------------------
model = create_model(input_shape=(X_train.shape[1],))
print("🏋️‍♂️ Entrenando modelo...")
model.fit(
    X_train,
    y_train,
    epochs=config['epochs'],
    batch_size=config['batch_size'],
    validation_split=0.2
)

# -----------------------------
# 6️⃣ Guardar modelo entrenado
# -----------------------------
os.makedirs('models', exist_ok=True)
model.save('models/model_v1.h5')
print("✅ Modelo guardado en models/model_v1.h5")
