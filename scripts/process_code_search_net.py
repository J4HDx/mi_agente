import os
import pandas as pd
from datasets import load_dataset
from utils.data_utils import load_data, preprocess_data, split_data

# Ruta donde guardaremos el dataset
raw_dir = "data/raw"
os.makedirs(raw_dir, exist_ok=True)
csv_path = os.path.join(raw_dir, "code_search_net_python.csv")

# Si no existe, lo descargamos de Hugging Face
if not os.path.exists(csv_path):
    print("📥 Descargando dataset 'code_search_net' (subset: python)...")
    dataset = load_dataset("code_search_net", "python")
    df = dataset["train"].to_pandas()
    df.to_csv(csv_path, index=False)
    print(f"✅ Guardado en {csv_path}")

# Ahora cargamos el CSV con tu función
print("📖 Cargando dataset...")
code_search_net_data = load_data(csv_path)

# Procesa, divide o guarda según tu flujo original
print("⚙️ Procesando dataset...")
processed_data = preprocess_data(code_search_net_data)
train_data, val_data, test_data = split_data(processed_data)

print("✅ Listo. Dataset procesado y dividido correctamente.")
