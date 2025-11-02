import json
import pandas as pd
from faker import Faker

# Cargar configuración
with open('config/config.json') as f:
    config = json.load(f)

# Generar datos sintéticos
fake = Faker()
data = []
for _ in range(1000):
    record = {
        'name': fake.name(),
        'age': fake.random_int(min=18, max=90),
        'income': fake.random_int(min=20000, max=200000),
        'target': fake.random_element(elements=(0, 1))
    }
    data.append(record)

# Guardar datos sintéticos
df = pd.DataFrame(data)
df.to_csv(config['synthetic_data_path'], index=False)
