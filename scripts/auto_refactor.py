import os
import re

def refactor_code(file_path):
    with open(file_path, 'r') as file:
        lines = file.readlines()

    refactored_lines = []
    for line in lines:
        # Ejemplo de refactorización: eliminar espacios en blanco al final de las líneas
        refactored_line = line.rstrip()
        refactored_lines.append(refactored_line)

    with open(file_path, 'w') as file:
        file.writelines(refactored_lines)

# Refactorizar todos los archivos .py en el directorio 'models'
for file_name in os.listdir('models'):
    if file_name.endswith('.py'):
        refactor_code(os.path.join('models', file_name))
