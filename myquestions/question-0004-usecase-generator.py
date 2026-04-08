import pandas as pd
import numpy as np
import random

from sklearn.ensemble import RandomForestClassifier
from sklearn.inspection import permutation_importance
from sklearn.model_selection import train_test_split

def generar_caso_importancia_permutacion():
    """
    Genera un caso de prueba aleatorio (input y output esperado)
    para la función calcular_importancia_permutacion.
    """
    
    # ---------------------------------------------------------
    # 1. Configuración aleatoria
    # ---------------------------------------------------------
    
    n_rows = random.randint(50, 150)
    n_features = random.randint(3, 6)
    
    # ---------------------------------------------------------
    # 2. Generar datos aleatorios
    # ---------------------------------------------------------
    
    X = np.random.randn(n_rows, n_features)
    y = np.random.randint(0, 2, size=n_rows)
    
    # Opcional: usar DataFrame para hacerlo más realista
    if random.choice([True, False]):
        feature_cols = [f'feature_{i}' for i in range(n_features)]
        X = pd.DataFrame(X, columns=feature_cols)
    
    # ---------------------------------------------------------
    # 3. Split train / validation
    # ---------------------------------------------------------
    
    X_train, X_val, y_train, y_val = train_test_split(
        X, y, test_size=0.3, random_state=42
    )
    
    # ---------------------------------------------------------
    # 4. Entrenar modelo base
    # ---------------------------------------------------------
    
    modelo = RandomForestClassifier(random_state=42)
    modelo.fit(X_train, y_train)
    
    # ---------------------------------------------------------
    # 5. Construir INPUT
    # ---------------------------------------------------------
    
    input_data = {
        'modelo': modelo,
        'X_val': X_val.copy(),
        'y_val': y_val.copy()
    }
    
    # ---------------------------------------------------------
    # 6. Calcular OUTPUT esperado (Ground Truth)
    # ---------------------------------------------------------
    
    resultado = permutation_importance(
        modelo,
        X_val,
        y_val,
        n_repeats=5,
        random_state=42
    )
    
    output_data = resultado.importances_mean
    
    return input_data, output_data


# ---------------------------------------------------------
# Ejemplo de uso
# ---------------------------------------------------------
if __name__ == "__main__":
    
    entrada, salida_esperada = generar_caso_importancia_permutacion()
    
    print("=== INPUT ===")
    print(f"Modelo: {type(entrada['modelo'])}")
    print(f"Shape X_val: {entrada['X_val'].shape}")
    print(f"Shape y_val: {entrada['y_val'].shape}")
    
    print("\n=== OUTPUT ESPERADO ===")
    print("Importancias promedio:")
    print(salida_esperada)
