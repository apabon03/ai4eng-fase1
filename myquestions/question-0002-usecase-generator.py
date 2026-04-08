import pandas as pd
import numpy as np
import random

def generar_caso_de_uso_transformar_tiempo_ciclico():
    """
    Genera un caso de prueba aleatorio (input y output esperado)
    para la función transformar_tempo_ciclico.
    """
    
    # ---------------------------------------------------------
    # 1. Configuración aleatoria
    # ---------------------------------------------------------
    
    n_rows = random.randint(5, 15)
    
    col_hora = random.choice(['hora', 'hour', 'time', 'h'])
    
    n_extra_cols = random.randint(1, 3)
    
    # ---------------------------------------------------------
    # 2. Generar datos
    # ---------------------------------------------------------
    
    horas = np.random.randint(0, 24, size=n_rows)
    
    df = pd.DataFrame({
        col_hora: horas
    })
    
    for i in range(n_extra_cols):
        df[f'feature_{i}'] = np.random.randn(n_rows)
    
    # ---------------------------------------------------------
    # 3. Construir INPUT
    # ---------------------------------------------------------
    
    input_data = {
        'df': df.copy(),
        'col_hora': col_hora
    }
    
    # ---------------------------------------------------------
    # 4. Calcular OUTPUT esperado
    # ---------------------------------------------------------
    
    df_expected = df.copy()
    
    angulo = 2 * np.pi * df_expected[col_hora] / 24
    
    df_expected['hora_sin'] = np.sin(angulo)
    df_expected['hora_cos'] = np.cos(angulo)
    
    df_expected = df_expected.drop(columns=[col_hora])
    
    output_data = df_expected
    
    return input_data, output_data


# ---------------------------------------------------------
# (Opcional) Ejemplo de uso
# ---------------------------------------------------------
if __name__ == "__main__":
    
    entrada, salida_esperada = generar_caso_de_uso_transformar_tiempo_ciclico()
    
    print("=== INPUT ===")
    print(f"Columna de hora: {entrada['col_hora']}")
    print(entrada['df'].head())
    
    print("\n=== OUTPUT ESPERADO ===")
    print(salida_esperada.head())
