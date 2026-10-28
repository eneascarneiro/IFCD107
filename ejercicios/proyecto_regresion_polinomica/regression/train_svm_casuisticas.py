"""Ejemplo comentado: primero crea el informe EDA y después entrena un SVR.

Desde la carpeta raíz del proyecto:

    .venv\\Scripts\\python.exe regression\\train_svm_casuisticas.py

El programa recorre los seis CSV de prácticas que están en el proyecto LaTeX.
Cada conjunto recibe su propio informe, modelo, predicciones y métricas en runs/.
"""

# Path sirve para construir rutas de carpetas sin depender desde dónde se lance
# el programa. Las demás bibliotecas se usan para leer datos, preparar el
# preprocesamiento y construir el modelo SVR.
from pathlib import Path
import csv
import json
import pickle
import sys

# Añadir la raíz a la ruta de imports permite encontrar regression/ cuando el
# programa se ejecuta por ruta desde scripts/.
PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

import numpy as np
import pandas as pd
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import train_test_split

from regression.eda_preprocesado import EdaPreprocesado
from regression.tabular_preprocessing import build_tabular_preprocessor
from regression.svm_regressor import build_svm_regression_pipeline


# La carpeta de datos y la salida se construyen desde la raíz del proyecto, así
# no dependen de la carpeta actual de la consola.
DATA_FOLDER = (PROJECT_ROOT / "proyecto_eda_preprocesado_latex" / "datos")
OUTPUT_FOLDER = PROJECT_ROOT / "runs" / "svm_casuisticas"

# Fijar una semilla hace reproducibles la partición train/test y las métricas.
RANDOM_STATE = 42
TEST_SIZE = 0.20
TARGET_COLUMN = "y"
ID_COLUMN = "expediente_id"

# Se especifican los seis archivos explícitamente para que sea fácil para un
# principiante ver qué conjuntos se van a analizar y entrenar.
DATASETS = [
    "01_referencia_limpia.csv",
    "02_valores_ausentes.csv",
    "03_valores_atipicos.csv",
    "04_categorias_raras.csv",
    "05_duplicados.csv",
    "06_sesgo_y_heterocedasticidad.csv",
]


def prepare_output_folder(dataset_name: str) -> Path:
    """Crea y devuelve la carpeta de resultados de un CSV."""
    # Cada conjunto usa su propia carpeta para no mezclar modelos o predicciones.
    dataset_folder = OUTPUT_FOLDER / Path(dataset_name).stem
    dataset_folder.mkdir(parents=True, exist_ok=True)
    return dataset_folder


def train_one_dataset(original_data: pd.DataFrame, dataset_name: str,
                      output_folder: Path, run_id: str,
                      processor: EdaPreprocesado | None = None) -> dict:
    """Genera el informe primero y luego entrena un SVR con una tabla.

    Esta función se puede llamar desde la API o desde la línea de comandos.
    Recibe una tabla ya leída, el nombre de origen, una carpeta de salida y
    un identificador único para nombrar los archivos generados.
    """
    print(f"\nProcesando {dataset_name}...")

    # Crear las subcarpetas donde se ordenan entrada, informe, predicciones y modelo.
    output_folder = Path(output_folder)
    folders = {
        "informes": output_folder / "informes",
        "csv_salida": output_folder / "csv_salida",
        "modelos": output_folder / "modelos",
    }
    for folder in folders.values():
        folder.mkdir(parents=True, exist_ok=True)

    # Copiar la tabla impide que las transformaciones cambien el objeto del llamador.
    original_data = original_data.copy()
    if original_data.empty:
        raise ValueError("El CSV no contiene filas.")
    processor = processor or EdaPreprocesado()

    # Paso 1: convertir categorías a números y representar los meses como ciclo.
    # La función también devuelve los mapas usados para poder interpretar códigos.
    prepared_data, category_maps = processor.transform(original_data)
    processed_csv_path = folders["csv_salida"] / f"{run_id}_preprocesado.csv"
    prepared_data.to_csv(processed_csv_path, index=False, encoding="utf-8-sig")

    # Paso 2: escribir el informe antes de ajustar el modelo.
    report_path = folders["informes"] / f"{run_id}_informe_eda.pdf"
    processor.generate_report(original_data, prepared_data, category_maps,
                             report_path, dataset_name)
    print(f"  Informe generado antes del ajuste: {report_path}")

    # Paso 3: comprobar la columna objetivo y convertir sus valores a números.
    if TARGET_COLUMN not in prepared_data.columns:
        raise ValueError(f"No existe la columna objetivo '{TARGET_COLUMN}'.")
    training_data = prepared_data.copy()
    training_data[TARGET_COLUMN] = pd.to_numeric(training_data[TARGET_COLUMN], errors="coerce")
    # Filas sin y numérico no se pueden usar para aprender un valor objetivo.
    training_data = training_data.dropna(subset=[TARGET_COLUMN]).reset_index(drop=True)
    if len(training_data) < 10:
        raise ValueError("Se necesitan al menos 10 filas con y numérico para entrenar.")

    # Paso 5: separar las entradas X del valor que queremos predecir, y.
    # El identificador sirve para reconocer la fila, pero no informa al modelo.
    feature_columns = [
        column for column in training_data.columns
        if column not in {TARGET_COLUMN, ID_COLUMN}
    ]
    if not feature_columns:
        raise ValueError("No quedan columnas predictoras para entrenar el SVR.")
    X = training_data[feature_columns].copy()
    y = training_data[TARGET_COLUMN].astype(float)

    # Los valores infinitos se tratan como ausentes para que el imputador los
    # rellene con la mediana aprendida de las filas de entrenamiento.
    numeric_columns = X.select_dtypes(include=["number", "bool"]).columns.tolist()
    categorical_columns = [column for column in feature_columns if column not in numeric_columns]
    for column in numeric_columns:
        X[column] = X[column].replace([np.inf, -np.inf], np.nan)
    usable_columns = [column for column in feature_columns if X[column].notna().any()]
    X = X[usable_columns]
    numeric_columns = [column for column in numeric_columns if column in usable_columns]
    categorical_columns = [column for column in categorical_columns if column in usable_columns]

    # Paso 6: separar datos de entrenamiento y prueba ANTES de ajustar el
    # imputador/escalador. El test queda fuera del aprendizaje y mide generalización.
    train_indices, test_indices = train_test_split(
        np.arange(len(training_data)), test_size=TEST_SIZE,
        random_state=RANDOM_STATE,
    )

    # Paso 7: crear las recetas de preprocesado. SVR usa distancias, por eso se
    # estandarizan las columnas numéricas; categorías restantes se codifican.
    preprocessor = build_tabular_preprocessor(
        numeric_columns, categorical_columns, scale_numeric=True
    )

    # Paso 8: crear el estimador SVR. RBF permite aprender relaciones curvas.
    options = {
        "kernel": "rbf",
        "C": 1.0,
        "epsilon": 0.1,
        "gamma": "scale",
        "degree": 3,
    }
    model = build_svm_regression_pipeline(preprocessor, options)

    # fit aprende el escalado/imputación y ajusta SVR solo con train.
    model.fit(X.iloc[train_indices], y.iloc[train_indices])

    # predict calcula una estimación para cada fila reservada en test.
    predicted = model.predict(X.iloc[test_indices])
    observed = y.iloc[test_indices].to_numpy()
    residuals = observed - predicted

    # Métricas de regresión: R² (ajuste), RMSE/MAE (tamaño del error) y sesgo.
    metrics = {
        "r2": float(r2_score(observed, predicted)),
        "rmse": float(np.sqrt(mean_squared_error(observed, predicted))),
        "mae": float(mean_absolute_error(observed, predicted)),
        "bias": float(np.mean(residuals)),
    }

    # Guardar cada predicción permite revisar en qué filas se equivoca el modelo.
    predictions = pd.DataFrame({
        "y_observado": observed,
        "y_predicho": predicted,
        "residuo": residuals,
    })
    if ID_COLUMN in training_data:
        predictions.insert(0, ID_COLUMN, training_data[ID_COLUMN].iloc[test_indices].to_numpy())
    predictions_path = folders["csv_salida"] / f"{run_id}_predicciones.csv"
    predictions.to_csv(predictions_path, index=False, encoding="utf-8-sig")

    # Serializar el pipeline conserva juntos el preprocesado y el modelo SVR.
    # Solo se deben cargar modelos pickle generados por fuentes de confianza.
    model_path = folders["modelos"] / f"{run_id}_svr.pkt"
    with model_path.open("wb") as model_file:
        pickle.dump(model, model_file, protocol=pickle.HIGHEST_PROTOCOL)

    result = {
        "dataset": dataset_name,
        "filas_originales": int(len(original_data)),
        "filas_con_objetivo": int(len(training_data)),
        "filas_train": int(len(train_indices)),
        "filas_test": int(len(test_indices)),
        "variables": usable_columns,
        "parametros_svr": options,
        "metricas": metrics,
        "artifacts": {
            "eda_report_pdf": str(report_path),
            "preprocessed_csv": str(processed_csv_path),
            "predictions_csv": str(predictions_path),
            "trained_model_pkt": str(model_path),
        },
        "codificaciones": category_maps,
    }
    metrics_path = output_folder / "logs_salida" / f"{run_id}_metricas.json"
    metrics_path.parent.mkdir(parents=True, exist_ok=True)
    (result["artifacts"]).update({"metrics_json": str(metrics_path)})
    metrics_path.write_text(
        json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    print(f"  SVR entrenado: R²={metrics['r2']:.4f}, RMSE={metrics['rmse']:.4f}, MAE={metrics['mae']:.4f}")
    return result


def main() -> None:
    """Ejecuta el flujo para los seis archivos y resume las métricas."""
    if not DATA_FOLDER.is_dir():
        raise FileNotFoundError(f"No existe la carpeta de CSV: {DATA_FOLDER}")
    processor = EdaPreprocesado()
    results = []

    # El bucle procesa una casuística cada vez: informe primero, SVR después.
    for dataset_name in DATASETS:
        csv_path = DATA_FOLDER / dataset_name
        if not csv_path.is_file():
            print(f"AVISO: no se encontró {csv_path}; se omite este archivo.")
            continue
        try:
            # La función común también la usa la API; aquí la llamamos con las
            # filas del CSV y una carpeta separada para este caso de estudio.
            original_data = pd.read_csv(csv_path, encoding="utf-8-sig")
            case_folder = prepare_output_folder(csv_path.name)
            result = train_one_dataset(
                original_data, csv_path.name, case_folder, "ejecucion_local", processor
            )
            # El resumen del lote guarda solo valores simples, no mapas completos.
            results.append({key: result[key] for key in (
                "dataset", "filas_originales", "filas_con_objetivo",
                "filas_train", "filas_test", "metricas"
            )})
        except Exception as error:
            # Se registra el error de una casuística y se continúa con las demás.
            print(f"  ERROR en {dataset_name}: {error}")
            results.append({"dataset": dataset_name, "error": str(error)})

    OUTPUT_FOLDER.mkdir(parents=True, exist_ok=True)
    summary_path = OUTPUT_FOLDER / "resumen_entrenamientos.csv"
    # El resumen plano abre directamente en Excel y facilita comparar datasets.
    with summary_path.open("w", newline="", encoding="utf-8-sig") as summary_file:
        columns = ["dataset", "filas_originales", "filas_con_objetivo", "filas_train",
                   "filas_test", "r2", "rmse", "mae", "bias", "error"]
        writer = csv.DictWriter(summary_file, fieldnames=columns, extrasaction="ignore")
        writer.writeheader()
        for result in results:
            row = dict(result)
            row.update(result.get("metricas", {}))
            writer.writerow(row)
    print(f"\nResumen comparativo: {summary_path}")


if __name__ == "__main__":
    main()
