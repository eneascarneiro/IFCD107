"""Ejemplo didáctico: ejecutar KNN con scikit-learn y explicar sus métricas con Ollama.

KNN produce las predicciones. Ollama solo redacta una explicación del resumen
estadístico; no entrena, no cambia el modelo y no recibe las filas del CSV.
"""
from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
from urllib.error import HTTPError, URLError
from urllib.request import Request, urlopen

import numpy as np
import pandas as pd
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsRegressor
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import RobustScaler, StandardScaler


# La carpeta del ejemplo es ejemplos/; con este valor no hay que escribir
# una ruta absoluta si ejecutamos el script desde la raíz del proyecto.
PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_CSV = PROJECT_ROOT / "ejemplos" / "datos_knn_5000_outliers.csv"


def explain_with_ollama(summary: dict, model: str, base_url: str) -> str:
    """Pide al modelo local una explicación breve de métricas ya calculadas."""
    prompt = (
        "Explica en español, para una persona principiante, estos resultados "
        "de un modelo KNN de regresión. Describe R2, RMSE y MAE en palabras "
        "sencillas. Menciona con prudencia que los datos tienen outliers y que "
        "las métricas no prueban causalidad. No inventes resultados ni "
        "recomiendes un modelo solo por una métrica.\n\n"
        f"Resumen calculado por scikit-learn: {json.dumps(summary, ensure_ascii=False)}"
    )

    # La API oficial genera una respuesta en /api/generate. Con stream=False
    # recibimos un único objeto JSON, que es más sencillo para este ejemplo.
    payload = {
        "model": model,
        "prompt": prompt,
        "stream": False,
        "options": {"temperature": 0.2},
    }
    request = Request(
        url=f"{base_url.rstrip('/')}/api/generate",
        data=json.dumps(payload).encode("utf-8"),
        headers={"Content-Type": "application/json"},
        method="POST",
    )

    # Ollama debe estar instalado, ejecutándose y tener el modelo descargado.
    with urlopen(request, timeout=180) as response:
        result = json.loads(response.read().decode("utf-8"))
    return result.get("response", "").strip()


def main() -> None:
    """Carga datos, entrena KNN, mide test y opcionalmente llama a Ollama."""
    parser = argparse.ArgumentParser(
        description="Entrena KNN y pide a Ollama una explicación de las métricas."
    )
    parser.add_argument("--csv", type=Path, default=DEFAULT_CSV, help="Ruta al CSV.")
    parser.add_argument("--target", default="y", help="Columna numérica objetivo.")
    parser.add_argument("--neighbors", type=int, default=5, help="Número de vecinos.")
    parser.add_argument(
        "--scaler",
        choices=("standard", "robust"),
        default="standard",
        help="Escalado: standard replica el ejemplo KNN del proyecto.",
    )
    parser.add_argument(
        "--model",
        default=os.environ.get("OLLAMA_MODEL", "llama3.2"),
        help="Nombre de un modelo ya descargado en Ollama.",
    )
    parser.add_argument(
        "--ollama-url",
        default=os.environ.get("OLLAMA_URL", "http://localhost:11434"),
        help="Dirección del servidor local Ollama.",
    )
    parser.add_argument(
        "--skip-ollama",
        action="store_true",
        help="Ejecuta KNN y métricas sin pedir explicación a Ollama.",
    )
    args = parser.parse_args()

    # Leer la tabla desde CSV. El dataset del curso tiene x1,x2,x3,x4 e y.
    if not args.csv.exists():
        raise SystemExit(
            f"No encuentro el CSV: {args.csv}\n"
            "Usa --csv para indicar dónde está datos_knn_5000_outliers.csv."
        )
    table = pd.read_csv(args.csv)
    if args.target not in table.columns:
        raise SystemExit(f"No existe la columna objetivo {args.target!r}.")

    # Las columnas del proyecto son numéricas. Excluir también un ID si el
    # usuario entrega otra tabla con una columna de identificadores.
    excluded = {args.target, "expediente_id"}
    feature_columns = [name for name in table.columns if name not in excluded]
    if not feature_columns:
        raise SystemExit("El CSV debe tener al menos una columna predictora.")

    try:
        X = table[feature_columns].apply(pd.to_numeric, errors="raise")
        y = pd.to_numeric(table[args.target], errors="raise")
    except (TypeError, ValueError) as error:
        raise SystemExit(f"KNN requiere columnas numéricas: {error}") from None

    # Para que el ejemplo sea fácil de entender, informamos si hay celdas
    # ausentes o infinitas en vez de ocultar cómo se limpian.
    if X.isna().any().any() or y.isna().any():
        raise SystemExit("El CSV contiene celdas vacías; completa o imputa esos datos.")
    if not (
        np.isfinite(X.to_numpy(dtype=float)).all()
        and np.isfinite(y.to_numpy(dtype=float)).all()
    ):
        raise SystemExit("El CSV contiene infinito; corrige esos valores antes de entrenar.")

    if args.neighbors < 1:
        raise SystemExit("--neighbors debe ser al menos 1.")
    if args.neighbors >= len(table):
        raise SystemExit("--neighbors debe ser menor que el número de filas.")

    # Dejar un 20 % aparte. Esas filas no participan en el ajuste.
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )

    # StandardScaler replica el pipeline del proyecto. RobustScaler puede
    # compararse en prácticas con outliers porque usa mediana y rango intercuartil.
    scaler = StandardScaler() if args.scaler == "standard" else RobustScaler()
    model = Pipeline(
        steps=[
            ("scale", scaler),
            ("knn", KNeighborsRegressor(n_neighbors=args.neighbors)),
        ]
    )

    # fit aprende el escalado y los vecinos usando solo entrenamiento.
    model.fit(X_train, y_train)
    # predict estima los valores de las filas reservadas para prueba.
    predictions = model.predict(X_test)

    # Calcular métricas con valores reales, no con texto generado por el LLM.
    metrics = {
        "r2": float(r2_score(y_test, predictions)),
        "rmse": float(mean_squared_error(y_test, predictions) ** 0.5),
        "mae": float(mean_absolute_error(y_test, predictions)),
    }
    summary = {
        "algorithm": "KNeighborsRegressor",
        "csv": args.csv.name,
        "rows_total": int(len(table)),
        "rows_train": int(len(X_train)),
        "rows_test": int(len(X_test)),
        "features": feature_columns,
        "n_neighbors": args.neighbors,
        "scaler": args.scaler,
        "metrics": metrics,
        "outliers_note": "El fichero de ejemplo contiene outliers añadidos para prácticas.",
    }

    print("Métricas calculadas por scikit-learn:")
    print(json.dumps(summary, indent=2, ensure_ascii=False))

    if args.skip_ollama:
        return

    try:
        explanation = explain_with_ollama(summary, args.model, args.ollama_url)
        print("\nExplicación generada localmente por Ollama:")
        print(explanation)
    except (HTTPError, URLError, TimeoutError, json.JSONDecodeError) as error:
        print(
            "\nKNN terminó correctamente, pero no se pudo conectar con Ollama. "
            "Comprueba que Ollama está iniciado y que el modelo está descargado."
        )
        print(f"Detalle de conexión: {error}")


if __name__ == "__main__":
    main()
