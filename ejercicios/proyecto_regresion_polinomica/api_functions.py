"""Lógica que comparten las rutas HTTP de la API.

Este módulo traduce entradas externas (JSON y CSV) a datos tabulares seguros,
orquesta el ajuste de los modelos y guarda los resultados. No define las rutas:
esas pertenecen a app.py. Tampoco implementa cada algoritmo: los constructores
de modelos están en regression/.
"""
from __future__ import annotations

import csv
import io
import logging
import re
import shutil
import subprocess
import sys
import tempfile
from datetime import datetime, timezone
from pathlib import Path
from uuid import uuid4
from sklearn.model_selection import train_test_split
import json
import numpy as np
from flask import g, jsonify
from scipy.stats import pearsonr

from regression.knn_regressor import build_knn_pipeline
from regression.polynomial_regressor import PolynomialRegressor,build_polynomial_search
from regression.random_forest_regressor import RandomForestRegressorModel
from regression.train_svm_casuisticas import train_one_dataset
from regression.tabular_preprocessing import build_tabular_preprocessor
from regression.svm_regressor import build_svm_regression_pipeline
from regression.adaboost_regressor import build_adaboost_regression_pipeline
from regression.gradient_boosting_regressor import build_gradient_boosting_regression_pipeline
from regression.parameter_search import PARAMETER_CATALOG
from regression.train_regresion_lineal_simple_ejercicio_casuisticas import train_algorithm_frame_lineal_regression

# Límites del procesamiento aplicados tanto a endpoints JSON como a CSV.
MAX_POINTS = 100_000
REFERENCE_RE = re.compile(r"^[A-Za-z0-9][A-Za-z0-9._-]{0,63}$")
PRIMITIVA_SCRIPTS = {
    "v01": Path(__file__).resolve().parent / "primitiva" / "primitiva.py",
    "v02": Path(__file__).resolve().parent / "primitiva" / "primitiva_v02.py",
}

def run_primitiva_model(output_root, reference, raw_csv, model_name, options):
    """Ajusta un modelo didáctico KNN/CNN y conserva sus resultados como JSON."""
    from primitiva.predictors import analyze_draws, load_draw_matrix

    run_id = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ") + "_" + uuid4().hex[:8]
    output_root = Path(output_root)
    base = output_root / reference / "primitiva"
    input_dir, result_dir, log_dir = (base / name for name in ("csv_entrada", "predicciones", "logs_salida"))
    for directory in (input_dir, result_dir, log_dir):
        directory.mkdir(parents=True, exist_ok=True)
    input_path = input_dir / f"{run_id}.csv"
    result_path = result_dir / f"{run_id}_{model_name}.json"
    log_path = log_dir / f"{run_id}_{model_name}.log"
    input_path.write_bytes(raw_csv)
    try:
        draws = load_draw_matrix(raw_csv)
        result = analyze_draws(draws, model_name=model_name, **options)
        result.update({"status": "completed", "reference": reference, "run_id": run_id})
        result["artifacts"] = {
            "input_csv": str(input_path.relative_to(output_root)),
            "result_json": str(result_path.relative_to(output_root)),
            "execution_log": str(log_path.relative_to(output_root)),
        }
        result_path.write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")
        log_path.write_text(
            f"run_id={run_id}\nmodel={model_name}\noptions={json.dumps(options, ensure_ascii=False)}\n"
            f"historical_draws={len(draws)}\nmetrics={json.dumps(result['metrics'], ensure_ascii=False)}\n",
            encoding="utf-8",
        )
        return jsonify(result), 200
    except ValueError as exc:
        log_path.write_text(f"run_id={run_id}\nmodel={model_name}\nerror={exc}\n", encoding="utf-8")
        return jsonify(error=str(exc), reference=reference, run_id=run_id), 422
    except Exception:
        logging.getLogger(__name__).exception("Error al analizar La Primitiva con %s.", model_name)
        log_path.write_text(f"run_id={run_id}\nmodel={model_name}\nerror=Error interno\n", encoding="utf-8")
        return jsonify(error="Error interno durante el análisis del modelo.", reference=reference,
                       run_id=run_id), 500


def run_regression_casuistica(output_root, reference, raw_csv, algorithm,
                              cv=5, n_iter=20, random_state=42):
    """Ejecuta una casuística de regresión elegida por slug validado en Flask."""
    import io
    import pandas as pd
    from regression.casuisticas import ALGORITHM_LABELS, train_algorithm_frame

    if algorithm not in ALGORITHM_LABELS:
        return jsonify(error="Algoritmo de casuística no reconocido."), 404
    run_id = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ") + "_" + uuid4().hex[:8]
    output_root = Path(output_root)
    base = output_root / reference / "casuisticas" / algorithm
    input_dir = base / "csv_entrada"
    log_dir = base / "logs_salida"
    input_dir.mkdir(parents=True, exist_ok=True)
    log_dir.mkdir(parents=True, exist_ok=True)
    input_path = input_dir / f"{run_id}.csv"
    log_path = log_dir / f"{run_id}.log"
    input_path.write_bytes(raw_csv)
    try:
        frame = pd.read_csv(io.BytesIO(raw_csv), encoding="utf-8-sig")
        if len(frame) > MAX_POINTS:
            raise ValueError(f"El CSV supera el límite de {MAX_POINTS} filas.")
        result = train_algorithm_frame(
            frame, algorithm, base / "resultados", run_id,
            cv=cv, n_iter=n_iter, random_state=random_state,
        )
        artifact_paths = {
            key: str(Path(value).relative_to(output_root))
            for key, value in result["artifacts"].items()
        }
        artifact_paths.update({
            "input_csv": str(input_path.relative_to(output_root)),
            "log": str(log_path.relative_to(output_root)),
        })
        result.update({"reference": reference, "run_id": run_id, "artifacts": artifact_paths})
        log_path.write_text(
            f"algorithm={algorithm}\nrun_id={run_id}\ncv={cv}\nn_iter={n_iter}\n"
            f"winner={result['winner']}\nbest_parameters={json.dumps(result['best_parameters'], default=str)}\n"
            f"metrics={json.dumps(result['metrics'])}\n",
            encoding="utf-8",
        )
        return jsonify(result), 200
    except (ValueError, UnicodeDecodeError, pd.errors.ParserError, pd.errors.EmptyDataError) as exc:
        log_path.write_text(f"algorithm={algorithm}\nrun_id={run_id}\nerror={exc}\n", encoding="utf-8")
        return jsonify(error=f"No se pudo entrenar la casuística: {exc}",
                       reference=reference, run_id=run_id), 422
    except Exception:
        logging.getLogger(__name__).exception("Error entrenando la casuística %s.", algorithm)
        log_path.write_text(f"algorithm={algorithm}\nrun_id={run_id}\nerror=Error interno\n", encoding="utf-8")
        return jsonify(error="Error interno durante el entrenamiento de la casuística.",
                       reference=reference, run_id=run_id), 500

def run_regression_casuistica_ejercicio(output_root, reference, raw_csv, algorithm,
                              cv=5, n_iter=20, random_state=42):
    """Ejecuta una casuística de regresión elegida por slug validado en Flask."""
    import io
    import pandas as pd
    from regression.casuisticas import ALGORITHM_LABELS, train_algorithm_frame

    if algorithm not in ALGORITHM_LABELS:
        return jsonify(error="Algoritmo de casuística no reconocido."), 404
    run_id = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ") + "_" + uuid4().hex[:8]
    output_root = Path(output_root)
    base = output_root / reference / "casuisticas" / algorithm
    input_dir = base / "csv_entrada"
    log_dir = base / "logs_salida"
    input_dir.mkdir(parents=True, exist_ok=True)
    log_dir.mkdir(parents=True, exist_ok=True)
    input_path = input_dir / f"{run_id}.csv"
    log_path = log_dir / f"{run_id}.log"
    input_path.write_bytes(raw_csv)
    # aqui empiezan loa cambios del ejercicio 
    # por cada nombre de algoritmo crear un método en
    # el py correspondiente y llamarlo
    try:
        frame = pd.read_csv(io.BytesIO(raw_csv), encoding="utf-8-sig")
        if len(frame) > MAX_POINTS:
            raise ValueError(f"El CSV supera el límite de {MAX_POINTS} filas.")
        # como solo hay un algoritmo llamo al de regresion lineal 
        result = train_algorithm_frame_lineal_regression(
            frame, base / "resultados", run_id,
            cv=cv, n_iter=n_iter, random_state=random_state,
        )
        artifact_paths = {
            key: str(Path(value).relative_to(output_root))
            for key, value in result["artifacts"].items()
        }
        artifact_paths.update({
            "input_csv": str(input_path.relative_to(output_root)),
            "log": str(log_path.relative_to(output_root)),
        })
        result.update({"reference": reference, "run_id": run_id, "artifacts": artifact_paths})
        log_path.write_text(
            f"algorithm={algorithm}\nrun_id={run_id}\ncv={cv}\nn_iter={n_iter}\n"
            f"winner={result['winner']}\nbest_parameters={json.dumps(result['best_parameters'], default=str)}\n"
            f"metrics={json.dumps(result['metrics'])}\n",
            encoding="utf-8",
        )
        return jsonify(result), 200
    except (ValueError, UnicodeDecodeError, pd.errors.ParserError, pd.errors.EmptyDataError) as exc:
        log_path.write_text(f"algorithm={algorithm}\nrun_id={run_id}\nerror={exc}\n", encoding="utf-8")
        return jsonify(error=f"No se pudo entrenar la casuística: {exc}",
                       reference=reference, run_id=run_id), 422
    except Exception:
        logging.getLogger(__name__).exception("Error entrenando la casuística %s.", algorithm)
        log_path.write_text(f"algorithm={algorithm}\nrun_id={run_id}\nerror=Error interno\n", encoding="utf-8")
        return jsonify(error="Error interno durante el entrenamiento de la casuística.",
                       reference=reference, run_id=run_id), 500


def validate_reference(value):
    """Acepta una referencia segura para formar nombres de carpetas y archivos."""
    # La validación se aplica antes de formar cualquier ruta. No se normaliza una
    # referencia inválida porque eso podría hacer colisionar expedientes distintos.
    if not isinstance(value, str) or not REFERENCE_RE.fullmatch(value):
        return None, "reference es obligatorio y debe tener 1-64 caracteres: letras, números, punto, guion o guion bajo."
    return value, None

def validate_reference_v1(value,txterror):
    """Acepta una referencia segura para formar nombres de carpetas y archivos."""
    # La validación se aplica antes de formar cualquier ruta. No se normaliza una
    # referencia inválida porque eso podría hacer colisionar expedientes distintos.
    if not isinstance(value, str) :
        return None, txterror
    return value, None


def parse_json_points(points):
    """Convierte puntos JSON en dos vectores numéricos y comprueba sus valores."""
    # Limitar número de filas antes de convertirlas a arrays evita asignaciones
    # grandes no necesarias; después se comprueba tipo y finitud en validate_xy.
    if not isinstance(points, list) or not 2 <= len(points) <= MAX_POINTS:
        raise ValueError(f"data debe contener entre 2 y {MAX_POINTS} pares x/y.")
    try:
        x = np.asarray([row["x"] for row in points], dtype=float)
        y = np.asarray([row["y"] for row in points], dtype=float)
    except (TypeError, ValueError, KeyError, IndexError):
        raise ValueError("Cada elemento de data debe tener valores numéricos x e y.") from None
    return validate_xy(x, y)

def parse_integer(name, default, minimum, maximum, source):
    """Lee un entero tanto de JSON como de un formulario HTML/Postman."""
    # Los números JSON llegan como int; los campos de un formulario llegan
    # como texto. Convertimos ambos formatos y después aplicamos el límite.
    raw = source.get(name, default)
    try:
        if isinstance(raw, bool) or (isinstance(raw, str) and not raw.strip()):
            raise ValueError
        parsed = int(raw)
        if isinstance(raw, float) and raw != parsed:
            raise ValueError
    except (TypeError, ValueError):
        raise ValueError(f"{name} debe ser un entero entre {minimum} y {maximum}.") from None
    if not minimum <= parsed <= maximum:
        raise ValueError(f"{name} debe ser un entero entre {minimum} y {maximum}.")
    return parsed

def parse_search_options(source):
    """Valida el interruptor y el presupuesto de una búsqueda de hiperparámetros."""
    raw = source.get("search", False)
    if isinstance(raw, bool):
        enabled = raw
    elif isinstance(raw, str) and raw.strip().lower() in {"true", "1", "yes", "si", "sí"}:
        enabled = True
    elif isinstance(raw, str) and raw.strip().lower() in {"false", "0", "no", ""}:
        enabled = False
    else:
        raise ValueError("search debe ser true o false.")
    return {
        "search": enabled,
        "n_iter": parse_integer("n_iter", 10, 1, 100, source),
        "search_cv": parse_integer("search_cv", 5, 2, 10, source),
    }

def parse_csv(text):
    """Extrae x/y de un CSV para los modelos polinómicos de una variable."""
    # DictReader permite conservar columnas contextuales del caso de terrazas,
    # aunque el algoritmo solo use x e y. Las filas extra no se descartan del
    # CSV archivado; solo se extraen aquí las dos variables del modelo.
    reader = csv.DictReader(io.StringIO(text))
    if not reader.fieldnames or not {"x", "y"}.issubset(set(reader.fieldnames)):
        raise ValueError("El encabezado debe contener las columnas x,y.")
    x_values, y_values = [], []
    for line_number, row in enumerate(reader, start=2):
        try:
            x_values.append(float(row["x"]))
            y_values.append(float(row["y"]))
        except (TypeError, ValueError):
            raise ValueError(f"Valores no numéricos en la línea {line_number}.") from None
        if len(x_values) > MAX_POINTS:
            raise ValueError(f"El CSV supera {MAX_POINTS} filas de datos.")
    return validate_xy(np.asarray(x_values), np.asarray(y_values))

def validate_xy(x, y):
    """Rechaza vectores x/y vacíos, desalineados o con NaN/infinito."""
    # NaN e infinito contaminan coeficientes y métricas sin producir siempre una
    # excepción clara; se rechazan en el borde HTTP antes de invocar NumPy.
    if x.ndim != 1 or y.ndim != 1 or len(x) != len(y) or len(x) < 2:
        raise ValueError("Se requieren al menos dos pares x/y del mismo tamaño.")
    if not np.isfinite(x).all() or not np.isfinite(y).all():
        raise ValueError("x e y solo pueden contener números finitos.")
    return x, y
def preparar_carpetas_y_logs(logger,output_root,reference,run_id=None):
    # - Generar run_id único con fecha UTC y sufijo aleatorio
    run_id = run_id or datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ") + "_" + uuid4().hex[:8]
    # - Combinar OUTPUT_ROOT con reference previamente validada.
    base = Path(output_root) / reference
    folders = {name: base / name for name in
               ("graficas", "csv_entrada", "logs_salida", "csv_salida", "modelos")}
    # - Crear subcarpetas de entrada, salida, modelos, gráficos y logs.
    for folder in folders.values():
        folder.mkdir(parents=True, exist_ok=True)
    logger.setLevel(logging.INFO)
    handler = logging.FileHandler(folders["logs_salida"] / f"{run_id}.log", encoding="utf-8")
    handler.setFormatter(logging.Formatter("%(asctime)s %(levelname)s %(message)s"))
    # - Guardar contexto de usuario/IP/huella de token, pero nunca el secreto.
    logger.addHandler(handler)

    return run_id,folders,handler
def random_forest_options(source):
    """Valida opciones del endpoint Random Forest desde JSON o formulario."""
    def parse_int(key, default):
        # Los parámetros multipart son texto, así que intentamos convertirlos
        # a enteros aquí en vez de esperar a que el estimador falle más tarde.
        raw = source.get(key, default)
        try:
            parsed = int(raw)
            if isinstance(raw, float) and raw != parsed:
                raise ValueError
            return parsed
        except (TypeError, ValueError):
            raise ValueError(f"{key} debe ser un entero válido.") from None
    n_estimators = parse_int("n_estimators", 200)
    random_state = parse_int("random_state", 42)
    raw_depth = source.get("max_depth", None)
    max_depth = None if raw_depth in (None, "", "null") else parse_int("max_depth", None)
    try:
        test_size = float(source.get("test_size", 0.2))
    except (TypeError, ValueError):
        raise ValueError("test_size debe ser un número válido.") from None
    target = source.get("target", "y")
    if not 10 <= n_estimators <= 1000:
        raise ValueError("n_estimators debe estar entre 10 y 1000.")
    if max_depth is not None and not 1 <= max_depth <= 100:
        raise ValueError("max_depth debe ser null o estar entre 1 y 100.")
    if not 0.1 <= test_size <= 0.4:
        raise ValueError("test_size debe estar entre 0.1 y 0.4.")
    if random_state < 0:
        raise ValueError("random_state debe ser un entero no negativo.")
    if not isinstance(target, str) or not target.strip():
        raise ValueError("target debe indicar el nombre de una columna.")
    return {"n_estimators": n_estimators, "max_depth": max_depth, "test_size": test_size,
            "random_state": random_state, "target": target, **parse_search_options(source)}


def regression_model_options(source, model_name):
    """Valida las opciones de SVR, AdaBoost y Gradient Boosting.

    `source` puede ser un formulario multipart de Flask. El diccionario
    devuelto contiene solo valores ya convertidos y dentro de límites; así, los
    constructores de modelos no tienen que interpretar texto recibido por HTTP.
    """
    target = source.get("target", "y")
    if not isinstance(target, str) or not target.strip():
        raise ValueError("target debe indicar el nombre de una columna.")
    try:
        test_size = float(source.get("test_size", 0.2))
    except (TypeError, ValueError):
        raise ValueError("test_size debe ser un número entre 0.1 y 0.4.") from None
    if not np.isfinite(test_size) or not 0.1 <= test_size <= 0.4:
        raise ValueError("test_size debe ser un número entre 0.1 y 0.4.")
    random_state = parse_integer("random_state", 42, 0, 2**32 - 1, source)
    options = {"target": target.strip(), "test_size": test_size, "random_state": random_state,
               **parse_search_options(source)}

    def parse_float(name, default, minimum, maximum):
        """Convierte un campo decimal y rechaza booleanos, NaN e infinitos."""
        raw = source.get(name, default)
        if isinstance(raw, bool):
            raise ValueError(f"{name} debe ser un número entre {minimum} y {maximum}.")
        try:
            value = float(raw)
        except (TypeError, ValueError):
            raise ValueError(f"{name} debe ser un número entre {minimum} y {maximum}.") from None
        if not np.isfinite(value) or not minimum <= value <= maximum:
            raise ValueError(f"{name} debe ser un número entre {minimum} y {maximum}.")
        return value

    if model_name == "svm":
        kernel = source.get("kernel", "rbf")
        if not isinstance(kernel, str) or kernel not in {"linear", "poly", "rbf", "sigmoid"}:
            raise ValueError("kernel debe ser linear, poly, rbf o sigmoid.")
        gamma_raw = source.get("gamma", "scale")
        if isinstance(gamma_raw, str) and gamma_raw in {"scale", "auto"}:
            gamma = gamma_raw
        else:
            gamma = parse_float("gamma", gamma_raw, 1e-12, 1e6)
        options.update({
            "kernel": kernel,
            "C": parse_float("C", 1.0, 1e-6, 1e6),
            "epsilon": parse_float("epsilon", 0.1, 0.0, 1e6),
            "gamma": gamma,
            "degree": parse_integer("degree", 3, 1, 20, source),
        })
    elif model_name in {"adaboost", "gradient_boosting"}:
        options.update({
            "n_estimators": parse_integer("n_estimators", 100, 10, 1000, source),
            "max_depth": parse_integer("max_depth", 2, 1, 30, source),
            "learning_rate": parse_float("learning_rate", 0.05, 1e-4, 2.0),
        })
        if model_name == "adaboost":
            loss = source.get("loss", "linear")
            if not isinstance(loss, str) or loss not in {"linear", "square", "exponential"}:
                raise ValueError("loss debe ser linear, square o exponential para AdaBoost.")
            options["loss"] = loss
        else:
            loss = source.get("loss", "squared_error")
            if not isinstance(loss, str) or loss not in {"squared_error", "absolute_error", "huber", "quantile"}:
                raise ValueError("loss no es válido para Gradient Boosting.")
            options["loss"] = loss
            options["subsample"] = parse_float("subsample", 1.0, 1e-3, 1.0)
    else:
        raise ValueError("Modelo de regresión no reconocido.")
    return options


def run_tabular_regressor(app, output_root, reference, raw_csv, options, model_name):
    """Ejecuta SVR o boosting sobre un CSV y registra todos sus resultados.

    El flujo didáctico es: conservar el archivo original, leer la tabla,
    separar objetivo e identificador, crear el preprocesamiento, dividir en
    entrenamiento/prueba, ajustar el pipeline solo con entrenamiento, medir en
    prueba y guardar predicciones, gráfica, modelo y metadatos.
    """
    from io import BytesIO
    import pickle
    import matplotlib.pyplot as plt
    import pandas as pd
    from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
    from sklearn.model_selection import KFold, RandomizedSearchCV, train_test_split

    # Cada nombre público selecciona un constructor distinto. Los tres
    # constructores reciben el mismo preprocesador, pero añaden otro estimador.
    builders = {
            "svm": build_svm_regression_pipeline,
        "adaboost": build_adaboost_regression_pipeline,
        "gradient_boosting": build_gradient_boosting_regression_pipeline,
    }
    labels = {"svm": "SVR", "adaboost": "AdaBoostRegressor",
              "gradient_boosting": "GradientBoostingRegressor"}
    if model_name not in builders:
        return jsonify(error="Modelo de regresión no reconocido."), 400

    # Generar el identificador antes de crear el logger permite que cada
    # entrenamiento escriba en un log separado, incluso si llegan a la vez.
    run_id = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ") + "_" + uuid4().hex[:8]
    logger = logging.getLogger(f"{model_name}.{reference}.{run_id}")
    run_id, folders, handler = preparar_carpetas_y_logs(
        logger, output_root, reference, run_id=run_id
    )

    try:
        logger.info("Petición algoritmo=%s username=%s client_ip=%s token_fingerprint=%s",
                    model_name, getattr(g, "authenticated_user", None) or "anonymous",
                    getattr(g, "client_ip", None) or "unknown",
                    getattr(g, "token_fingerprint", None) or "none")
        # Guardamos los bytes originales antes de analizarlos: así se puede
        # repetir la ejecución con exactamente la misma entrada.
        input_path = folders["csv_entrada"] / f"{run_id}.csv"
        input_path.write_bytes(raw_csv)
        try:
            frame = pd.read_csv(BytesIO(raw_csv), encoding="utf-8-sig")
        except (UnicodeDecodeError, pd.errors.ParserError, pd.errors.EmptyDataError) as exc:
            raise ValueError(f"CSV no válido: {exc}") from None
        if frame.empty or len(frame) > MAX_POINTS:
            raise ValueError(f"El CSV debe tener entre 1 y {MAX_POINTS} filas de datos.")
        if frame.columns.duplicated().any():
            raise ValueError("El CSV no puede contener nombres de columna duplicados.")
        target = options["target"]
        if target not in frame.columns:
            raise ValueError(f"No existe la columna objetivo '{target}'.")

        # y es el valor que se quiere predecir. No numéricos y filas sin y no
        # sirven para una regresión y se descartan antes de separar X e y.
        data = frame.copy()
        data[target] = pd.to_numeric(data[target], errors="coerce")
        valid_target = data[target].notna() & np.isfinite(data[target].to_numpy(dtype=float))
        data = data.loc[valid_target].reset_index(drop=True)
        if len(data) < 10:
            raise ValueError("Se requieren al menos 10 filas con objetivo numérico.")
        id_column = "expediente_id" if "expediente_id" in data.columns else None
        feature_columns = [column for column in data.columns if column not in {target, id_column}]
        if not feature_columns:
            raise ValueError("El CSV debe aportar al menos una columna predictora.")

        # Convertimos columnas de texto que en realidad contienen números;
        # las columnas que siguen siendo texto se tratarán como categorías.
        X = data[feature_columns].copy()
        for column in feature_columns:
            if X[column].dtype == object:
                converted = pd.to_numeric(X[column], errors="coerce")
                non_empty = X[column].notna() & (X[column].astype(str).str.strip() != "")
                if non_empty.any() and converted[non_empty].notna().all():
                    X[column] = converted
        numeric_columns = X.select_dtypes(include=["number", "bool"]).columns.tolist()
        categorical_columns = [column for column in feature_columns if column not in numeric_columns]
        for column in numeric_columns:
            X[column] = X[column].replace([np.inf, -np.inf], np.nan)
        # Columnas completamente vacías no aportan información y pueden impedir
        # que algunos imputadores aprendan una transformación válida.
        usable_columns = [column for column in feature_columns if X[column].notna().any()]
        X = X[usable_columns]
        numeric_columns = [column for column in numeric_columns if column in usable_columns]
        categorical_columns = [column for column in categorical_columns if column in usable_columns]
        if not usable_columns:
            raise ValueError("No quedan características con valores utilizables.")

        # El preprocesador queda dentro del Pipeline. fit() lo ajusta con train
        # solamente y predict() reutiliza esas mismas transformaciones en test.
        preprocessor = build_tabular_preprocessor(
            numeric_columns, categorical_columns, scale_numeric=(model_name == "svm")
        )
        estimator = builders[model_name](preprocessor, options)
        y = data[target].astype(float)
        train_idx, test_idx = train_test_split(
            np.arange(len(data)), test_size=options["test_size"],
            random_state=options["random_state"],
        )
        # Se reservan filas que el entrenamiento no verá. Las métricas de abajo
        # describen esas predicciones de prueba, no el ajuste sobre train.
        search_result = None
        if options.get("search"):
            search_result = RandomizedSearchCV(
                estimator=estimator,
                param_distributions=PARAMETER_CATALOG.get_param_distributions_iterator(model_name),
                n_iter=options["n_iter"],
                cv=KFold(n_splits=min(options["search_cv"], len(train_idx)), shuffle=True,
                         random_state=options["random_state"]),
                scoring="neg_mean_squared_error",
                n_jobs=1,
                random_state=options["random_state"],
                refit=True,
                error_score="raise",
            )
            search_result.fit(X.iloc[train_idx], y.iloc[train_idx])
            estimator = search_result.best_estimator_
        else:
            estimator.fit(X.iloc[train_idx], y.iloc[train_idx])
        predicted = estimator.predict(X.iloc[test_idx])
        observed = y.iloc[test_idx].to_numpy(dtype=float)
        residuals = observed - predicted
        # R² mide variación explicada; RMSE da más peso a errores grandes;
        # MAE resume el error absoluto y bias indica su dirección promedio.
        metrics = {
            "r2": float(r2_score(observed, predicted)),
            "rmse": float(np.sqrt(mean_squared_error(observed, predicted))),
            "mae": float(mean_absolute_error(observed, predicted)),
            "bias": float(np.mean(residuals)),
        }

        # Una fila de salida por observación de prueba permite revisar cada
        # error, además de consultar las métricas agregadas.
        prediction_frame = pd.DataFrame({
            "y_observado": observed, "y_predicho": predicted, "residuo": residuals,
        })
        if id_column:
            prediction_frame.insert(0, id_column, data[id_column].iloc[test_idx].to_numpy())
        csv_path = folders["csv_salida"] / f"{run_id}_predicciones.csv"
        prediction_frame.to_csv(csv_path, index=False, encoding="utf-8")
        graph_path = folders["graficas"] / f"{run_id}_{model_name}.png"
        limits = [float(min(observed.min(), predicted.min())),
                  float(max(observed.max(), predicted.max()))]
        fig, ax = plt.subplots(figsize=(8, 5.5))
        ax.scatter(observed, predicted, alpha=0.7, color="#1f77b4")
        ax.plot(limits, limits, "--", color="#d62728", label="Predicción ideal")
        ax.set(title=f"{labels[model_name]}: observado frente a predicho",
               xlabel="Valor observado", ylabel="Valor predicho")
        ax.grid(True, alpha=0.25)
        ax.legend()
        fig.tight_layout()
        fig.savefig(graph_path, dpi=160)
        plt.close(fig)
        model_path = folders["modelos"] / f"{run_id}_{model_name}.pkt"
        with model_path.open("wb") as stream:
            pickle.dump(estimator, stream, protocol=pickle.HIGHEST_PROTOCOL)

        metadata = {
            "reference": reference, "run_id": run_id, "algorithm": labels[model_name],
            "target": target, "feature_columns": usable_columns,
            "parameters": (search_result.best_params_ if search_result is not None else
                           {key: value for key, value in options.items() if key not in {"target", "search", "n_iter", "search_cv"}}),
            "requested_parameters": {key: value for key, value in options.items() if key not in {"target", "search", "n_iter", "search_cv"}},
            "search": {
                "enabled": search_result is not None,
                "n_iter": options["n_iter"],
                "cv": min(options["search_cv"], len(train_idx)) if search_result is not None else None,
                "best_parameters": search_result.best_params_ if search_result is not None else None,
                "best_cv_score_neg_mse": float(search_result.best_score_) if search_result is not None else None,
            },
            "n_rows_input": int(len(frame)), "n_rows_train": int(len(train_idx)),
            "n_rows_test": int(len(test_idx)), "metrics": metrics,
            "artifacts": {
                "input_csv": str(input_path.relative_to(output_root)),
                "output_csv": str(csv_path.relative_to(output_root)),
                "graph_png": str(graph_path.relative_to(output_root)),
                "trained_model_pkt": str(model_path.relative_to(output_root)),
                "log": str((folders["logs_salida"] / f"{run_id}.log").relative_to(output_root)),
            },
        }
        # Los árboles de boosting exponen importancia por reducción de
        # impureza. SVR no tiene una medida general equivalente para todos sus
        # kernels, por lo que no añadimos ese campo a su respuesta.
        if model_name in {"adaboost", "gradient_boosting"}:
            fitted_model = estimator.named_steps[model_name if model_name == "adaboost" else "gradient_boosting"]
            names = estimator.named_steps["preprocess"].get_feature_names_out()
            scores = fitted_model.feature_importances_
            metadata["feature_importances_top_15"] = [
                {"feature": str(name), "importance": float(score)}
                for name, score in sorted(zip(names, scores), key=lambda row: row[1], reverse=True)[:15]
            ]
        logger.info("Entrenamiento %s completado: n=%d search=%s best_params=%s metrics=%s",
                    labels[model_name], len(data), search_result is not None,
                    search_result.best_params_ if search_result is not None else None, metrics)
        return jsonify(metadata), 200
    except (ValueError, UnicodeDecodeError) as exc:
        logger.exception("No se pudo entrenar %s: %s", model_name, exc)
        return jsonify(error=f"No se pudo entrenar el modelo: {exc}",
                       reference=reference, run_id=run_id), 422
    except Exception:
        logger.exception("Error interno entrenando %s", model_name)
        return jsonify(error="Error interno durante el entrenamiento.",
                       reference=reference, run_id=run_id), 500
    finally:
        handler.close()
        logger.removeHandler(handler)


def run_primitiva_analysis(output_root, reference, raw_csv, version):
    """Ejecuta de forma aislada uno de los analizadores locales de La Primitiva.

    Solo se pueden seleccionar los dos scripts incluidos en el proyecto; el
    fichero recibido por HTTP se trata como datos CSV y nunca como código.
    """
    script_path = PRIMITIVA_SCRIPTS.get(version)
    if script_path is None:
        return jsonify(error="version debe ser 'v01' o 'v02'."), 400
    if not script_path.is_file():
        return jsonify(error=f"No se encuentra el script configurado para {version}."), 500

    run_id = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ") + "_" + uuid4().hex[:8]
    output_root = Path(output_root)
    base = output_root / reference / "primitiva"
    input_dir = base / "csv_entrada"
    report_dir = base / "informes"
    log_dir = base / "logs_salida"
    for directory in (input_dir, report_dir, log_dir):
        directory.mkdir(parents=True, exist_ok=True)

    saved_input = input_dir / f"{run_id}.csv"
    saved_report = report_dir / f"{run_id}_informe.pdf"
    saved_log = log_dir / f"{run_id}.log"
    saved_input.write_bytes(raw_csv)

    try:
        # El código legado busca cualquier CSV del directorio actual y escribe
        # en generated/. Una carpeta temporal por petición evita colisiones.
        with tempfile.TemporaryDirectory(prefix=f"primitiva-{run_id}-") as temp_dir:
            work_dir = Path(temp_dir)
            shutil.copyfile(saved_input, work_dir / "historico.csv")
            result = subprocess.run(
                [sys.executable, str(script_path)],
                cwd=work_dir,
                capture_output=True,
                text=True,
                encoding="utf-8",
                errors="replace",
                timeout=90,
                check=False,
            )
            log_text = (
                f"script={script_path.name}\n"
                f"return_code={result.returncode}\n\n"
                f"--- stdout ---\n{result.stdout}\n"
                f"--- stderr ---\n{result.stderr}"
            )
            saved_log.write_text(log_text, encoding="utf-8")

            generated_report = work_dir / "generated" / "analisis_combinaciones_primitiva.pdf"
            if result.returncode != 0:
                return jsonify(
                    error="El analizador de La Primitiva terminó con error.",
                    reference=reference,
                    run_id=run_id,
                    artifacts={"input_csv": str(saved_input.relative_to(output_root)),
                               "execution_log": str(saved_log.relative_to(output_root))},
                ), 500
            if not generated_report.is_file():
                return jsonify(
                    error="El CSV no contiene los datos válidos necesarios (columnas N1 a N6).",
                    reference=reference,
                    run_id=run_id,
                    artifacts={"input_csv": str(saved_input.relative_to(output_root)),
                               "execution_log": str(saved_log.relative_to(output_root))},
                ), 422
            shutil.copyfile(generated_report, saved_report)

        return jsonify(
            status="completed",
            reference=reference,
            run_id=run_id,
            algorithm=f"primitiva_{version}",
            note=("El análisis de frecuencias históricas no predice sorteos futuros; "
                  "cada combinación conserva la misma probabilidad teórica."),
            artifacts={
                "input_csv": str(saved_input.relative_to(output_root)),
                "report_pdf": str(saved_report.relative_to(output_root)),
                "execution_log": str(saved_log.relative_to(output_root)),
            },
        ), 200
    except subprocess.TimeoutExpired:
        saved_log.write_text("La ejecución superó el límite de 90 segundos.", encoding="utf-8")
        return jsonify(error="El análisis superó el límite de 90 segundos.", reference=reference,
                       run_id=run_id), 504
    except Exception:
        logging.getLogger(__name__).exception("Error al ejecutar el análisis de La Primitiva.")
        saved_log.write_text("Error interno al ejecutar el analizador.", encoding="utf-8")
        return jsonify(error="Error interno al ejecutar el analizador.", reference=reference,
                       run_id=run_id), 500

def run_random_forest(app, output_root, reference, raw_csv, options, source_name):
    """Coordina el entrenamiento de Random Forest y la escritura de artefactos."""
    from io import BytesIO
    import pandas as pd

    run_id = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ") + "_" + uuid4().hex[:8]
    base = Path(output_root) / reference
    folders = {name: base / name for name in ("graficas", "csv_entrada", "logs_salida", "csv_salida", "modelos")}
    for folder in folders.values():
        folder.mkdir(parents=True, exist_ok=True)
    logger = logging.getLogger(f"random_forest.{reference}.{run_id}")
    logger.setLevel(logging.INFO)
    handler = logging.FileHandler(folders["logs_salida"] / f"{run_id}.log", encoding="utf-8")
    handler.setFormatter(logging.Formatter("%(asctime)s %(levelname)s %(message)s"))
    logger.addHandler(handler)
    try:
        # Vincula cada entrenamiento a la identidad auditada por la capa Flask.
        logger.info("Petición algoritmo=random_forest username=%s client_ip=%s token_fingerprint=%s source=%s",
                    getattr(g, "authenticated_user", None) or "anonymous",
                    getattr(g, "client_ip", None) or "unknown",
                    getattr(g, "token_fingerprint", None) or "none", source_name)
        # Guardamos los bytes originales antes de entrenar para conservar una
        # copia exacta de la entrada que produjo este resultado.
        input_path = folders["csv_entrada"] / f"{run_id}.csv"
        input_path.write_bytes(raw_csv)
        try:
            frame = pd.read_csv(BytesIO(raw_csv), encoding="utf-8-sig")
        except (UnicodeDecodeError, pd.errors.ParserError, pd.errors.EmptyDataError) as exc:
            raise ValueError(f"CSV no válido: {exc}") from None
        if len(frame) > MAX_POINTS:
            raise ValueError(f"El CSV supera {MAX_POINTS} filas de datos.")
        if frame.columns.duplicated().any():
            raise ValueError("El CSV no puede contener nombres de columna duplicados.")
        # La clase de regresión prepara las columnas, divide los datos y entrena
        # el pipeline; aquí la función de API coordina la ejecución completa.
        search_options = None
        if options.get("search"):
            search_options = {
                "param_distributions": PARAMETER_CATALOG.get_param_distributions_iterator("random_forest"),
                "n_iter": options["n_iter"],
                "cv": min(options["search_cv"], max(2, int((len(frame) * (1 - options["test_size"])) // 2))),
            }
        model = RandomForestRegressorModel(n_estimators=options["n_estimators"],
                                           max_depth=options["max_depth"],
                                           test_size=options["test_size"],
                                           random_state=options["random_state"],
                                           search_options=search_options)
        result = model.fit(frame, target=options["target"])
        # Escribimos los tres resultados del modelo antes de construir la respuesta.
        csv_path = folders["csv_salida"] / f"{run_id}_predicciones.csv"
        graph_path = folders["graficas"] / f"{run_id}_random_forest.png"
        model_path = folders["modelos"] / f"{run_id}_random_forest.pkt"
        model.save(model_path, result)
        model.write_predictions(csv_path, result)
        model.write_plot(graph_path, result)
        metadata = {
            "reference": reference, "run_id": run_id,
            "algorithm": "RandomForestRegressor", "target": result.target,
            "feature_columns": result.feature_columns,
            "parameters": (result.best_parameters if result.best_parameters is not None else
                           {key: options[key] for key in ("n_estimators", "max_depth", "test_size", "random_state")}),
            "requested_parameters": {key: options[key] for key in ("n_estimators", "max_depth", "test_size", "random_state")},
            "search": {"enabled": bool(options.get("search")), "n_iter": options["n_iter"],
                       "cv": search_options["cv"] if search_options else None,
                       "best_parameters": result.best_parameters,
                       "best_cv_score_neg_mse": result.best_cv_score},
            "n_rows_input": int(len(frame)), "n_rows_train": int(len(frame) - len(result.predictions)),
            "n_rows_test": int(len(result.predictions)), "metrics": result.metrics,
            "feature_importances_top_15": result.feature_importances[:15],
            "artifacts": {
                "input_csv": str(input_path.relative_to(output_root)),
                "output_csv": str(csv_path.relative_to(output_root)),
                "graph_png": str(graph_path.relative_to(output_root)),
                "trained_model_pkt": str(model_path.relative_to(output_root)),
                "log": str((folders["logs_salida"] / f"{run_id}.log").relative_to(output_root)),
            },
        }
        logger.info("Entrenamiento Random Forest completado: n=%d search=%s best_params=%s metrics=%s",
                    len(frame), bool(options.get("search")), result.best_parameters, result.metrics)
        return jsonify(metadata), 200
    except (ValueError, UnicodeDecodeError) as exc:
        logger.exception("No se pudo entrenar el Random Forest: %s", exc)
        return jsonify(error=f"No se pudo entrenar el modelo: {exc}", reference=reference, run_id=run_id), 422
    except Exception as exc:
        logger.exception("Error interno entrenando Random Forest: %s", exc)
        return jsonify(error="Error interno durante el entrenamiento.", reference=reference, run_id=run_id), 500
    finally:
        handler.close()
        logger.removeHandler(handler)

def run_eda_preprocesado(app, output_root, reference, raw_csv, source_name,
                         array_features_excluidas=None, min_correlation=0.0,
                         max_correlation=1.0):
    """Genera el informe EDA y después entrena SVR sobre esos datos procesados."""
    from io import BytesIO
    import pandas as pd

    run_id = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ") + "_" + uuid4().hex[:8]
    base = Path(output_root) / reference
    folders = {name: base / name for name in
               ("informes", "csv_entrada", "csv_salida", "modelos", "logs_salida")}
    for folder in folders.values():
        folder.mkdir(parents=True, exist_ok=True)
    logger = logging.getLogger(f"eda_preprocesado.{reference}.{run_id}")
    logger.setLevel(logging.INFO)
    handler = logging.FileHandler(folders["logs_salida"] / f"{run_id}.log", encoding="utf-8")
    handler.setFormatter(logging.Formatter("%(asctime)s %(levelname)s %(message)s"))
    logger.addHandler(handler)
    try:
        logger.info("Petición algoritmo=eda_preprocesado username=%s client_ip=%s token_fingerprint=%s source=%s",
                    getattr(g, "authenticated_user", None) or "anonymous",
                    getattr(g, "client_ip", None) or "unknown",
                    getattr(g, "token_fingerprint", None) or "none", source_name)
        input_path = folders["csv_entrada"] / f"{run_id}.csv"
        input_path.write_bytes(raw_csv)
        try:
            frame = pd.read_csv(BytesIO(raw_csv), encoding="utf-8-sig")
        except (UnicodeDecodeError, pd.errors.ParserError, pd.errors.EmptyDataError) as exc:
            raise ValueError(f"CSV no válido: {exc}") from None
        if len(frame) > MAX_POINTS:
            raise ValueError(f"El CSV supera {MAX_POINTS} filas de datos.")
        if frame.columns.duplicated().any():
            raise ValueError("El CSV no puede contener nombres de columna duplicados.")
        if not len(frame.columns):
            raise ValueError("El CSV debe incluir columnas con encabezado.")

        # La función compartida produce la matriz de correlación, elimina las
        # variables fuera del intervalo y conserva la exclusión manual existente.
        result = train_one_dataset(
            frame, source_name, base, run_id,
            min_correlation=min_correlation,
            max_correlation=max_correlation,
            excluded_features=array_features_excluidas or [],
        )

        artifact_paths = {
            name: str(Path(path).relative_to(output_root))
            for name, path in result["artifacts"].items()
        }
        artifact_paths.update({
            "input_csv": str(input_path.relative_to(output_root)),
            "log": str((folders["logs_salida"] / f"{run_id}.log").relative_to(output_root)),
        })
        metadata = {
            "reference": reference, "run_id": run_id,
            "algorithm": "EdaPreprocesado+SVR", "target": "y",
            "n_rows_input": int(len(frame)),
            "n_rows_train": result["filas_train"], "n_rows_test": result["filas_test"],
            "input_columns": [str(column) for column in frame.columns],
            "feature_columns": result["variables"],
            "correlation_filter": result["correlation_filter"],
            "transformations": result["codificaciones"],
            "parameters": result["parametros_svr"],
            "best_training": result["best_training"],
            "metrics": result["metricas"],
            "artifacts": artifact_paths,
        }
        logger.info("Informe EDA generado y SVR entrenado: filas=%d metrics=%s",
                    len(frame), result["metricas"])
        return jsonify(metadata), 200
    except (ValueError, UnicodeDecodeError) as exc:
        logger.exception("No se pudo procesar el CSV: %s", exc)
        return jsonify(error=f"No se pudo procesar el CSV: {exc}",
                       reference=reference, run_id=run_id), 422
    except Exception:
        logger.exception("Error interno durante EDA/preprocesado")
        return jsonify(error="Error interno durante EDA y preprocesado.",
                       reference=reference, run_id=run_id), 500
    finally:
        handler.close()
        logger.removeHandler(handler)

def run_model(app, output_root, reference, degree, x, y, input_kind, raw_csv=None):
    """Ajusta el modelo polinómico básico y crea salidas por ejecución."""
    # El sufijo aleatorio evita sobrescrituras si llegan dos peticiones para la
    # misma referencia dentro del mismo segundo. UTC facilita correlación de logs.
    run_id = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ") + "_" + uuid4().hex[:8]
    base = Path(output_root) / reference
    # Un directorio raíz por referencia y carpetas por tipo de artefacto hacen
    # posible auditar/reprocesar cada caso sin mezclar entrada, modelo y salida.
    folders = {name: base / name for name in ("graficas", "csv_entrada", "logs_salida", "csv_salida", "modelos")}
    for folder in folders.values():
        folder.mkdir(parents=True, exist_ok=True)
    logger = logging.getLogger(f"regression.{reference}.{run_id}")
    logger.setLevel(logging.INFO)
    handler = logging.FileHandler(folders["logs_salida"] / f"{run_id}.log", encoding="utf-8")
    handler.setFormatter(logging.Formatter("%(asctime)s %(levelname)s %(message)s"))
    logger.addHandler(handler)
    try:
        # Cada log de ejecución enlaza el trabajo con usuario, IP y huella de
        # token de la petición autenticada, sin almacenar el Bearer en claro.
        logger.info(
            "Petición username=%s client_ip=%s token_fingerprint=%s",
            getattr(g, "authenticated_user", None) or "anonymous",
            getattr(g, "client_ip", None) or "unknown",
            getattr(g, "token_fingerprint", None) or "none",
        )
        input_path = folders["csv_entrada"] / f"{run_id}.csv"
        if raw_csv is None:
            # Para JSON también se guarda una entrada tabular mínima; así cada run
            # tiene una fuente de datos recuperable independientemente del canal.
            with input_path.open("w", newline="", encoding="utf-8") as stream:
                writer = csv.writer(stream)
                writer.writerow(["x", "y"])
                writer.writerows(zip(x.tolist(), y.tolist()))
        else:
            # Guardar bytes originales preserva columnas extra, orden, formato y
            # encabezados exactamente como los envió el cliente.
            input_path.write_bytes(raw_csv)


        X_train, X_test, y_train, y_test = train_test_split(
            x, y, test_size=0.20, random_state=42
        )
        model = PolynomialRegressor(X_train, X_test, y_train, y_test,degree=degree)
        result = model.fit()
        csv_path = folders["csv_salida"] / f"{run_id}_predicciones.csv"
        graph_path = folders["graficas"] / f"{run_id}_ajuste.png"
        model_path = folders["modelos"] / f"{run_id}_modelo.pkt"
        # Persistir el modelo inmediatamente tras entrenar permite volver a usarlo
        # aunque la API se reinicie; el endpoint devuelve la ruta para trazabilidad.
        model.save(model_path)
        model.write_predictions(csv_path, result)
        model.write_plot(graph_path, X_test, y_test, result)
        metadata = {
            "reference": reference,
            "run_id": run_id,
            "degree": degree,
            "n_observations": int(len(X_test)),
            "coefficients_descending": result.coefficients.tolist(),
            "metrics": result.metrics,
            "artifacts": {
                "input_csv": str(input_path.relative_to(output_root)),
                "output_csv": str(csv_path.relative_to(output_root)),
                "graph_png": str(graph_path.relative_to(output_root)),
                "trained_model_pkt": str(model_path.relative_to(output_root)),
                "log": str((folders["logs_salida"] / f"{run_id}.log").relative_to(output_root)),
            },
        }
        logger.info("Ajuste completado input=%s degree=%d n=%d metrics=%s", input_kind, degree, len(x), result.metrics)
        return jsonify(metadata), 200
    except (ValueError, np.linalg.LinAlgError) as exc:
        # Errores de datos/ajuste se registran con el run_id y reciben 422; el log
        # conserva traceback para diagnóstico sin mezclarlo con la respuesta normal.
        logger.exception("No se pudo ajustar el modelo: %s", exc)
        return jsonify(error=f"No se pudo ajustar el modelo: {exc}", reference=reference, run_id=run_id), 422
    except Exception as exc:  # No devolver rutas, stack traces ni secretos al cliente.
        logger.exception("Error interno en el ajuste: %s", exc)
        return jsonify(error="Error interno durante el ajuste.", reference=reference, run_id=run_id), 500
    finally:
        # Cerrar el handler por solicitud evita descriptores abiertos durante un
        # servicio prolongado y permite leer/archivar el log al terminar la petición.
        handler.close()
        logger.removeHandler(handler)


def run_polynomial_search(output_root, reference, x, y, options, weights=None, raw_csv=None):
    """Separa los datos, busca hiperparámetros y evalúa el mejor modelo en prueba."""
    from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
    from sklearn.model_selection import train_test_split
    import pickle
    import matplotlib.pyplot as plt

    # El ID único agrupa todos los archivos que pertenecen a esta petición.
    run_id = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ") + "_" + uuid4().hex[:8]
    base = Path(output_root) / reference
    folders = {name: base / name for name in ("graficas", "csv_entrada", "logs_salida", "csv_salida", "modelos")}
    for folder in folders.values():
        folder.mkdir(parents=True, exist_ok=True)
    logger = logging.getLogger(f"polynomial_search.{reference}.{run_id}")
    logger.setLevel(logging.INFO)
    handler = logging.FileHandler(folders["logs_salida"] / f"{run_id}.log", encoding="utf-8")
    handler.setFormatter(logging.Formatter("%(asctime)s %(levelname)s %(message)s"))
    logger.addHandler(handler)
    try:
        source_kind = "csv" if raw_csv is not None else "json"
        logger.info("RandomizedSearchCV user=%s client_ip=%s source=%s",
                    getattr(g, "authenticated_user", None) or "anonymous",
                    getattr(g, "client_ip", None) or "unknown", source_kind)
        # Conservamos el archivo subido sin cambios; para JSON creamos un CSV
        # sencillo para que también quede guardada la entrada de la búsqueda.
        input_path = folders["csv_entrada"] / f"{run_id}.csv"
        if raw_csv is not None:
            input_path.write_bytes(raw_csv)
        else:
            with input_path.open("w", newline="", encoding="utf-8") as stream:
                writer = csv.writer(stream)
                writer.writerow(["x", "y"])
                writer.writerows(zip(x.tolist(), y.tolist()))

        # Apartamos un 20% para medir el rendimiento final. RandomizedSearchCV
        # solo ve el 80% restante, así la prueba no influye en la selección.
        indices = np.arange(len(x))
        train_idx, test_idx = train_test_split(
            indices, test_size=0.2, random_state=options["random_state"]
        )
        # El pipeline prueba combinaciones usando validación cruzada dentro del
        # conjunto de entrenamiento y vuelve a ajustar la ganadora (refit=True).
        search = build_polynomial_search(
            n_iter=options["n_iter"], cv=options["cv"], random_state=options["random_state"]
        )
        X_train = x[train_idx].reshape(-1, 1)
        X_test = x[test_idx].reshape(-1, 1)
        fit_kwargs = {}
        if weights is not None:
            # Pipeline necesita el prefijo del paso para enviar los pesos al modelo.
            fit_kwargs["polynomial__sample_weight"] = weights[train_idx]
        search.fit(X_train, y[train_idx], **fit_kwargs)
        # La evaluación se calcula con datos que no participaron en la búsqueda.
        predicted = search.best_estimator_.predict(X_test)
        observed = y[test_idx]
        residuals = observed - predicted
        pearson, pearson_pvalue = pearsonr(observed, predicted)
        metrics = {
            "r2": float(r2_score(observed, predicted)),
            "rmse": float(np.sqrt(mean_squared_error(observed, predicted))),
            "mae": float(mean_absolute_error(observed, predicted)),
            "pearson": float(pearson),
            "bias": float(np.mean(observed - predicted)),
        }
        # La tabla de salida pone cada observación junto a su predicción y error.
        csv_path = folders["csv_salida"] / f"{run_id}_predicciones.csv"
        np.savetxt(csv_path, np.column_stack((x[test_idx], observed, predicted, residuals)),
                   delimiter=",", header="x,y_observado,y_predicho,residuo", comments="", fmt="%.10g")
        graph_path = folders["graficas"] / f"{run_id}_randomized_search.png"
        order = np.argsort(x[test_idx])
        fig, ax = plt.subplots(figsize=(9, 5.5))
        ax.scatter(x[test_idx], observed, label="Observaciones", color="#1f77b4", alpha=0.85)
        ax.plot(x[test_idx][order], predicted[order], label="Mejor ajuste", color="#d62728")
        ax.set(title="Regresión polinómica optimizada", xlabel="x", ylabel="y")
        ax.grid(True, alpha=0.25)
        ax.legend()
        fig.tight_layout()
        fig.savefig(graph_path, dpi=160)
        plt.close(fig)
        # Guardamos el pipeline ganador (incluye el estimador y sus parámetros).
        model_path = folders["modelos"] / f"{run_id}_randomized_search.pkt"
        with model_path.open("wb") as stream:
            pickle.dump(search.best_estimator_, stream, protocol=pickle.HIGHEST_PROTOCOL)
        metadata = {
            "reference": reference,
            "run_id": run_id,
            "algorithm": "PolynomialRegressionRandomizedSearchCV",
            "best_parameters": search.best_params_,
            "best_cv_rmse": float(np.sqrt(-search.best_score_)),
            "metrics": metrics,
            "min_observed":np.min(observed),
            "max_observed":np.max(observed),
            "mean_observed": np.mean(observed),
            "n_rows_input": int(len(x)),
            "n_rows_train": int(len(train_idx)),
            "n_rows_test": int(len(test_idx)),
            "artifacts": {
                "input_csv": str(input_path.relative_to(output_root)),
                "output_csv": str(csv_path.relative_to(output_root)),
                "graph_png": str(graph_path.relative_to(output_root)),
                "trained_model_pkt": str(model_path.relative_to(output_root)),
                "log": str((folders["logs_salida"] / f"{run_id}.log").relative_to(output_root)),
            },
        }
        best_estimator_params = search.best_estimator_.named_steps["polynomial"].get_params(deep=False)
        logger.info(
            "RandomizedSearchCV completado param_distributions=%s best_estimator_params=%s best=%s metrics=%s",
            json.dumps(search.param_distributions, sort_keys=True, default=str),
            json.dumps(best_estimator_params, sort_keys=True, default=str),
            search.best_params_, metrics,
        )
        return jsonify(metadata), 200
    except (ValueError, np.linalg.LinAlgError) as exc:
        logger.exception("No se pudo entrenar RandomizedSearchCV: %s", exc)
        return jsonify(error=f"No se pudo entrenar el modelo: {exc}", reference=reference, run_id=run_id), 422
    except Exception:
        logger.exception("Error interno en RandomizedSearchCV")
        return jsonify(error="Error interno durante la búsqueda del modelo.", reference=reference, run_id=run_id), 500
    finally:
        handler.close()
        logger.removeHandler(handler)


def run_polynomial_search_with_cv_predict(
    output_root, reference, x, y, options, weights=None, raw_csv=None
):
    """Busca grados polinómicos y conserva métricas y predicciones de cada fold.

    OOF (out-of-fold) significa que cada observación se predice cuando queda
    fuera del pliegue usado para entrenar el modelo. Así no se evalúa una fila
    con un estimador que ya la ha visto durante el ajuste.
    """
    import pickle
    import matplotlib.pyplot as plt
    from sklearn.metrics import (
        mean_absolute_error,
        mean_squared_error,
        r2_score,
    )
    from sklearn.model_selection import KFold, cross_val_predict, train_test_split

    # El ID relaciona el CSV de entrada, el modelo y los resultados de esta petición.
    run_id = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ") + "_" + uuid4().hex[:8]
    base = Path(output_root) / reference
    folders = {
        name: base / name
        for name in ("graficas", "csv_entrada", "logs_salida", "csv_salida", "modelos", "metadata")
    }
    for folder in folders.values():
        folder.mkdir(parents=True, exist_ok=True)

    logger = logging.getLogger(f"polynomial_search_cv_predict.{reference}.{run_id}")
    logger.setLevel(logging.INFO)
    handler = logging.FileHandler(folders["logs_salida"] / f"{run_id}.log", encoding="utf-8")
    handler.setFormatter(logging.Formatter("%(asctime)s %(levelname)s %(message)s"))
    logger.addHandler(handler)
    try:
        source_kind = "csv" if raw_csv is not None else "json"
        logger.info(
            "RandomizedSearchCV con cross_val_predict user=%s client_ip=%s source=%s",
            getattr(g, "authenticated_user", None) or "anonymous",
            getattr(g, "client_ip", None) or "unknown",
            source_kind,
        )

        # Archivamos la entrada original si llegó como CSV; para JSON escribimos
        # los pares x/y a CSV para que también quede una copia reproducible.
        input_path = folders["csv_entrada"] / f"{run_id}.csv"
        if raw_csv is not None:
            input_path.write_bytes(raw_csv)
        else:
            with input_path.open("w", newline="", encoding="utf-8") as stream:
                writer = csv.writer(stream)
                writer.writerow(["x", "y"])
                writer.writerows(zip(x.tolist(), y.tolist()))

        # Separamos la prueba final antes de buscar parámetros para que no influya
        # ni en la búsqueda ni en las predicciones out-of-fold.
        indices = np.arange(len(x))
        train_idx, test_idx = train_test_split(
            indices, test_size=0.2, random_state=options["random_state"]
        )
        X_train = x[train_idx].reshape(-1, 1)
        X_test = x[test_idx].reshape(-1, 1)
        y_train = y[train_idx]
        fit_kwargs = {}
        if weights is not None:
            # Pipeline distribuye estos pesos entre los datos de entrenamiento
            # de cada pliegue durante RandomizedSearchCV y cross_val_predict.
            fit_kwargs["polynomial__sample_weight"] = weights[train_idx]

        # Primero se escogen los mejores hiperparámetros con validación cruzada.
        search = build_polynomial_search(
            n_iter=options["n_iter"],
            cv=options["cv"],
            random_state=options["random_state"],
        )
        search.fit(X_train, y_train, **fit_kwargs)

        # KFold usa el mismo tipo de partición (sin barajar) que cv=un_entero en
        # RandomizedSearchCV. cross_val_predict devuelve una predicción OOF por fila.
        cv_splitter = KFold(n_splits=options["cv"])
        cv_predictions = cross_val_predict(
            search.best_estimator_,
            X_train,
            y_train,
            cv=cv_splitter,
            params=fit_kwargs or None,
            method="predict",
        )

        # Cada diccionario representa un pliegue. Los índices apuntan a las filas
        # originales de la petición y las predicciones corresponden a sus filas
        # de validación, que no participaron en el ajuste de ese pliegue.
        fold_metadata = []
        for fold_number, (fold_train, fold_valid) in enumerate(
            cv_splitter.split(X_train, y_train), start=1
        ):
            y_valid = y_train[fold_valid]
            prediction_valid = cv_predictions[fold_valid]
            residuals_valid = y_valid - prediction_valid
            pearson, pearson_pvalue = pearsonr(y_valid, prediction_valid)
            fold_metadata.append({
                "fold": fold_number,
                "train_indices": train_idx[fold_train].copy(),
                "validation_indices": train_idx[fold_valid].copy(),
                "x_validation": X_train[fold_valid, 0].copy(),
                "y_true": y_valid.copy(),
                "y_predicted_oof": prediction_valid.copy(),
                "residuals": residuals_valid.copy(),
                "best_parameters": dict(search.best_params_),
                "metrics": {
                    "r2": float(r2_score(y_valid, prediction_valid)),
                    "rmse": float(np.sqrt(mean_squared_error(y_valid, predicted))),
                    "mae": float(mean_absolute_error(y_valid, prediction_valid)),
                    "pearson": float(pearson),
                    "bias": float(np.mean(y_valid - prediction_valid)),
                },
            })

        # Object arrays can store dictionaries and their variable-length arrays.
        # Loading a NumPy object file requires allow_pickle=True; only load files
        # created by this trusted application.
        cv_metadata_path = folders["metadata"] / f"{run_id}_cv_metadata.npy"
        metadata_array = np.empty(len(fold_metadata), dtype=object)
        metadata_array[:] = fold_metadata
        np.save(cv_metadata_path, metadata_array, allow_pickle=True)

        # La prueba final se conserva aparte de la validación cruzada.
        predicted = search.best_estimator_.predict(X_test)
        observed = y[test_idx]
        residuals = observed - predicted
        pearson, _ = pearsonr(observed, predicted)
        metrics = {
            "r2": float(r2_score(observed, predicted)),
            "rmse": float(np.sqrt(mean_squared_error(observed, predicted))),
            "mae": float(mean_absolute_error(observed, predicted)),
            "pearson": float(pearson),
            "bias": float(np.mean(residuals)),
        }

        # Guardamos predicciones y gráfica del conjunto de prueba para mantener
        # la misma salida práctica que el método original.
        csv_path = folders["csv_salida"] / f"{run_id}_predicciones.csv"
        np.savetxt(
            csv_path,
            np.column_stack((x[test_idx], observed, predicted, residuals)),
            delimiter=",",
            header="x,y_observado,y_predicho,residuo",
            comments="",
            fmt="%.10g",
        )
        graph_path = folders["graficas"] / f"{run_id}_randomized_search_cv_predict.png"
        order = np.argsort(x[test_idx])
        fig, ax = plt.subplots(figsize=(9, 5.5))
        ax.scatter(x[test_idx], observed, label="Observaciones", color="#1f77b4", alpha=0.85)
        ax.plot(x[test_idx][order], predicted[order], label="Mejor ajuste", color="#d62728")
        ax.set(title="Regresión polinómica con validación cruzada", xlabel="x", ylabel="y")
        ax.grid(True, alpha=0.25)
        ax.legend()
        fig.tight_layout()
        fig.savefig(graph_path, dpi=160)
        plt.close(fig)

        model_path = folders["modelos"] / f"{run_id}_randomized_search.pkt"
        with model_path.open("wb") as stream:
            pickle.dump(search.best_estimator_, stream, protocol=pickle.HIGHEST_PROTOCOL)

        metadata = {
            "reference": reference,
            "run_id": run_id,
            "algorithm": "PolynomialRegressionRandomizedSearchCVWithCrossValPredict",
            "best_parameters": search.best_params_,
            "best_cv_rmse": float(np.sqrt(-search.best_score_)),
            "metrics": metrics,
            "n_rows_input": int(len(x)),
            "n_rows_train": int(len(train_idx)),
            "n_rows_test": int(len(test_idx)),
            "n_cv_folds": int(len(fold_metadata)),
            "artifacts": {
                "input_csv": str(input_path.relative_to(output_root)),
                "output_csv": str(csv_path.relative_to(output_root)),
                "graph_png": str(graph_path.relative_to(output_root)),
                "trained_model_pkt": str(model_path.relative_to(output_root)),
                "cv_metadata_npy": str(cv_metadata_path.relative_to(output_root)),
                "log": str((folders["logs_salida"] / f"{run_id}.log").relative_to(output_root)),
            },
        }
        logger.info(
            "Búsqueda con cross_val_predict completada folds=%d best=%s metrics=%s cv_metadata=%s",
            len(fold_metadata), search.best_params_, metrics, cv_metadata_path,
        )
        return jsonify(metadata), 200
    except (ValueError, np.linalg.LinAlgError) as exc:
        logger.exception("No se pudo completar la búsqueda con cross_val_predict: %s", exc)
        return jsonify(error=f"No se pudo entrenar el modelo: {exc}", reference=reference, run_id=run_id), 422
    except Exception:
        logger.exception("Error interno en búsqueda con cross_val_predict")
        return jsonify(error="Error interno durante la búsqueda del modelo.", reference=reference, run_id=run_id), 500
    finally:
        handler.close()
        logger.removeHandler(handler)


def run_knn_cross_validation(output_root, reference, raw_csv, options, method):
    """Evalúa KNN con K-Fold o Leave-One-Out y guarda predicciones OOF.

    OOF significa que cada predicción se genera cuando esa fila está fuera del
    pliegue de entrenamiento. Después se ajusta un modelo final con todas las
    filas y se guarda para futuras predicciones.
    """
    from io import BytesIO
    import pickle
    import matplotlib.pyplot as plt
    import pandas as pd
    from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
    from sklearn.model_selection import KFold, LeaveOneOut, RandomizedSearchCV, cross_val_predict

    if method not in {"kfold", "leave_one_out"}:
        raise ValueError("method debe ser 'kfold' o 'leave_one_out'.")

    run_id = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ") + "_" + uuid4().hex[:8]
    base = Path(output_root) / reference
    folders = {
        name: base / name
        for name in ("graficas", "csv_entrada", "logs_salida", "csv_salida", "modelos")
    }
    for folder in folders.values():
        folder.mkdir(parents=True, exist_ok=True)

    logger = logging.getLogger(f"knn_{method}.{reference}.{run_id}")
    logger.setLevel(logging.INFO)
    handler = logging.FileHandler(folders["logs_salida"] / f"{run_id}.log", encoding="utf-8")
    handler.setFormatter(logging.Formatter("%(asctime)s %(levelname)s %(message)s"))
    logger.addHandler(handler)
    try:
        # Leemos CSV en memoria para no tener que crear un archivo temporal.
        frame = pd.read_csv(BytesIO(raw_csv), encoding="utf-8-sig")
        if frame.empty or len(frame) > 5_000:
            raise ValueError("El CSV debe tener entre 1 y 5.000 filas de datos.")
        if frame.columns.duplicated().any():
            raise ValueError("El CSV no puede tener nombres de columna repetidos.")
        target = options["target"]
        if target not in frame.columns:
            raise ValueError(f"No existe la columna objetivo '{target}'.")

        feature_columns = [column for column in frame.columns if column != target]
        if not feature_columns:
            raise ValueError("El CSV debe tener al menos una columna predictora además del objetivo.")
        try:
            numeric_frame = frame[feature_columns + [target]].apply(pd.to_numeric, errors="raise")
        except (TypeError, ValueError):
            raise ValueError("KNN requiere que todas las columnas predictoras y el objetivo sean numéricos.") from None
        values = numeric_frame.to_numpy(dtype=float)
        if not np.isfinite(values).all():
            raise ValueError("El CSV no puede contener valores vacíos, NaN o infinitos.")

        X = numeric_frame[feature_columns]
        y = numeric_frame[target].to_numpy(dtype=float)
        if method == "kfold":
            if options["cv"] > len(frame):
                raise ValueError("cv no puede superar el número de filas del CSV.")
            smallest_train_fold = len(frame) - int(np.ceil(len(frame) / options["cv"]))
        else:
            smallest_train_fold = len(frame) - 1
        if not options.get("search") and options["n_neighbors"] > smallest_train_fold:
            raise ValueError("n_neighbors supera el número de filas disponibles en un pliegue de entrenamiento.")

        # Cada ruta elige su propio separador: KFold usa varios grupos y LOO
        # deja una fila fuera en cada turno. cross_val_predict crea predicciones
        # para las filas que no participaron en el ajuste de su turno.
        if method == "kfold":
            splitter = KFold(
                n_splits=options["cv"],
                shuffle=True,
                random_state=options["random_state"],
            )
            method_name = "KFold"
            n_splits = options["cv"]
        else:
            splitter = LeaveOneOut()
            method_name = "LeaveOneOut"
            n_splits = len(frame)

        model = build_knn_pipeline(options["n_neighbors"])
        final_search = None
        if options.get("search"):
            if len(frame) < 8:
                raise ValueError("La búsqueda de hiperparámetros KNN requiere al menos 8 filas.")
            if method == "leave_one_out" and len(frame) > 200:
                raise ValueError("La búsqueda con Leave-One-Out se limita a 200 filas; use KFold para conjuntos mayores.")

            base_distribution = PARAMETER_CATALOG.get_param_distributions_iterator("knn")
            predicted = np.empty(len(y), dtype=float)
            # La selección de k, ponderación y distancia se repite dentro de
            # cada pliegue externo para que la predicción OOF no vea su objetivo.
            for train_part, validation_part in splitter.split(X, y):
                inner_cv = min(options["search_cv"], len(train_part))
                smallest_inner_train = len(train_part) - int(np.ceil(len(train_part) / inner_cv))
                distribution = dict(base_distribution)
                distribution["knn__n_neighbors"] = [
                    value for value in distribution["knn__n_neighbors"]
                    if value <= smallest_inner_train
                ]
                if not distribution["knn__n_neighbors"]:
                    raise ValueError("No hay valores de n_neighbors válidos para los pliegues elegidos.")
                candidate_count = (
                    len(distribution["knn__n_neighbors"])
                    * len(distribution["knn__weights"])
                    * len(distribution["knn__p"])
                )
                fold_search = RandomizedSearchCV(
                    estimator=build_knn_pipeline(),
                    param_distributions=distribution,
                    n_iter=min(options["n_iter"], candidate_count),
                    cv=KFold(n_splits=inner_cv, shuffle=True,
                             random_state=options["random_state"]),
                    scoring="neg_mean_squared_error",
                    n_jobs=1,
                    random_state=options["random_state"],
                    refit=True,
                    error_score="raise",
                )
                fold_search.fit(X.iloc[train_part], y[train_part])
                predicted[validation_part] = fold_search.predict(X.iloc[validation_part])

            final_distribution = dict(base_distribution)
            final_distribution["knn__n_neighbors"] = [
                value for value in final_distribution["knn__n_neighbors"]
                if value <= len(y) - int(np.ceil(len(y) / options["search_cv"]))
            ]
            final_candidate_count = (
                len(final_distribution["knn__n_neighbors"])
                * len(final_distribution["knn__weights"])
                * len(final_distribution["knn__p"])
            )
            final_search = RandomizedSearchCV(
                estimator=build_knn_pipeline(),
                param_distributions=final_distribution,
                n_iter=min(options["n_iter"], final_candidate_count),
                cv=KFold(n_splits=min(options["search_cv"], len(y)), shuffle=True,
                         random_state=options["random_state"]),
                scoring="neg_mean_squared_error",
                n_jobs=1,
                random_state=options["random_state"],
                refit=True,
                error_score="raise",
            )
            final_search.fit(X, y)
            final_model = final_search.best_estimator_
        else:
            predicted = cross_val_predict(model, X, y, cv=splitter, n_jobs=1)
            final_model = build_knn_pipeline(options["n_neighbors"])
            final_model.fit(X, y)
        residuals = y - predicted
        metrics = {
            "r2": float(r2_score(y, predicted)),
            "rmse": float(np.sqrt(mean_squared_error(y, predicted))),
            "mae": float(mean_absolute_error(y, predicted)),
        }

        # Dejamos la entrada exacta que se usó para que sea posible repetir el run.
        input_path = folders["csv_entrada"] / f"{run_id}.csv"
        input_path.write_bytes(raw_csv)
        output_path = folders["csv_salida"] / f"{run_id}_predicciones_cv.csv"
        predictions_frame = frame[feature_columns].copy()
        predictions_frame.insert(0, "y_observado", y)
        predictions_frame["y_predicho_oof"] = predicted
        predictions_frame["residuo"] = residuals
        predictions_frame.to_csv(output_path, index=False, encoding="utf-8")

        # Entrenamos un modelo final con todas las filas; este es el modelo que
        # se guarda para predecir casos futuros después de evaluar la CV.
        model_path = folders["modelos"] / f"{run_id}_knn_{method}.pkt"
        with model_path.open("wb") as stream:
            pickle.dump(final_model, stream, protocol=pickle.HIGHEST_PROTOCOL)

        graph_path = folders["graficas"] / f"{run_id}_knn_{method}.png"
        fig, ax = plt.subplots(figsize=(7, 6))
        ax.scatter(y, predicted, alpha=0.55, color="#1f77b4", s=14)
        limits = [float(min(y.min(), predicted.min())), float(max(y.max(), predicted.max()))]
        ax.plot(limits, limits, "--", color="#d62728", label="Predicción ideal")
        ax.set(title=f"KNN: {method_name}", xlabel="Valor real", ylabel="Predicción out-of-fold")
        ax.grid(True, alpha=0.25)
        ax.legend()
        fig.tight_layout()
        fig.savefig(graph_path, dpi=160)
        plt.close(fig)

        metadata = {
            "reference": reference,
            "run_id": run_id,
            "algorithm": "KNeighborsRegressor",
            "cross_validation": method_name,
            "n_splits": int(n_splits),
            "n_rows": int(len(frame)),
            "target": target,
            "features": feature_columns,
            "parameters": ({"n_neighbors": options["n_neighbors"]} if final_search is None
                           else final_search.best_params_),
            "search": {
                "enabled": final_search is not None,
                "n_iter": options["n_iter"],
                "cv": min(options["search_cv"], len(y)) if final_search is not None else None,
                "best_parameters": final_search.best_params_ if final_search is not None else None,
                "best_cv_score_neg_mse": float(final_search.best_score_) if final_search is not None else None,
            },
            "metrics_oof": metrics,
            "artifacts": {
                "input_csv": str(input_path.relative_to(output_root)),
                "predictions_csv": str(output_path.relative_to(output_root)),
                "graph_png": str(graph_path.relative_to(output_root)),
                "trained_model_pkt": str(model_path.relative_to(output_root)),
                "log": str((folders["logs_salida"] / f"{run_id}.log").relative_to(output_root)),
            },
        }
        logger.info(
            "KNN CV completada method=%s rows=%d splits=%d parameters=%s metrics=%s",
            method_name, len(frame), n_splits, metadata["parameters"], metrics,
        )
        return jsonify(metadata), 200
    except (ValueError, UnicodeDecodeError, pd.errors.ParserError, pd.errors.EmptyDataError) as exc:
        logger.exception("No se pudo ejecutar KNN con %s: %s", method, exc)
        return jsonify(error=f"No se pudo entrenar el modelo: {exc}", reference=reference, run_id=run_id), 422
    except Exception:
        logger.exception("Error interno durante la validación cruzada KNN (%s)", method)
        return jsonify(error="Error interno durante la validación cruzada KNN.", reference=reference, run_id=run_id), 500
    finally:
        handler.close()
        logger.removeHandler(handler)
