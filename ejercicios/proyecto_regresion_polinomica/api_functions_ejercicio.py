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
from regression.tabular_preprocessing import build_tabular_preprocessor
from regression.svm_regressor import build_svm_regression_pipeline
from regression.adaboost_regressor import build_adaboost_regression_pipeline
from regression.gradient_boosting_regressor import build_gradient_boosting_regression_pipeline
from regression.parameter_search import PARAMETER_CATALOG

# Límites del procesamiento aplicados tanto a endpoints JSON como a CSV.
MAX_POINTS = 100_000
REFERENCE_RE = re.compile(r"^[A-Za-z0-9][A-Za-z0-9._-]{0,63}$")





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



def validate_xy(x, y):
    """Rechaza vectores x/y vacíos, desalineados o con NaN/infinito."""
    # NaN e infinito contaminan coeficientes y métricas sin producir siempre una
    # excepción clara; se rechazan en el borde HTTP antes de invocar NumPy.
    if x.ndim != 1 or y.ndim != 1 or len(x) != len(y) or len(x) < 2:
        raise ValueError("Se requieren al menos dos pares x/y del mismo tamaño.")
    if not np.isfinite(x).all() or not np.isfinite(y).all():
        raise ValueError("x e y solo pueden contener números finitos.")
    return x, y

# ============================================================================
# EJERCICIO: completar las funciones comunes de entrada, entrenamiento y salida
# ============================================================================
# Este archivo es una guía de PSEUDOCÓDIGO, no contiene implementaciones Python.
# Cada bloque describe la función que el alumno debe escribir en esta capa.
# Las rutas HTTP pertenecen a app_ejercicio.py; los estimadores pertenecen a
# regression/*_def_ejercicio.py.
#
# 1. VALIDAR REFERENCE
def validate_reference(value):
    # - Recibir una referencia de texto.-> leo value
    # - Aceptar solo letras, dígitos, punto, guion y guion bajo, con longitud acotada.
    # if not isinstance(value, str) or not REFERENCE_RE.fullmatch(value):
    # - Si no cumple, devolver error legible sin construir una ruta de salida.
    # - Si cumple, devolver la referencia segura.
    if not isinstance(value, str) or not REFERENCE_RE.fullmatch(value):
        # - Si no cumple, devolver error legible sin construir una ruta de salida.
        return None, "reference es obligatorio y debe tener 1-64 caracteres: letras, números, punto, guion o guion bajo."
    # - Si cumple, devolver la referencia segura.
    return value, None



#
# 2. PARSEAR PUNTOS JSON Y VALIDAR X/Y
def parse_json_points(points):
    """Convierte puntos JSON en dos vectores numéricos y comprueba sus valores."""
    # Limitar número de filas antes de convertirlas a arrays evita asignaciones
    # grandes no necesarias; después se comprueba tipo y finitud en validate_xy.
    # - Exigir una lista de filas JSON con claves x e y.
    # - Comprobar mínimo y máximo de filas antes de convertir.
    if not isinstance(points, list) or not 2 <= len(points) <= MAX_POINTS:
        raise ValueError(f"data debe contener entre 2 y {MAX_POINTS} pares x/y.")
    try:
        # - Convertir ambas columnas a números de punto flotante.
        x = np.asarray([row["x"] for row in points], dtype=float)
        y = np.asarray([row["y"] for row in points], dtype=float)
    except (TypeError, ValueError, KeyError, IndexError):
        raise ValueError("Cada elemento de data debe tener valores numéricos x e y.") from None
    # - Rechazar claves ausentes, texto no numérico, NaN e infinito.
    # - Comprobar que X e y sean vectores unidimensionales de igual longitud.
    return validate_xy(x, y)


#
# 3. PARSEAR ENTEROS, REALES Y OPCIONES
def parse_integer(name, default, minimum, maximum, source):
    """Lee un entero tanto de JSON como de un formulario HTML/Postman."""
    # Los números JSON llegan como int; los campos de un formulario llegan
    # como texto. Convertimos ambos formatos y después aplicamos el límite.
    # - Leer un valor recibido como número JSON o texto de formulario multipart.
    raw = source.get(name, default)
    try:
        if isinstance(raw, bool) or (isinstance(raw, str) and not raw.strip()):
            # - Rechazar bools cuando se espera entero, decimales no enteros y valores
            raise ValueError
        parsed = int(raw)
        if isinstance(raw, float) and raw != parsed:
            raise ValueError
    except (TypeError, ValueError):
        raise ValueError(f"{name} debe ser un entero entre {minimum} y {maximum}.") from None
    if not minimum <= parsed <= maximum:
        #   fuera del intervalo permitido.
        raise ValueError(f"{name} debe ser un entero entre {minimum} y {maximum}.")
    return parsed
    # - Aplicar valor por defecto cuando el campo no se envíe.-> no se hace
    # para futuro
    # - Para búsqueda, convertir el interruptor a booleano y validar n_iter y cv.
    # - Para Random Forest, validar n_estimators, max_depth, test_size, seed y target.
    # - Para modelos tabulares, validar por separado las opciones propias de SVR,
    #   AdaBoost y Gradient Boosting.
    # - No confiar en campos desconocidos para instanciar una clase o ejecutar código.


#
# 4. LEER CSV POLINÓMICO
def parse_csv(text):
    """Extrae x/y de un CSV para los modelos polinómicos de una variable."""
    # DictReader permite conservar columnas contextuales del caso de terrazas,
    # aunque el algoritmo solo use x e y. Las filas extra no se descartan del
    # CSV archivado; solo se extraen aquí las dos variables del modelo.
    reader = csv.DictReader(io.StringIO(text))
    if not reader.fieldnames or not {"x", "y"}.issubset(set(reader.fieldnames)):
        # - Analizar encabezados y exigir x e y.
        raise ValueError("El encabezado debe contener las columnas x,y.")
    # - Leer cada fila con número de línea para producir errores comprensibles.
    x_values, y_values = [], []
    for line_number, row in enumerate(reader, start=2):
        try:
            # - Convertir x/y a números finitos y respetar el máximo de registros.
            # - Preservar los bytes originales si luego se guardará el archivo recibido.
            x_values.append(float(row["x"]))
            y_values.append(float(row["y"]))
        except (TypeError, ValueError):
            raise ValueError(f"Valores no numéricos en la línea {line_number}.") from None
        if len(x_values) > MAX_POINTS:
            raise ValueError(f"El CSV supera {MAX_POINTS} filas de datos.")
    return validate_xy(np.asarray(x_values), np.asarray(y_values))


# 5. PREPARAR CARPETAS Y LOGS
def preparar_carpetas_y_logs(logger,output_root,reference):
    # - Generar run_id único con fecha UTC y sufijo aleatorio
    run_id = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ") + "_" + uuid4().hex[:8]
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



#
# 6. FLUJO COMÚN DE REGRESIÓN TABULAR (SVR / BOOSTING)
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

    #inicializa el logger
    logger = logging.getLogger(f"{model_name}.{reference}.{run_id}")
    run_id,folders,handler =  preparar_carpetas_y_logs(logger,output_root,reference)

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
            # - Leer CSV y validar encabezados no repetidos, objetivo y límites de filas.
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
        # - Convertir objetivo a número; excluir filas con objetivo ausente/inválido.
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
        # - Separar predictores  X y objetivo Y; excluir columna identificadora si existe.
        X = data[feature_columns].copy()
        # Recorrer todas las columnas para categorizar
        for column in feature_columns:
            if X[column].dtype == object:
                converted = pd.to_numeric(X[column], errors="coerce")
                non_empty = X[column].notna() & (X[column].astype(str).str.strip() != "")
                if non_empty.any() and converted[non_empty].notna().all():
                    X[column] = converted
        #Bloque para columnas numericas
        numeric_columns = X.select_dtypes(include=["number", "bool"]).columns.tolist()
        # Lo que no es numerico es texto
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

# - Convertir objetivo a número; excluir filas con objetivo ausente/inválido.
# - Separar predictores y objetivo; excluir columna identificadora si existe.
# - Detectar columnas numéricas/categóricas y limpiar infinito como dato ausente.
# - Construir preprocesador y pipeline sin ajustar transformaciones sobre test.
# - Separar entrenamiento y prueba de forma reproducible.
# - Si se solicita búsqueda, ejecutar RandomizedSearchCV con CV dentro de train;
#   reutilizar el mejor estimador reentrenado.
# - Si no hay búsqueda, ajustar pipeline únicamente con train.
# - Predecir el conjunto reservado y calcular R², RMSE, MAE y residuo medio.
# - Guardar predicciones, gráfico, pipeline, CSV y metadatos; construir JSON.
# - Convertir errores de entrada a 422 y errores inesperados a 500 genérico.
#
# 7. FLUJO DE RANDOM FOREST
# - Reutilizar el análisis tabular: objetivo, predictores, tipos, faltantes e ID.
# - Construir pipeline: imputación numérica, imputación/codificación categórica,
#   ColumnTransformer y RandomForestRegressor.
# - Reservar test antes de búsqueda para no usarlo al escoger hiperparámetros.
# - Si se activa búsqueda, usar RandomizedSearchCV + KFold sobre train; registrar
#   mejor combinación y puntuación CV.
# - Ajustar o recuperar mejor estimador; predecir test y calcular métricas.
# - Extraer importancia de variables transformadas y ordenarla de mayor a menor.
# - Guardar modelo, predicciones, gráfico de ajuste/importancias, entrada y log.
#
# 8. REGRESIÓN POLINÓMICA BÁSICA
# - Guardar CSV recibido o crear CSV de entrada equivalente para JSON.
# - Separar train/test con semilla fija.
# - Crear PolynomialRegressor con el grado solicitado y opciones rcond/full/w/cov.
# - Ajustar solo con train; calcular predicciones y métricas en test.
# - Persistir coeficientes/modelo, predicciones, gráfico, metadatos y log.
#
# 9. BÚSQUEDA POLINÓMICA RANDOMIZEDSEARCHCV
# - Guardar entrada y separar un holdout que no participa en la búsqueda.
# - Construir pipeline del adaptador PolynomialSearchEstimator.
# - Consultar distribución segura de parámetros; crear RandomizedSearchCV con
#   CV interna reproducible y scoring de error cuadrático negativo.
# - Ajustar search solo con train; recuperar best_estimator_ y best_params_.
# - Evaluar el estimador ganador una vez en holdout.
# - Guardar los parámetros del estimador ganador, coeficientes y artifacts.
#
# 10. BÚSQUEDA POLINÓMICA + CROSS_VAL_PREDICT
# - Ejecutar la selección de hiperparámetros con pliegues internos en train.
# - Para el mejor conjunto de opciones, producir predicciones out-of-fold sobre
#   las filas que se emplearon en la validación cruzada exterior solicitada.
# - En cada pliegue: separar índices de entrenamiento y validación; registrar
#   los índices, los metadatos de la clase, parámetros y resultados del ajuste.
# - Guardar por pliegue un objeto NumPy estructurado con metadatos serializables,
#   sin convertir la predicción resumida en una matriz no trazable.
# - Mantener separada la evaluación final sobre holdout para evitar optimismo.
# - Conservar el array global OOF, archivo por-fold, métricas y best_estimator_.
#
# 11. KNN CON K-FOLD O LEAVE-ONE-OUT
# - Leer tabla numérica, comprobar objetivo, predictores, NaN/infinito y límites.
# - Calcular el tamaño del menor pliegue de entrenamiento y validar n_neighbors.
# - Construir pipeline de escalado seguido por KNeighborsRegressor.
# - Para K-Fold, dividir en k partes con barajado y semilla fija.
# - Para LOO, reservar exactamente una fila en cada iteración y limitar tamaño.
# - Si se activa búsqueda, realizar selección interna solo con entrenamiento del
#   pliegue externo para evitar fuga de información desde su validación.
# - Obtener predicción OOF para cada fila, residuo y métricas agregadas.
# - Ajustar un modelo final con todas las filas y guardar sus artefactos.
#
# 12. ESCRITURA DE ARTEFACTOS Y RESPUESTA
# - Usar nombres formados con run_id, no con nombres arbitrarios del cliente.
# - Guardar CSV original exacto y un CSV fila a fila de y observado/predicho.
# - Guardar gráfico, modelo, log y metadatos con rutas relativas a OUTPUT_ROOT.
# - Construir JSON con reference, run_id, algoritmo, parámetros, métricas,
#   filas, columnas y referencias a los archivos generados.
# - No devolver stack trace, contenido privado del CSV ni secretos.
#
# 13. PRUEBAS QUE DEBERÁN ACOMPAÑAR LA IMPLEMENTACIÓN
# - CSV/JSON correcto, cabecera ausente, filas vacías, tipos inválidos y NaN.
# - reference con caracteres peligrosos o longitud excesiva.
# - Parámetros justo en límites inferior/superior y fuera de ellos.
# - Test de respuesta, artefactos y metadatos por cada workflow importante.
# - Caso de búsqueda para comprobar que test no contamina selección de modelo.
