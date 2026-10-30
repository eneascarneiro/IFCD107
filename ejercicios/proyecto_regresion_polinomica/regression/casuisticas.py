"""Entrenamiento común para las casuísticas comparativas de regresión.

Cada algoritmo usa el mismo flujo: separación train/test, preprocesado dentro
de un Pipeline, GridSearchCV y RandomizedSearchCV, comparación por RMSE y
evaluación final sobre el mismo conjunto de prueba.
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import loguniform
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import GridSearchCV, RandomizedSearchCV, train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, PolynomialFeatures, StandardScaler

from regression.eda_preprocesado import EdaPreprocesado


ALGORITHM_LABELS = {
    "linear_simple": "Regresión Lineal Simple",
    "linear_multiple": "Regresión Lineal Múltiple",
    "ridge": "Regresión Ridge (L2)",
    "lasso": "Regresión Lasso (L1)",
    "elastic_net": "Elastic Net",
    "bayesian_ridge": "Regresión Lineal Bayesiana",
    "polynomial": "Regresión Polinomial",
    "huber": "Regresión de Huber",
    "ransac": "Regresión RANSAC",
    "theil_sen": "Regresión Theil-Sen",
    "quantile": "Regresión Cuantílica",
    "decision_tree": "Árbol de Decisión",
    "random_forest": "Random Forest Regressor",
    "gradient_boosting": "Gradient Boosting Regressor",
    "xgboost": "XGBoost Regressor",
    "lightgbm": "LightGBM Regressor",
    "catboost": "CatBoost Regressor",
    "svr": "Support Vector Regression",
    "knn": "KNN Regressor",
    "gaussian_process": "Gaussian Process Regression",
    "mlp": "MLP Regressor",
}

TARGET_COLUMN = "y"
ID_COLUMN = "expediente_id"
TEST_SIZE = 0.2
RANDOM_STATE = 42


def _native_scalar(value):
    """Convierte escalares NumPy a tipos que Flask y JSON pueden serializar."""
    return value.item() if isinstance(value, np.generic) else value


def _make_estimator(algorithm: str):
    """Importa y crea el estimador seleccionado sin cargar librerías opcionales al arrancar la API."""
    if algorithm in {"linear_simple", "linear_multiple", "polynomial"}:
        from sklearn.linear_model import LinearRegression
        return LinearRegression()
    if algorithm == "ridge":
        from sklearn.linear_model import Ridge
        return Ridge()
    if algorithm == "lasso":
        from sklearn.linear_model import Lasso
        return Lasso(max_iter=10_000, random_state=RANDOM_STATE)
    if algorithm == "elastic_net":
        from sklearn.linear_model import ElasticNet
        return ElasticNet(max_iter=10_000, random_state=RANDOM_STATE)
    if algorithm == "bayesian_ridge":
        from sklearn.linear_model import BayesianRidge
        return BayesianRidge()
    if algorithm == "huber":
        from sklearn.linear_model import HuberRegressor
        return HuberRegressor(max_iter=1_000)
    if algorithm == "ransac":
        from sklearn.linear_model import RANSACRegressor
        from sklearn.linear_model import LinearRegression
        return RANSACRegressor(estimator=LinearRegression(), random_state=RANDOM_STATE)
    if algorithm == "theil_sen":
        from sklearn.linear_model import TheilSenRegressor
        return TheilSenRegressor(random_state=RANDOM_STATE, max_subpopulation=500)
    if algorithm == "quantile":
        from sklearn.linear_model import QuantileRegressor
        return QuantileRegressor(solver="highs")
    if algorithm == "decision_tree":
        from sklearn.tree import DecisionTreeRegressor
        return DecisionTreeRegressor(random_state=RANDOM_STATE)
    if algorithm == "random_forest":
        from sklearn.ensemble import RandomForestRegressor
        return RandomForestRegressor(random_state=RANDOM_STATE, n_jobs=1)
    if algorithm == "gradient_boosting":
        from sklearn.ensemble import GradientBoostingRegressor
        return GradientBoostingRegressor(random_state=RANDOM_STATE)
    if algorithm == "xgboost":
        from xgboost import XGBRegressor
        return XGBRegressor(objective="reg:squarederror", random_state=RANDOM_STATE,
                            n_jobs=1, verbosity=0)
    if algorithm == "lightgbm":
        from lightgbm import LGBMRegressor
        return LGBMRegressor(random_state=RANDOM_STATE, n_jobs=1, verbosity=-1)
    if algorithm == "catboost":
        from catboost import CatBoostRegressor
        return CatBoostRegressor(random_seed=RANDOM_STATE, verbose=False,
                                 allow_writing_files=False, thread_count=1)
    if algorithm == "svr":
        from sklearn.svm import SVR
        return SVR()
    if algorithm == "knn":
        from sklearn.neighbors import KNeighborsRegressor
        return KNeighborsRegressor(n_jobs=1)
    if algorithm == "gaussian_process":
        from sklearn.gaussian_process import GaussianProcessRegressor
        from sklearn.gaussian_process.kernels import Matern
        return GaussianProcessRegressor(kernel=Matern(), normalize_y=True,
                                        random_state=RANDOM_STATE, n_restarts_optimizer=0)
    if algorithm == "mlp":
        from sklearn.neural_network import MLPRegressor
        return MLPRegressor(random_state=RANDOM_STATE, max_iter=1_000,
                            early_stopping=False)
    raise ValueError(f"Algoritmo no reconocido: {algorithm}.")


def _search_spaces(algorithm: str) -> tuple[list[dict], list[dict]]:
    """Devuelve cuadrícula discreta y distribuciones para las dos búsquedas."""
    if algorithm in {"linear_simple", "linear_multiple"}:
        return ([{"model__fit_intercept": [True, False]}],
                [{"model__fit_intercept": [True, False]}])
    if algorithm == "ridge":
        return ([{"model__alpha": [0.01, 0.1, 1.0, 10.0, 100.0]}],
                [{"model__alpha": loguniform(1e-4, 1e3)}])
    if algorithm == "lasso":
        return ([{"model__alpha": [0.001, 0.01, 0.1, 1.0, 10.0]}],
                [{"model__alpha": loguniform(1e-4, 1e2)}])
    if algorithm == "elastic_net":
        return ([{"model__alpha": [0.001, 0.01, 0.1, 1.0],
                  "model__l1_ratio": [0.1, 0.5, 0.9]}],
                [{"model__alpha": loguniform(1e-4, 1e2),
                  "model__l1_ratio": loguniform(0.01, 0.99)}])
    if algorithm == "bayesian_ridge":
        return ([{"model__alpha_1": [1e-6, 1e-5, 1e-4],
                  "model__lambda_1": [1e-6, 1e-5, 1e-4]}],
                [{"model__alpha_1": loguniform(1e-7, 1e-3),
                  "model__lambda_1": loguniform(1e-7, 1e-3)}])
    if algorithm == "polynomial":
        return ([{"polynomial__degree": [2, 3, 4]}],
                [{"polynomial__degree": [2, 3, 4, 5]}])
    if algorithm == "huber":
        return ([{"model__epsilon": [1.1, 1.35, 1.5, 1.75],
                  "model__alpha": [0.0001, 0.001, 0.01]}],
                [{"model__epsilon": [1.1, 1.2, 1.35, 1.5, 1.75, 2.0],
                  "model__alpha": loguniform(1e-5, 1e-1)}])
    if algorithm == "ransac":
        return ([{"model__min_samples": [None, 0.5, 0.75],
                  "model__max_trials": [50, 100, 200]}],
                [{"model__min_samples": [None, 0.5, 0.75, 0.9],
                  "model__max_trials": [50, 100, 200, 300]}])
    if algorithm == "theil_sen":
        return ([{"model__max_subpopulation": [100, 250, 500]}],
                [{"model__max_subpopulation": [100, 150, 250, 350, 500, 750]}])
    if algorithm == "quantile":
        return ([{"model__quantile": [0.25, 0.5, 0.75],
                  "model__alpha": [0.0, 0.01, 0.1]}],
                [{"model__quantile": [0.1, 0.25, 0.5, 0.75, 0.9],
                  "model__alpha": loguniform(1e-5, 1.0)}])
    if algorithm == "decision_tree":
        return ([{"model__max_depth": [None, 3, 5, 10],
                  "model__min_samples_leaf": [1, 2, 5]}],
                [{"model__max_depth": [None, 2, 3, 5, 8, 12],
                  "model__min_samples_leaf": [1, 2, 3, 5, 8]}])
    if algorithm == "random_forest":
        return ([{"model__n_estimators": [50, 100],
                  "model__max_depth": [None, 5, 10],
                  "model__min_samples_leaf": [1, 2]}],
                [{"model__n_estimators": [50, 75, 100, 150],
                  "model__max_depth": [None, 3, 5, 8, 12],
                  "model__min_samples_leaf": [1, 2, 4]}])
    if algorithm == "gradient_boosting":
        return ([{"model__n_estimators": [50, 100],
                  "model__learning_rate": [0.03, 0.1],
                  "model__max_depth": [2, 3]}],
                [{"model__n_estimators": [50, 75, 100, 150],
                  "model__learning_rate": loguniform(0.01, 0.3),
                  "model__max_depth": [2, 3, 4]}])
    if algorithm == "xgboost":
        return ([{"model__n_estimators": [50, 100],
                  "model__max_depth": [2, 4],
                  "model__learning_rate": [0.03, 0.1]}],
                [{"model__n_estimators": [50, 75, 100, 150],
                  "model__max_depth": [2, 3, 4, 6],
                  "model__learning_rate": loguniform(0.01, 0.3),
                  "model__subsample": [0.7, 0.85, 1.0]}])
    if algorithm == "lightgbm":
        return ([{"model__n_estimators": [50, 100],
                  "model__num_leaves": [7, 15, 31],
                  "model__learning_rate": [0.03, 0.1]}],
                [{"model__n_estimators": [50, 75, 100, 150],
                  "model__num_leaves": [7, 15, 23, 31],
                  "model__learning_rate": loguniform(0.01, 0.3)}])
    if algorithm == "catboost":
        return ([{"model__iterations": [50, 100], "model__depth": [4, 6],
                  "model__learning_rate": [0.03, 0.1]}],
                [{"model__iterations": [50, 75, 100, 150],
                  "model__depth": [3, 4, 5, 6, 7],
                  "model__learning_rate": loguniform(0.01, 0.3)}])
    if algorithm == "svr":
        return ([{"model__kernel": ["linear"], "model__C": [0.1, 1.0, 10.0],
                  "model__epsilon": [0.01, 0.1, 0.5]},
                 {"model__kernel": ["rbf"], "model__C": [0.1, 1.0, 10.0],
                  "model__epsilon": [0.01, 0.1, 0.5],
                  "model__gamma": ["scale", "auto", 0.01]}],
                [{"model__kernel": ["linear"], "model__C": loguniform(1e-2, 1e3),
                  "model__epsilon": loguniform(1e-3, 1e1)},
                 {"model__kernel": ["rbf"], "model__C": loguniform(1e-2, 1e3),
                  "model__epsilon": loguniform(1e-3, 1e1),
                  "model__gamma": loguniform(1e-4, 1e1)}])
    if algorithm == "knn":
        return ([{"model__n_neighbors": [2, 3, 5, 7],
                  "model__weights": ["uniform", "distance"],
                  "model__p": [1, 2]}],
                [{"model__n_neighbors": [2, 3, 4, 5, 7, 9, 11],
                  "model__weights": ["uniform", "distance"],
                  "model__p": [1, 2]}])
    if algorithm == "gaussian_process":
        return ([{"model__alpha": [1e-8, 1e-5, 1e-3, 0.01]}],
                [{"model__alpha": loguniform(1e-9, 1e-1)}])
    if algorithm == "mlp":
        return ([{"model__hidden_layer_sizes": [(32,), (64,), (32, 16)],
                  "model__activation": ["relu", "tanh"],
                  "model__alpha": [0.0001, 0.001]}],
                [{"model__hidden_layer_sizes": [(16,), (32,), (64,), (32, 16), (64, 32)],
                  "model__activation": ["relu", "tanh"],
                  "model__alpha": loguniform(1e-5, 1e-1)}])
    raise ValueError(f"No hay una búsqueda configurada para {algorithm}.")


def _randomized_candidate_count(distributions: list[dict]) -> int | None:
    """Cuenta candidatos cuando todas las opciones son listas finitas."""
    total = 0
    for parameter_set in distributions:
        combinations = 1
        for values in parameter_set.values():
            if hasattr(values, "rvs"):
                return None
            combinations *= len(values)
        total += combinations
    return total


def _create_pipeline(algorithm: str, frame: pd.DataFrame) -> tuple[Pipeline, list[str]]:
    """Prepara categorías/números en un Pipeline para evitar fuga entre pliegues."""
    if TARGET_COLUMN not in frame:
        raise ValueError(f"El CSV debe incluir la columna objetivo '{TARGET_COLUMN}'.")
    columns = [column for column in frame.columns if column not in {TARGET_COLUMN, ID_COLUMN}]
    if algorithm == "linear_simple":
        columns = columns[:1]
    if not columns:
        raise ValueError("El CSV debe contener al menos una característica predictora además de y.")
    X = frame[columns].copy()
    numeric_columns = X.select_dtypes(include=["number", "bool"]).columns.tolist()
    for column in numeric_columns:
        X[column] = X[column].replace([np.inf, -np.inf], np.nan)
    columns = [column for column in columns if X[column].notna().any()]
    X = X[columns]
    numeric_columns = [column for column in numeric_columns if column in columns]
    categorical_columns = [column for column in columns if column not in numeric_columns]
    transformers = []
    if numeric_columns:
        numeric_pipeline = Pipeline([
            ("imputer", SimpleImputer(strategy="median")),
            ("scale", StandardScaler()),
        ])
        transformers.append(("numeric", numeric_pipeline, numeric_columns))
    if categorical_columns:
        categorical_pipeline = Pipeline([
            ("imputer", SimpleImputer(strategy="most_frequent")),
            ("onehot", OneHotEncoder(handle_unknown="ignore", sparse_output=False)),
        ])
        transformers.append(("categorical", categorical_pipeline, categorical_columns))
    if not transformers:
        raise ValueError("No hay características utilizables después de limpiar valores vacíos.")
    preprocessor = ColumnTransformer(transformers=transformers, remainder="drop",
                                     sparse_threshold=0.0)
    estimator = _make_estimator(algorithm)
    if algorithm == "polynomial":
        pipeline = Pipeline([
            ("preprocess", preprocessor),
            ("polynomial", PolynomialFeatures(include_bias=False)),
            ("model", estimator),
        ])
    else:
        pipeline = Pipeline([("preprocess", preprocessor), ("model", estimator)])
    return pipeline, columns


def train_algorithm_frame(
    frame: pd.DataFrame,
    algorithm: str,
    output_dir: Path,
    run_id: str,
    cv: int = 5,
    n_iter: int = 20,
    random_state: int = RANDOM_STATE,
) -> dict:
    """Ejecuta búsquedas, genera informe y persiste modelo/métricas para un CSV."""
    if algorithm not in ALGORITHM_LABELS:
        raise ValueError(f"Algoritmo no reconocido: {algorithm}.")
    if cv < 2 or n_iter < 1:
        raise ValueError("cv debe ser >= 2 y n_iter debe ser >= 1.")
    frame = frame.copy()
    if len(frame) < 10:
        raise ValueError("Se necesitan al menos 10 filas para una separación train/test fiable.")
    frame.columns = [str(column).strip() for column in frame.columns]
    if frame.columns.duplicated().any():
        raise ValueError("El CSV no puede contener nombres de columna duplicados.")
    processor = EdaPreprocesado()
    prepared, mappings = processor.transform(frame)
    if TARGET_COLUMN not in prepared:
        raise ValueError(f"El CSV debe incluir la columna objetivo '{TARGET_COLUMN}'.")
    prepared[TARGET_COLUMN] = pd.to_numeric(prepared[TARGET_COLUMN], errors="coerce")
    prepared[TARGET_COLUMN] = prepared[TARGET_COLUMN].replace([np.inf, -np.inf], np.nan)
    prepared = prepared.dropna(subset=[TARGET_COLUMN]).reset_index(drop=True)
    if len(prepared) < 10:
        raise ValueError("Se necesitan al menos 10 filas con y numérico.")
    output_dir = Path(output_dir)
    directories = {name: output_dir / name for name in
                   ("informes", "csv_salida", "modelos", "busquedas")}
    for directory in directories.values():
        directory.mkdir(parents=True, exist_ok=True)

    indices = np.arange(len(prepared))
    train_indices, test_indices = train_test_split(
        indices, test_size=TEST_SIZE, random_state=random_state,
    )
    X = prepared.drop(columns=[TARGET_COLUMN, ID_COLUMN], errors="ignore")
    y = prepared[TARGET_COLUMN].astype(float)
    pipeline, feature_columns = _create_pipeline(algorithm, prepared)
    grid, distributions = _search_spaces(algorithm)
    cv_folds = min(cv, len(train_indices))
    finite_candidates = _randomized_candidate_count(distributions)
    effective_n_iter = min(n_iter, finite_candidates) if finite_candidates is not None else n_iter
    shared = {"estimator": pipeline, "scoring": "neg_root_mean_squared_error",
              "cv": cv_folds, "n_jobs": 1, "refit": True, "error_score": "raise"}
    grid_search = GridSearchCV(param_grid=grid, **shared)
    random_search = RandomizedSearchCV(
        param_distributions=distributions, n_iter=effective_n_iter,
        random_state=random_state, **shared,
    )
    X_train, y_train = X.iloc[train_indices], y.iloc[train_indices]
    X_test, y_test = X.iloc[test_indices], y.iloc[test_indices]
    grid_search.fit(X_train, y_train)
    random_search.fit(X_train, y_train)
    searches = {"GridSearchCV": grid_search, "RandomizedSearchCV": random_search}
    winner = max(searches, key=lambda name: searches[name].best_score_)
    winner_search = searches[winner]
    model_comparisons = {}
    for name, search in searches.items():
        predicted = search.best_estimator_.predict(X_test)
        model_comparisons[name] = {
            "y_true": y_test.to_numpy(), "y_predicted": predicted,
            "rmse": float(np.sqrt(mean_squared_error(y_test, predicted))),
            "r2": float(r2_score(y_test, predicted)),
        }
    winner_prediction = model_comparisons[winner]["y_predicted"]
    residuals = y_test.to_numpy() - winner_prediction
    metrics = {
        "r2": float(r2_score(y_test, winner_prediction)),
        "rmse": float(np.sqrt(mean_squared_error(y_test, winner_prediction))),
        "mae": float(mean_absolute_error(y_test, winner_prediction)),
        "bias": float(np.mean(residuals)),
    }
    result_dir = directories["csv_salida"]
    predictions_path = result_dir / f"{run_id}_predicciones.csv"
    predictions_frame = pd.DataFrame({"y_real": y_test.to_numpy(),
                                      "y_predicho": winner_prediction,
                                      "residuo": residuals})
    if ID_COLUMN in prepared:
        predictions_frame.insert(0, ID_COLUMN, prepared[ID_COLUMN].iloc[test_indices].to_numpy())
    predictions_frame.to_csv(predictions_path, index=False, encoding="utf-8-sig")

    model_path = directories["modelos"] / f"{run_id}_{algorithm}.joblib"
    from joblib import dump
    dump(winner_search.best_estimator_, model_path)
    grid_results_path = directories["busquedas"] / f"{run_id}_gridsearch.csv"
    random_results_path = directories["busquedas"] / f"{run_id}_randomizedsearch.csv"
    pd.DataFrame(grid_search.cv_results_).to_csv(grid_results_path, index=False, encoding="utf-8-sig")
    pd.DataFrame(random_search.cv_results_).to_csv(random_results_path, index=False, encoding="utf-8-sig")

    correlation_data = prepared.select_dtypes(include=["number", "bool"])
    correlation_matrix = correlation_data.corr(method="pearson")
    report_path = directories["informes"] / f"{run_id}_informe.pdf"
    EdaPreprocesado().generate_report(
        frame, prepared, mappings, report_path, ALGORITHM_LABELS[algorithm],
        correlation_matrix, model_comparisons=model_comparisons,
    )
    parameters = {key.removeprefix("model__").removeprefix("polynomial__"): _native_scalar(value)
                  for key, value in winner_search.best_params_.items()}
    search_comparison = {
        name: {"best_parameters": {
                   key.removeprefix("model__").removeprefix("polynomial__"): _native_scalar(value)
                   for key, value in search.best_params_.items()},
               "best_cv_rmse": float(-search.best_score_),
               "candidates_evaluated": int(len(search.cv_results_["params"]))}
        for name, search in searches.items()
    }
    summary = {
        "status": "completed", "algorithm": algorithm,
        "algorithm_label": ALGORITHM_LABELS[algorithm], "winner": winner,
        "best_parameters": parameters,
        "best_cv_rmse": float(-winner_search.best_score_),
        "search_comparison": search_comparison,
        "cv_folds": int(cv_folds), "n_iter_requested": int(n_iter),
        "n_iter_used": int(effective_n_iter), "n_rows": int(len(prepared)),
        "n_features": int(len(feature_columns)), "feature_columns": feature_columns,
        "metrics": metrics,
        "comparison_metrics": {name: {"rmse": values["rmse"], "r2": values["r2"]}
                               for name, values in model_comparisons.items()},
        "artifacts": {
            "report_pdf": str(report_path), "predictions_csv": str(predictions_path),
            "model": str(model_path), "gridsearch_results_csv": str(grid_results_path),
            "randomizedsearch_results_csv": str(random_results_path),
        },
    }
    metadata_path = directories["busquedas"] / f"{run_id}_resumen.json"
    summary["run_id"] = run_id
    summary["artifacts"]["summary_json"] = str(metadata_path)
    metadata_path.write_text(json.dumps(summary, ensure_ascii=False, indent=2, default=str),
                              encoding="utf-8")
    return summary


def run_algorithm_casuisticas(algorithm: str, datasets_dir: Path | None = None,
                              output_root: Path | None = None) -> list[dict]:
    """Ejecuta un algoritmo en las seis casuísticas incluidas en el proyecto."""
    project_root = Path(__file__).resolve().parents[1]
    datasets_dir = Path(datasets_dir or project_root / "proyecto_eda_preprocesado_latex" / "datos")
    output_root = Path(output_root or project_root / "runs" / f"{algorithm}_casuisticas")
    dataset_names = [
        "01_referencia_limpia.csv", "02_valores_ausentes.csv",
        "03_valores_atipicos.csv", "04_categorias_raras.csv",
        "05_duplicados.csv", "06_sesgo_y_heterocedasticidad.csv",
    ]
    results = []
    for dataset_name in dataset_names:
        csv_path = datasets_dir / dataset_name
        if not csv_path.is_file():
            results.append({"dataset": dataset_name, "error": "No se encontró el CSV."})
            continue
        try:
            frame = pd.read_csv(csv_path, encoding="utf-8-sig")
            case_output = output_root / csv_path.stem
            run_id = f"{csv_path.stem}_{RANDOM_STATE}"
            result = train_algorithm_frame(frame, algorithm, case_output, run_id)
            result["dataset"] = dataset_name
            results.append(result)
            print(f"{dataset_name}: ganador={result['winner']}, RMSE={result['metrics']['rmse']:.4g}")
        except Exception as error:
            results.append({"dataset": dataset_name, "error": str(error)})
            print(f"ERROR {dataset_name}: {error}")
    output_root.mkdir(parents=True, exist_ok=True)
    summary_path = output_root / "resumen_entrenamientos.csv"
    rows = []
    for result in results:
        row = {key: result.get(key) for key in ("dataset", "winner", "best_cv_rmse", "error")}
        row["best_parameters"] = json.dumps(result.get("best_parameters", {}), ensure_ascii=False, default=str)
        row.update(result.get("metrics", {}))
        rows.append(row)
    pd.DataFrame(rows).to_csv(summary_path, index=False, encoding="utf-8-sig")
    print(f"Resumen de {algorithm}: {summary_path}")
    return results
