"""Entrenamiento y evaluación de regresión lineal para datos x/y."""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import train_test_split


@dataclass
class LinearRegressionResult:
    """Modelo reajustado con todos los datos y métricas del holdout."""

    model: LinearRegression
    metrics: dict[str, float]
    train_rows: int
    test_rows: int


class LinearRegressor:
    """Ajusta una recta con un holdout reproducible y devuelve un modelo final."""

    def __init__(self, test_size: float = 0.2, random_state: int = 42):
        if not 0.1 <= test_size <= 0.4:
            raise ValueError("test_size debe estar entre 0.1 y 0.4.")
        if isinstance(random_state, bool) or not isinstance(random_state, int) or random_state < 0:
            raise ValueError("random_state debe ser un entero no negativo.")
        self.test_size = float(test_size)
        self.random_state = random_state

    def fit(self, X, y) -> LinearRegressionResult:
        """Evalúa con datos reservados y reajusta el estimador para publicación."""
        features = np.asarray(X, dtype=float)
        target = np.asarray(y, dtype=float).reshape(-1)
        if features.ndim == 1:
            features = features.reshape(-1, 1)
        if features.ndim != 2 or features.shape[1] != 1:
            raise ValueError("La regresión lineal de este ejercicio requiere una columna x.")
        if len(features) != len(target) or len(target) < 10:
            raise ValueError("X e y deben tener la misma longitud y al menos 10 filas.")
        if not np.isfinite(features).all() or not np.isfinite(target).all():
            raise ValueError("Los datos deben contener solo valores finitos.")

        X_train, X_test, y_train, y_test = train_test_split(
            features, target, test_size=self.test_size, random_state=self.random_state
        )
        evaluator = LinearRegression().fit(X_train, y_train)
        predicted = evaluator.predict(X_test)
        metrics = {
            "r2": float(r2_score(y_test, predicted)),
            "rmse": float(np.sqrt(mean_squared_error(y_test, predicted))),
            "mae": float(mean_absolute_error(y_test, predicted)),
        }
        # Las métricas quedan ligadas al holdout; para inferencia se aprovechan
        # todas las filas una vez terminada la evaluación.
        serving_model = LinearRegression().fit(features, target)
        return LinearRegressionResult(serving_model, metrics, len(X_train), len(X_test))
