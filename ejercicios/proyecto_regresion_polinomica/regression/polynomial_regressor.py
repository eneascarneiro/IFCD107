"""Clase de regresión polinómica y generación de artefactos."""
from dataclasses import dataclass
import pickle
from pathlib import Path

import numpy as np
from sklearn.base import BaseEstimator, RegressorMixin
from sklearn.model_selection import RandomizedSearchCV
from sklearn.pipeline import Pipeline
from regression.parameter_search import PARAM_DISTRIBUTIONS





@dataclass
class FitResult:
    """Agrupa las predicciones y mediciones que produce un ajuste."""

    coefficients: np.ndarray
    x: np.ndarray
    predicted: np.ndarray
    residuals: np.ndarray
    metrics: dict
    rank: int | None = None
    singular_values: np.ndarray | None = None
    rcond: float | None = None
    covariance: np.ndarray | None = None
    fit_residuals: np.ndarray | None = None


class PolynomialRegressor:
    """Ajusta un polinomio por mínimos cuadrados usando numpy."""

    def __init__(self, X_train, X_test, y_train, y_test,degree: int = 2):
        # Guardamos por separado entrenamiento y prueba: solo el primero calcula
        # el polinomio, mientras que el segundo sirve para comprobar el resultado.
        if isinstance(degree, bool) or not isinstance(degree, int) or not 1 <= degree <= 10:
            raise ValueError("degree debe estar entre 1 y 10.")
        self.degree = degree
        self.coefficients_ = None
        self.X_train = X_train
        self.X_test = X_test
        self.y_train = y_train
        self.y_test = y_test

    def fit(self, rcond=None, full: bool = False, w=None, cov=False) -> FitResult:
        """Ajusta el polinomio; los argumentos siguen las opciones de ``numpy.polyfit``.

        ``full=True`` incluye rango, valores singulares y residuos del ajuste en
        FitResult. ``cov`` puede ser True (covarianza escalada) o ``'unscaled'``.
        """
        if not isinstance(full, (bool, np.bool_)):
            raise TypeError("full debe ser booleano.")
        if not (isinstance(cov, (bool, np.bool_)) or (isinstance(cov, str) and cov == "unscaled")):
            raise ValueError("cov debe ser False, True o 'unscaled'.")
        cov = "unscaled" if isinstance(cov, str) else bool(cov)
        # Convertimos a arrays numéricos una sola vez. x/y de train aprenden los
        # coeficientes; x/y de test se reservan para medir el ajuste.
        x = np.asarray(self.X_train, dtype=float)
        y = np.asarray(self.y_train, dtype=float)
        x_t = np.asarray(self.X_test, dtype=float)
        y_t = np.asarray(self.y_test, dtype=float)
        if x.ndim != 1 or y.ndim != 1 or len(x) != len(y):
            raise ValueError("x e y deben ser vectores de igual longitud.")
        # Para grado d se requieren al menos d+1 observaciones y d+1 valores x
        # distintos; de lo contrario el problema está subdeterminado.
        if len(x) <= self.degree:
            raise ValueError("Se necesitan más observaciones que el grado del polinomio.")
        if not np.isfinite(x).all() or not np.isfinite(y).all():
            raise ValueError("Los datos deben ser finitos.")
        if np.unique(x).size <= self.degree:
            raise ValueError("Se necesitan más valores x distintos que el grado del polinomio.")
        weights = None if w is None else np.asarray(w, dtype=float)
        if weights is not None and (weights.ndim != 1 or len(weights) != len(x)):
            raise ValueError("w debe ser un vector con un peso por observación de entrenamiento.")
        if weights is not None and (not np.isfinite(weights).all() or np.any(weights < 0)):
            raise ValueError("w debe contener pesos finitos no negativos.")
        if rcond is not None:
            if isinstance(rcond, bool) or not np.isscalar(rcond):
                raise ValueError("rcond debe ser un número finito no negativo o None.")
            try:
                rcond = float(rcond)
            except (TypeError, ValueError):
                raise ValueError("rcond debe ser un número finito no negativo o None.") from None
            if not np.isfinite(rcond) or rcond < 0:
                raise ValueError("rcond debe ser un número finito no negativo o None.")
        # polyfit calcula los coeficientes. w pondera los residuos; full pide
        # diagnósticos SVD y cov pide la incertidumbre de los coeficientes.
        fitted = np.polyfit(x, y, self.degree, rcond=rcond, full=bool(full), w=weights, cov=cov)
        if full:
            coefficients, fit_residuals, rank, singular_values, used_rcond = fitted
            covariance = None
        elif cov is not False:
            coefficients, covariance = fitted
            fit_residuals = np.array([], dtype=float)
            rank = None
            singular_values = None
            used_rcond = rcond
        else:
            coefficients = fitted
            covariance = None
            fit_residuals = np.array([], dtype=float)
            rank = None
            singular_values = None
            used_rcond = rcond
        self.coefficients_ = coefficients.copy()
        # Evaluamos con X_test, que no se usó para calcular los coeficientes.
        predicted = np.polyval(coefficients, x_t)
        # El residuo es el valor real menos la predicción para cada fila.
        residuals = y_t - predicted
        # R² no está definido si y es constante. En el caso perfecto se informa 1;
        # si no lo es, se informa 0 como valor neutro y se conserva RMSE/MAE.
        # R² compara el error del modelo con el error de predecir siempre la media.
        # RMSE y MAE resumen el tamaño típico de los errores en las unidades de y.
        ss_res = float(np.sum(residuals ** 2))
        ss_tot = float(np.sum((y_t - np.mean(y_t)) ** 2))
        r2 = 1.0 if ss_tot == 0 and ss_res == 0 else (0.0 if ss_tot == 0 else 1 - ss_res / ss_tot)
        metrics = {
            "r2": float(r2),
            "rmse": float(np.sqrt(np.mean(residuals ** 2))),
            "mae": float(np.mean(np.abs(residuals))),
        }
        return FitResult(np.asarray(coefficients), x_t.copy(), np.asarray(predicted), residuals, metrics,
                         rank, np.asarray(singular_values) if singular_values is not None else None,
                         float(used_rcond) if used_rcond is not None else None,
                         np.asarray(covariance) if covariance is not None else None,
                         np.asarray(fit_residuals) if full else None)

    def predict(self, x):
        """Predice con un modelo ya entrenado; x puede ser escalar o vector."""
        if self.coefficients_ is None:
            raise ValueError("El modelo aún no está entrenado.")
        values = np.asarray(x, dtype=float)
        if not np.isfinite(values).all():
            raise ValueError("x debe contener valores finitos.")
        # Los coeficientes se guardan en orden descendente, que es el convenio
        # esperado por polyval: a_n*x^n + ... + a_1*x + a_0.
        return np.polyval(self.coefficients_, values)

    def save(self, path: Path):
        """Serializa el modelo entrenado como pickle con extensión .pkt.

        Cargar archivos pickle únicamente si proceden de una fuente confiable.
        """
        if self.coefficients_ is None:
            raise ValueError("No se puede guardar un modelo sin entrenar.")
        path = Path(path)
        # Pickle conserva el objeto Python entrenado y sus coeficientes. El archivo
        # solo debe cargarse desde un origen confiable; pickle no es un formato
        # seguro para modelos recibidos de usuarios o servicios externos.
        with path.open("wb") as stream:
            pickle.dump(self, stream, protocol=pickle.HIGHEST_PROTOCOL)

    @classmethod
    def load(cls, path: Path):
        """Carga un .pkt creado por esta clase. No cargar archivos no confiables."""
        # Un pickle puede ejecutar código durante unpickle: validar el tipo después
        # no neutraliza un archivo malicioso, por eso el origen debe ser confiable.
        with Path(path).open("rb") as stream:
            model = pickle.load(stream)
        if not isinstance(model, cls) or model.coefficients_ is None:
            raise ValueError("El archivo no contiene un modelo PolynomialRegressor entrenado.")
        return model

    @staticmethod
    def write_predictions(path: Path, result: FitResult):
        path = Path(path)
        # La salida conserva observación, predicción y residuo fila a fila para que
        # el usuario pueda revisar errores y no solo el resumen agregado del modelo.
        values = np.column_stack((result.x, result.predicted + result.residuals, result.predicted, result.residuals))
        np.savetxt(path, values, delimiter=",", header="x,y_observado,y_predicho,residuo", comments="", fmt="%.10g")

    def write_plot(self, path: Path, x, y, result: FitResult):
        path = Path(path)
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt

        # Ordenar solo para dibujar la curva evita segmentos zigzagueantes cuando
        # las observaciones llegan desordenadas; el ajuste usa todas las filas.
        # matplotlib une puntos en el orden recibido; ordenar x dibuja una línea
        # continua aunque las observaciones se hayan recibido desordenadas.
        order = np.argsort(x)
        fig, ax = plt.subplots(figsize=(9, 5.5))
        ax.scatter(x, y, label="Observaciones", color="#1f77b4", alpha=0.85)
        ax.plot(np.asarray(x)[order], result.predicted[order], label=f"Ajuste polinómico (grado {self.degree})", color="#d62728")
        ax.set(title="Regresión polinómica", xlabel="x", ylabel="y")
        ax.grid(True, alpha=0.25)
        ax.legend()
        fig.tight_layout()
        fig.savefig(path, dpi=160)
        plt.close(fig)


class PolynomialSearchEstimator(RegressorMixin, BaseEstimator):
    """Adaptador para que la clase NumPy se pueda usar en Pipeline y CV.

    scikit-learn llama a fit(X, y) en cada pliegue y a predict(X) para puntuarlo.
    Este adaptador convierte la columna X al formato que espera PolynomialRegressor.
    """

    def __init__(self, degree=2, rcond=None, full=False, w=None, cov=False):
        self.degree = degree
        self.rcond = rcond
        self.full = full
        self.w = w
        self.cov = cov

    def fit(self, X, y, sample_weight=None):
        # Un polinomio de esta práctica tiene un único predictor, x. sklearn lo
        # entrega como matriz (filas, columnas); la clase base usa un vector.
        x = np.asarray(X, dtype=float)
        if x.ndim == 2 and x.shape[1] == 1:
            x = x[:, 0]
        if x.ndim != 1:
            raise ValueError("PolynomialSearchEstimator acepta exactamente una columna X.")
        # sample_weight se recorta automáticamente por RandomizedSearchCV para
        # cada pliegue. self.w queda como opción para un ajuste sin validación.
        weights = sample_weight if sample_weight is not None else self.w
        self.regressor_ = PolynomialRegressor(x, x, y, y, degree=self.degree)
        self.fit_result_ = self.regressor_.fit(
            rcond=self.rcond, full=self.full, w=weights, cov=self.cov
        )
        self.coef_ = self.regressor_.coefficients_
        self.n_features_in_ = 1
        return self

    def predict(self, X):
        if not hasattr(self, "regressor_"):
            raise ValueError("El estimador aún no está entrenado.")
        x = np.asarray(X, dtype=float)
        if x.ndim == 2 and x.shape[1] == 1:
            x = x[:, 0]
        if x.ndim != 1:
            raise ValueError("PolynomialSearchEstimator acepta exactamente una columna X.")
        return self.regressor_.predict(x)


# RandomizedSearchCV elige valores de estas listas para el paso `polynomial`.
# Los nombres llevan `polynomial__` porque ése es el nombre del paso del Pipeline.
# `w` se deja en None porque una matriz de pesos de tamaño fijo no es válida al
# cambiar los pliegues CV; pase los pesos reales a search.fit como sample_weight.
POLYNOMIAL_PARAM_DISTRIBUTIONS = PARAM_DISTRIBUTIONS["polynomial"]


def build_polynomial_search(n_iter=20, cv=5, random_state=42, param_distributions=None):
    """Construye la búsqueda reproducible para una variable x."""
    # Pipeline reúne el estimador bajo un nombre estable para que sus parámetros
    # puedan escribirse en el diccionario de búsqueda con el prefijo `polynomial__`.
    pipeline = Pipeline([("polynomial", PolynomialSearchEstimator())])
    return RandomizedSearchCV(
        estimator=pipeline,
        param_distributions=(param_distributions if param_distributions is not None
                             else POLYNOMIAL_PARAM_DISTRIBUTIONS),
        n_iter=n_iter,
        cv=cv,
        random_state=random_state,
        scoring="neg_mean_squared_error",
        refit=True,
    )
