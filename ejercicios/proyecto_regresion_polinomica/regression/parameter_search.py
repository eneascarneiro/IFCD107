"""Espacios de opciones discretas para las búsquedas de regresión.

Cada valor está escrito como una lista para poder mostrarlo al cliente y
reutilizar exactamente el mismo catálogo en RandomizedSearchCV.
"""
from __future__ import annotations

from copy import deepcopy


PARAM_DISTRIBUTIONS = {
    "polynomial": {
        "polynomial__degree": [1, 2, 3, 4, 5],
        "polynomial__rcond": [None, 1e-12, 1e-10, 1e-8],
        "polynomial__full": [False, True],
        "polynomial__w": [None],
        "polynomial__cov": [False, "unscaled"],
    },
    "random_forest": {
        "random_forest__n_estimators": [50, 100, 200, 500, 1000],
        "random_forest__max_depth": [None, 5, 10, 20, 30],
        "random_forest__min_samples_split": [2, 5, 10],
        "random_forest__min_samples_leaf": [1, 2, 4],
        "random_forest__max_features": ["sqrt", "log2", None],
        "random_forest__bootstrap": [True, False],
    },
    "adaboost": {
        "adaboost__n_estimators": [50, 100, 200, 500, 1000],
        "adaboost__learning_rate": [0.01, 0.1, 1.0],
        "adaboost__loss": ["linear", "square", "exponential"],
        "adaboost__estimator__max_depth": [1, 2, 3, 5],
    },
    "gradient_boosting": {
        "gradient_boosting__n_estimators": [100, 200, 400, 800],
        "gradient_boosting__learning_rate": [0.01, 0.03, 0.1, 0.2],
        "gradient_boosting__max_depth": [2, 3, 5, 7],
        "gradient_boosting__min_samples_split": [2, 5, 10],
        "gradient_boosting__min_samples_leaf": [1, 2, 4],
        "gradient_boosting__subsample": [0.6, 0.8, 1.0],
        "gradient_boosting__max_features": ["sqrt", "log2", None],
    },
    "svm": {
        "svm__C": [0.1, 1.0, 10.0, 50.0, 100.0],
        "svm__gamma": ["scale", "auto", 0.0001, 0.001, 0.01, 0.1, 1.0],
        "svm__kernel": ["rbf", "poly", "sigmoid"],
        "svm__epsilon": [0.001, 0.01, 0.1, 0.25, 0.5],
        "svm__degree": [2, 3, 4, 5],
    },
    "knn": {
        "knn__n_neighbors": [3, 5, 7, 9, 11, 15, 20, 30, 39],
        "knn__weights": ["uniform", "distance"],
        "knn__p": [1, 2],
    },
}


class ParameterSearchCatalog:
    """Devuelve listas candidatas y su representación pública."""

    def get_param_distributions_iterator(self, model_name: str) -> dict:
        """Devuelve una copia del espacio de búsqueda del modelo solicitado."""
        if model_name not in PARAM_DISTRIBUTIONS:
            raise ValueError(f"No existe un espacio de búsqueda para '{model_name}'.")
        return deepcopy(PARAM_DISTRIBUTIONS[model_name])

    def as_items(self) -> dict:
        """Lista cada parámetro con sus opciones para la respuesta de la API."""
        return {
            model_name: [
                {"parameter": parameter, "items": list(values)}
                for parameter, values in self.get_param_distributions_iterator(model_name).items()
            ]
            for model_name in PARAM_DISTRIBUTIONS
        }


PARAMETER_CATALOG = ParameterSearchCatalog()
