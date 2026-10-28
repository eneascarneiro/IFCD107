"""Construcción del modelo AdaBoostRegressor para la API tabular."""
from sklearn.ensemble import AdaBoostRegressor
from sklearn.pipeline import Pipeline
from sklearn.tree import DecisionTreeRegressor


def build_adaboost_regression_pipeline(preprocessor, options):
    """Crea un pipeline AdaBoost con árboles de regresión como aprendices base.

    AdaBoost entrena una secuencia de estimadores y aumenta la atención a los
    casos que los pasos anteriores predijeron peor. Un árbol poco profundo es
    un modelo base sencillo; n_estimators y learning_rate controlan el conjunto.
    """
    # Este árbol es una pieza del ensamblado, no el modelo que usa la API por sí
    # solo. Limitar su profundidad mantiene cada pieza relativamente simple.
    base_estimator = DecisionTreeRegressor(
        max_depth=options["max_depth"],
        random_state=options["random_state"],
    )
    # AdaBoost combina las predicciones de muchos árboles base en una salida.
    estimator = AdaBoostRegressor(
        estimator=base_estimator,
        n_estimators=options["n_estimators"],
        learning_rate=options["learning_rate"],
        loss=options["loss"],
        random_state=options["random_state"],
    )
    return Pipeline([("preprocess", preprocessor), ("adaboost", estimator)])
