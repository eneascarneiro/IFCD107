"""Construcción del modelo GradientBoostingRegressor para la API tabular."""
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.pipeline import Pipeline


def build_gradient_boosting_regression_pipeline(preprocessor, options):
    """Crea un pipeline de árboles que corrigen errores por etapas.

    Cada iteración añade árboles a la predicción acumulada. learning_rate reduce
    cuánto aporta cada etapa; n_estimators indica cuántas etapas se ajustan.
    max_depth limita la complejidad de cada árbol base.
    """
    # El regresor final recibe los datos ya imputados y codificados. El
    # preprocesador se ajusta solo cuando se llama a fit() sobre el pipeline.
    estimator = GradientBoostingRegressor(
        n_estimators=options["n_estimators"],
        learning_rate=options["learning_rate"],
        max_depth=options["max_depth"],
        loss=options["loss"],
        subsample=options["subsample"],
        random_state=options["random_state"],
    )
    return Pipeline([("preprocess", preprocessor), ("gradient_boosting", estimator)])
