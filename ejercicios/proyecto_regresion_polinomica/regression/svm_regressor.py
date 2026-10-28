"""Construcción del modelo SVR usado por el endpoint de regresión SVM.

SVR predice una cantidad numérica. El preprocesador recibido transforma las
columnas antes del ajuste; mantenerlo dentro del mismo Pipeline evita que las
estadísticas del conjunto de prueba se filtren al entrenamiento.
"""
from sklearn.pipeline import Pipeline
from sklearn.svm import SVR


def build_svm_regression_pipeline(preprocessor, options):
    """Crea el pipeline y conecta las opciones validadas con sklearn.SVR.

    C controla la penalización de errores, epsilon define una banda de error
    tolerada y kernel selecciona la forma de la relación aprendida. gamma y
    degree solo tienen efecto en algunos kernels.
    """
    # SVR es el estimador final. En el ajuste, Pipeline aprende primero las
    # transformaciones y luego entrega la matriz transformada a SVR.
    estimator = SVR(
        kernel=options["kernel"],
        C=options["C"],
        epsilon=options["epsilon"],
        gamma=options["gamma"],
        degree=options["degree"],
    )
    return Pipeline([("preprocess", preprocessor), ("svm", estimator)])
