"""Entrenamiento de Regresión Ridge para las seis casuísticas del proyecto.

Ejecutar desde la raíz del proyecto:
    python regression/train_ridge_casuisticas.py

El módulo común compara GridSearchCV y RandomizedSearchCV y guarda
informes, modelos, predicciones y parámetros ganadores bajo runs/.
"""
from pathlib import Path
import sys

# Añade la raíz del proyecto a las importaciones al ejecutar este archivo directo.
PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from regression.casuisticas import run_algorithm_casuisticas

ALGORITHM = "ridge"

if __name__ == "__main__":
    run_algorithm_casuisticas(ALGORITHM)
