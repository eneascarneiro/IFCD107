# Técnicas de EDA y preprocesado con Python

Proyecto didáctico completo. `main.tex` es el informe LaTeX; `scripts/eda_preprocesado.py` ejecuta el análisis; `datos/datos_ejemplo_complejo.csv` es el conjunto sintético. Consulta el informe para objetivos, conceptos, decisiones y pasos de ejecución.

## Ejecutar

Desde la carpeta del proyecto:

```powershell
python -m venv .venv
.venv\Scripts\Activate.ps1
python -m pip install -r requirements.txt
python scripts/eda_preprocesado.py
```

Las salidas se guardan en `resultados/`: perfil y estadísticas, datos limpios, particiones preprocesadas, transformador serializado y gráficos. La preparación usa split 80/20, semilla fija y ajusta las transformaciones exclusivamente con entrenamiento.

## Compilar LaTeX

Abre el proyecto en el editor LaTeX de Codex y compila `main.tex`, o utiliza `latexmk -pdf main.tex` si tienes LaTeX instalado. El documento fuente también se puede editar con cualquier editor.

## Datos

Los datos son enteramente sintéticos. `x` es un índice didáctico de complejidad y `y` días simulados; no son expedientes, plazos ni resultados reales. No se deben extraer conclusiones sobre administraciones reales.
