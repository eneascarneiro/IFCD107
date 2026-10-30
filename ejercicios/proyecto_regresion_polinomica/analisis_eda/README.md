# EDA y preprocesado

El flujo usa `ejemplos/datos_ejemplo_complejo.csv`, un conjunto sintético incluido en el proyecto. Para ejecutar desde la raíz del proyecto:

```powershell
python analisis_eda/eda_preprocesado.py
```

Requiere las dependencias ya declaradas en `requirements.txt` (pandas, scikit-learn, matplotlib y seaborn; `joblib` se instala con scikit-learn). Genera `resultados/` con perfil de columnas, estadísticas descriptivas, CSV limpio, conjuntos train/test codificados, preprocesador serializado, informe Markdown y tres gráficas.

El identificador `expediente_id` queda fuera de los predictores. El objetivo es `y`; el resto de columnas son predictores. El split es 80/20 con semilla 42. Imputación, escalado y one-hot se ajustan únicamente con train para evitar fuga de información. Los valores extremos se señalan visualmente y se conservan.
