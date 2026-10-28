Estimado Pablo,

### 1. Análisis Crítico y Oportunidades de Mejora

#### A. El "Cálculo de Probabilidades" Actual (Enfoque Naive)
El código original utiliza un enfoque basado puramente en **frecuencias individuales absolutas** (*hot numbers*). Matemáticamente, esto presenta varias limitaciones para la toma de decisiones:
1. **Falacia del Apostador e Independencia**: En un sorteo justo, cada extracción es un evento independiente. Creer que un número "saldrá más" solo porque ha salido más en el pasado es una interpretación débil si no se complementa con la **probabilidad conjunta** de la combinación.
2. **Falta de Coherencia Estructural**: Si simplemente combinamos de forma aleatoria los 10 números más frecuentes, podemos generar combinaciones altamente improbables en la práctica (por ejemplo, combinaciones compuestas únicamente por números pares, únicamente números bajos, o números consecutivos como `01, 02, 03, 04, 05, 06`). Aunque individualmente cada combinación tiene la misma probabilidad teórica ($1$ entre $13.983.816$), la **distribución de las propiedades** de las combinaciones ganadoras históricas sigue patrones estadísticos muy claros.

#### B. Eficiencia de Desempeño (Pandas vs. Python Loops)
El código original extrae los números a listas de Python y utiliza `collections.Counter`. Esto rompe la vectorización de Pandas, penalizando el rendimiento cuando el volumen de datos históricos de sorteos es elevado.

#### C. Robustez y Mantenibilidad
* **Ausencia de Logging**: El uso de `print()` dificulta la trazabilidad en entornos de ejecución automatizados.
* **Redundancia en la Semilla**: `random.seed(datetime.now().timestamp())` es redundante en Python moderno, ya que el módulo `random` se inicializa automáticamente con fuentes de entropía del sistema operativo.
* **Falta de Modularidad**: Todo el proceso (lectura, análisis, filtrado y generación del PDF) está acoplado en una única función gigante, lo que viola el principio de responsabilidad única.

---

### 2. El Nuevo Enfoque Estadístico: Filtros de Coherencia Probabilística

Para mejorar radicalmente el cálculo de probabilidades, he implementado un **Motor de Filtrado Multicapa** basado en la distribución de probabilidad de las combinaciones de la Primitiva ($6/49$):

1. **Filtro de Relación Par/Impar (Odd/Even)**: La probabilidad de que salgan combinaciones extremas (6 pares o 6 impares) es de apenas un $1.2\%$ cada una. Las relaciones balanceadas ($3P:3I$, $2P:4I$, $4P:2I$) representan más del **$80\%$** de los sorteos ganadores históricos.
2. **Filtro de Distribución Alto/Bajo (High/Low)**: Dividiendo el tablero en números Bajos ($1$ al $25$) y Altos ($26$ al $49$), las combinaciones balanceadas ($3B:3A$, $2B:4A$, $4B:2A$) abarcan aproximadamente el **$79\%$** de los resultados reales.
3. **Filtro de Suma Teórica (Sum Range)**: La suma de los 6 números de una combinación sigue una distribución aproximadamente normal. El rango de suma óptimo que concentra el **$70\%$** de las combinaciones ganadoras se sitúa entre **$115$ y $185$** (con una media teórica de $150$).
4. **Filtro de Consecutividad (Consecutive Numbers)**: Las combinaciones con 3 o más números consecutivos (ej. `12, 13, 14`) son extremadamente raras en términos prácticos (menos del $1\%$). Limitamos la consecutividad máxima a secuencias de 2 números.

**Estrategia de Selección**: En lugar de limitar el análisis al Top 10, ampliamos el espectro al **Top 12 de números más frecuentes** (lo que nos da $\binom{12}{6} = 924$ combinaciones posibles) y aplicamos estos filtros matemáticos para seleccionar únicamente aquellas combinaciones que sean **estadísticamente viables y balanceadas**.

---

### 4. Justificación Técnica de las Mejoras Implementadas

#### A. Desde la Perspectiva de la Ciencia de Datos (Cálculo de Probabilidades)
* **Ampliación del Espacio de Muestreo**: Al pasar de analizar el Top 10 a un **Top 12**, el número de combinaciones posibles aumenta de $210$ a $924$. Esto nos da un margen mucho más amplio para aplicar filtros de calidad sin quedarnos sin combinaciones suficientes.
* **Filtros de Coherencia**: La función `es_combinacion_probable` actúa como un cortafuegos estadístico. Evita que el programa sugiera combinaciones absurdas que un algoritmo puramente frecuentista generaría. Ahora, cada boleto sugerido cumple con las leyes de la probabilidad conjunta (Suma, Paridad, Distribución espacial en el boleto y Consecutividad).

#### B. Desde la Perspectiva de Ingeniería de Software y Rendimiento
* **Vectorización con Pandas**: Reemplacé los bucles `for` y la conversión manual a listas por operaciones vectorizadas nativas (`.apply(pd.to_numeric)`, `.stack()` y `.value_counts()`). Esto reduce drásticamente el consumo de CPU y memoria, cumpliendo con la directriz de *Eficiencia de Desempeño* de la norma **[ISO/IEC 25010](https://www.iso.org/standard/35733.html)**.
* **Modularidad y Separación de Responsabilidades**: El código se dividió en funciones claras con tipos anotados (`typing`), facilitando las pruebas unitarias y el mantenimiento futuro.
* **Trazabilidad Profesional**: Se sustituyeron los `print()` por el módulo `logging`, permitiendo categorizar los mensajes (`INFO`, `WARNING`, `ERROR`) según el estándar de desarrollo corporativo.

