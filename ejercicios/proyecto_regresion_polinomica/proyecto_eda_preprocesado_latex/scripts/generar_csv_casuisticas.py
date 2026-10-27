"""Genera seis CSV sintéticos para practicar EDA y preprocesado.

Uso: python scripts/generar_csv_casuisticas.py
Solo usa la biblioteca estándar de Python.
"""
import csv
import math
import random
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
DEST = ROOT / "datos"
FIELDS = ["expediente_id", "distrito", "tipo_tramite", "superficie_solicitada_m2",
          "requerimientos_documentacion", "mes_presentacion", "x", "y"]
DISTRICTS = ["Centro", "Chamberí", "Retiro", "Salamanca", "Usera"]
TYPES = ["alta", "renovacion", "modificacion"]


def base_rows(seed=107, n=180):
    rng = random.Random(seed)
    rows = []
    for i in range(n):
        area = round(rng.uniform(12, 110), 1)
        docs = rng.choices([0, 1, 2, 3], weights=[.48, .30, .17, .05])[0]
        month = rng.randint(1, 12)
        district = rng.choice(DISTRICTS)
        kind = rng.choices(TYPES, weights=[.4, .38, .22])[0]
        x = round(min(100, max(10, 12 + area * .48 + docs * 4 + rng.gauss(0, 7))), 2)
        type_effect = {"alta": 8, "renovacion": -3, "modificacion": 3}[kind]
        y = round(18 + .52 * x + .003 * x * x + docs * 3 + type_effect + rng.gauss(0, 6), 1)
        rows.append({"expediente_id": f"CASO-{i+1:04d}", "distrito": district,
                     "tipo_tramite": kind, "superficie_solicitada_m2": area,
                     "requerimientos_documentacion": docs, "mes_presentacion": month,
                     "x": x, "y": y})
    return rows


def write(name, rows):
    with (DEST / name).open("w", newline="", encoding="utf-8-sig") as f:
        writer = csv.DictWriter(f, fieldnames=FIELDS)
        writer.writeheader()
        writer.writerows(rows)


def main():
    DEST.mkdir(parents=True, exist_ok=True)
    # 1. Referencia limpia: valores completos, rangos plausibles y claves únicas.
    write("01_referencia_limpia.csv", base_rows())

    # 2. Ausencias: mezcla de faltantes numéricos y categóricos, incluida y.
    rows = base_rows(seed=108)
    for i in range(0, len(rows), 13):
        rows[i]["superficie_solicitada_m2"] = ""
    for i in range(5, len(rows), 17):
        rows[i]["distrito"] = ""
    for i in range(9, len(rows), 31):
        rows[i]["y"] = ""
    write("02_valores_ausentes.csv", rows)

    # 3. Atípicos: casos extremos deliberados para revisar IQR y contexto.
    rows = base_rows(seed=109)
    for i, (area, x, y) in enumerate([(420.0, 1000.0, 980.0), (0.2, 4.0, 3.0),
                                      (850.0, 1500.0, 1600.0), (11.0, 8.0, 210.0)]):
        rows[i]["superficie_solicitada_m2"] = area
        rows[i]["x"] = x
        rows[i]["y"] = y
    write("03_valores_atipicos.csv", rows)

    # 4. Categorías raras, espacios y mayúsculas para estudiar normalización.
    rows = base_rows(seed=110)
    for i in range(0, len(rows), 23):
        rows[i]["distrito"] = "Distrito experimental muy poco frecuente"
    for i in range(7, len(rows), 29):
        rows[i]["tipo_tramite"] = "  ALTA "
    for i in range(11, len(rows), 37):
        rows[i]["distrito"] = "Centro "
    write("04_categorias_raras.csv", rows)

    # 5. Duplicados exactos y claves repetidas con valores incompatibles.
    rows = base_rows(seed=111, n=150)
    rows.extend([dict(rows[4]), dict(rows[19])])
    conflict = dict(rows[25])
    conflict["y"] = round(float(conflict["y"]) + 35, 1)
    rows.append(conflict)
    write("05_duplicados.csv", rows)

    # 6. Predictor sesgado y varianza del objetivo creciente con x.
    rng = random.Random(112)
    rows = []
    for i in range(180):
        area = round(min(200, 8 + rng.lognormvariate(2.7, .8)), 1)
        docs = rng.choices([0, 1, 2, 3], weights=[.55, .25, .15, .05])[0]
        month = rng.randint(1, 12)
        district, kind = rng.choice(DISTRICTS), rng.choice(TYPES)
        x = round(min(100, 10 + 90 * (area / 210) ** .72), 2)
        noise = rng.gauss(0, 2 + .18 * x)
        y = round(12 + .68 * x + .006 * x * x + 4 * docs + noise, 1)
        rows.append({"expediente_id": f"CASO-{i+1:04d}", "distrito": district,
                     "tipo_tramite": kind, "superficie_solicitada_m2": area,
                     "requerimientos_documentacion": docs, "mes_presentacion": month,
                     "x": x, "y": y})
    write("06_sesgo_y_heterocedasticidad.csv", rows)
    print(f"Generados 6 CSV en {DEST}")


if __name__ == "__main__":
    main()
