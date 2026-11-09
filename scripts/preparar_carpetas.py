"""Arma las carpetas que espera analisis_consultoria.py a partir de data/ (no modifica data/).

analisis_consultoria.py es copia del Anexo E.2 del informe final y trabaja sobre copias de los datos:
  datos/processed, datos/raw, datos/miscanti     copia de trabajo
  respaldo/data/processed, respaldo/data/raw,     copia de respaldo (para comparar huellas SHA-256)
  respaldo/miscanti
  ejecucion_torca/data/processed                 resultados del notebook de Torca (corrida del 8-nov)

Uso:  python scripts/preparar_carpetas.py   y luego   python scripts/analisis_consultoria.py
"""
import shutil
from datetime import datetime, timezone
from pathlib import Path

BASE = Path(__file__).resolve().parents[1]
PROC, RAW = BASE / "data" / "processed", BASE / "data" / "raw"
miscanti = sorted(p for p in PROC.glob("*.csv") if "miscanti" in p.name)
torca_aculeo = sorted(p for p in PROC.glob("*.csv") if "miscanti" not in p.name)
crudos = sorted(RAW.glob("*.csv"))

destinos = {
    "datos/processed": torca_aculeo, "datos/raw": crudos, "datos/miscanti": miscanti,
    "respaldo/data/processed": torca_aculeo, "respaldo/data/raw": crudos, "respaldo/miscanti": miscanti,
    "ejecucion_torca/data/processed": torca_aculeo,
}
for carpeta, archivos in destinos.items():
    destino = BASE / carpeta
    destino.mkdir(parents=True, exist_ok=True)
    for archivo in archivos:
        shutil.copy2(archivo, destino / archivo.name)
    print(f"{carpeta}: {len(archivos)} archivos")

# El paso 1 del análisis lee la fecha en que se hizo el respaldo.
(BASE / "salidas").mkdir(exist_ok=True)
(BASE / "salidas" / "fecha_respaldo.txt").write_text(datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"))
