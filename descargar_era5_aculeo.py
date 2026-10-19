# -*- coding: utf-8 -*-
"""Descarga ERA5-Land mensual para la zona de la Laguna de Aculeo (Copernicus CDS).

Requisitos:
  pip install cdsapi netCDF4
  Archivo de credenciales ~/.cdsapirc  (en Windows: C:\\Users\\<usuario>\\.cdsapirc) con
      url: https://cds.climate.copernicus.eu/api
      key: <tu token personal>
  Haber aceptado la licencia del dataset en la página del CDS (una sola vez).

Uso:  python descargar_era5_aculeo.py [carpeta_de_salida]
  Sin argumento guarda en <carpeta del script>/data/raw/aculeo/
"""
import sys
from pathlib import Path

import cdsapi

CARPETA = Path(sys.argv[1]) if len(sys.argv) > 1 else Path(__file__).resolve().parent / "data" / "raw" / "aculeo"
SALIDA = CARPETA / "era5land_aculeo_mensual.nc"

dataset = "reanalysis-era5-land-monthly-means"
request = {
    "product_type": ["monthly_averaged_reanalysis"],
    "variable": ["total_precipitation", "total_evaporation",
                 "volumetric_soil_water_layer_1", "2m_temperature"],
    "year": [str(y) for y in range(1985, 2026)],
    "month": [f"{m:02d}" for m in range(1, 13)],
    "time": ["00:00"],
    "data_format": "netcdf",
    "download_format": "unarchived",
    "area": [-33.6, -71.2, -34.1, -70.7],   # norte, oeste, sur, este (Aculeo)
}

if __name__ == "__main__":
    SALIDA.parent.mkdir(parents=True, exist_ok=True)
    client = cdsapi.Client()
    client.retrieve(dataset, request).download(str(SALIDA))
    print("Listo:", SALIDA.resolve())
