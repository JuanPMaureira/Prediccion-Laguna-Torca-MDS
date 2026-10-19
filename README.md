# Predicción de la superficie hídrica de la Laguna Torca

Proyecto de Data Science (Magíster) · Juan Pablo Maureira

Análisis multitemporal de la superficie hídrica del humedal Laguna Torca (Región del Maule, Chile) con imágenes Landsat (Google Earth Engine), su relación con el clima (CR2MET) y su proyección hasta 2100 bajo los escenarios RCP 2.6 y RCP 8.5. Incluye además un análisis de la Laguna de Aculeo como cuerpo de agua candidato.

## Estructura

```
notebooks/
  Laguna_Torca_v2_CR2MET.ipynb   notebook principal (Torca)
  Laguna_Aculeo_v1.ipynb         mismo análisis para la Laguna de Aculeo
data/
  raw/         proyecciones climáticas CR2 (CSV) y descargas ERA5-Land (.nc)
  processed/   series satelitales, clima extraído y resultados que generan los notebooks
informe/       informe en LaTeX (informe_laguna_torca.tex) y sus figuras
outputs/presentations/   presentaciones Beamer (.tex y .pdf)
docs/          documento del proyecto
descargar_era5_aculeo.py script para descargar ERA5-Land mensual (Copernicus CDS)
```

## Cómo ejecutar

Los notebooks están pensados para Google Colab con la carpeta del proyecto en Google Drive (`MyDrive/Predicción laguna torca/data`). Si no encuentran Drive, buscan `data/` en la carpeta actual, en `../data` o en `../../data`, así que también funcionan clonando este repositorio.

La serie satelital y el clima ya extraído están en `data/processed/`, por lo que los notebooks corren sin volver a llamar a Earth Engine ni a los archivos grandes de CR2MET.

## Datos que no están en el repositorio

Estos archivos superan el límite de 100 MB de GitHub y quedan fuera (ver `.gitignore`):

| Archivo | Tamaño | Origen |
|---|---|---|
| `data/raw/CR2MET_pr_v2.0_mon_1979_2019_005deg.nc` | 346 MB | CR2MET v2.0, producto grillado mensual del (CR)2 — https://www.cr2.cl |
| `data/raw/CR2MET_tmax_v2.0_mon_1979_2019_005deg.nc` | 346 MB | ídem |
| `data/raw/CR2MET_tmin_v2.0_mon_1979_2019_005deg.nc` | 346 MB | ídem |

Solo hacen falta para volver a extraer el clima de otra ubicación. Para Torca y Aculeo ya están extraídos en `data/processed/clima_cr2met_laguna_*_mensual_1979_2019.csv`.
