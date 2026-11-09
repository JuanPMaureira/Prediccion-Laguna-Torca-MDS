# ¿Se está secando la Laguna Torca?

Proyecto de Consultoría Estadística · Magíster en Data Science, Universidad Católica del Maule · Juan Pablo Maureira

Análisis de 40 años de imágenes Landsat de la superficie de agua del humedal Laguna Torca (Vichuquén, Región del Maule) y de su relación con la lluvia y la temperatura (CR2MET). El mismo análisis se repite en dos lagunas de comparación: la **Laguna de Aculeo**, que se secó con la megasequía, y la **Laguna Miscanti**, laguna altiplánica sin intervención humana conocida que sirve de control natural.

> El cliente (Comité Ciudadano del Humedal Laguna Torca) y las reuniones son **simulados** para el curso. Los datos y los resultados son reales.

## Resultados principales

| Pregunta | Respuesta |
|---|---|
| ¿Se está secando la laguna? | No hay señales. Separando el cambio de satélite, la superficie cambia −0,07 ha/año (IC 95 %: −0,28 a +0,13) en torno a 129 ha. El aparente crecimiento desde 2013 se debe sobre todo al cambio de Landsat 5-7 a Landsat 8-9. |
| ¿La sequía la está achicando? | La lluvia explica como mucho cerca del 4 % de las diferencias entre años, y con 31 años solo se podía detectar una correlación de 0,48 o más. |
| ¿Cuánta agua tendrá en 100 años? | No se puede responder de forma confiable: el clima no mejora ni el pronóstico del año siguiente, y faltan extracciones, aguas subterráneas y nivel medido. Las curvas a 2095 son solo un análisis de sensibilidad. |
| ¿Y en una laguna sin manejo? | En Miscanti el clima tampoco mejora el pronóstico, aunque el método sí detecta una baja lenta (−0,57 ha/año según JRC, 1987-2021). La poca señal climática de la Torca no se explica solo por su manejo. |

Los informes completos están en [`outputs/reports/`](outputs/reports):

- [`Informe_final_Laguna_Torca.pdf`](outputs/reports/Informe_final_Laguna_Torca.pdf) (y `.docx`): informe final. Incluye como anexos la bitácora de reuniones (Hito 2), el avance de datos (Hito 3), la verificación de supuestos y el código del análisis.
- [`Informe_final_Laguna_Miscanti.pdf`](outputs/reports/Informe_final_Laguna_Miscanti.pdf) (y `.docx`): informe complementario de la laguna de control.

## Estructura

```
notebooks/
  Laguna_Torca_v2_CR2MET.ipynb    análisis principal (Torca), ejecutado el 8-nov-2026 con el clima corregido
  Laguna_Aculeo_v1.ipynb          mismo análisis en la Laguna de Aculeo (contraste)
  Laguna_Miscanti_v1.ipynb        mismo análisis en la Laguna Miscanti (control natural)
scripts/
  analisis_consultoria.py         cifras, tablas y figuras del informe final (copia del Anexo E.2)
  preparar_carpetas.py            arma las carpetas de trabajo que espera el script anterior
data/
  raw/         proyecciones climáticas del CR2 (CSV) para Torca y Miscanti, y descargas ERA5-Land (.nc)
  processed/   series satelitales, clima extraído y resultados que generan los notebooks
outputs/reports/   informes finales (.pdf y .docx)
docs/              enunciado del proyecto
descargar_era5_aculeo.py          descarga ERA5-Land mensual para Aculeo (Copernicus CDS)
```

Los archivos de `data/processed` llevan el nombre de su laguna: sin sufijo o `_v2` = Torca, `_aculeo_v1` = Aculeo y `_miscanti_v1` o `miscanti` = Miscanti.

## Cómo reproducir

```bash
pip install -r requirements.txt
```

**Notebooks.** Están pensados para Google Colab con la carpeta del proyecto en Google Drive (`MyDrive/Predicción laguna torca/data`). Si no encuentran Drive, buscan `data/` en la carpeta actual, en `../data` o en `../../data`, así que también corren clonando este repositorio. Las series satelitales y el clima ya extraídos están en `data/processed/` como cachés, así que no hace falta volver a llamar a Earth Engine ni a los archivos grandes de CR2MET. Para volver a descargar desde Earth Engine se necesita un proyecto propio de Google Cloud (los notebooks usan `master-chess-412203`).

La sección 13 de `Laguna_Miscanti_v1.ipynb` lee las cifras de Torca desde `Claude outputs/corrida_corregida_2026-11-08/`, la carpeta local donde se guardó la corrida corregida. En este repositorio esos mismos archivos están en `data/processed/`. Si la carpeta no existe, la columna de Torca de esa tabla queda vacía; la tabla ya calculada está en `data/processed/comparacion_torca_vs_miscanti_v1.csv`.

**Cifras del informe final.** Desde la raíz del repositorio:

```bash
python scripts/preparar_carpetas.py
python scripts/analisis_consultoria.py
```

El análisis tarda unos 2 minutos y deja `salidas/resultados.json`, 24 tablas y 11 figuras (`salidas/` y `figs/` no se suben). Se comprobó el 8-nov-2026 con los datos de este repositorio. Reproduce, por ejemplo, la tendencia de −0,07 ha/año (IC 95 %: −0,28 a +0,13), la correlación mínima detectable de 0,48 y la pendiente de Miscanti de −0,57 ha/año.

## Corrección del 8 de noviembre de 2026

En la versión anterior de este repositorio (19-oct), `base_mensual_torca_v2.csv` y los resultados de Torca usaban por error el clima de la Laguna de Aculeo. El patrón de búsqueda `clima_cr2met*mensual*.csv` tomaba el primer archivo de clima que encontraba. El notebook de Torca ahora busca `clima_cr2met_laguna_torca*mensual*.csv`, se volvió a ejecutar, y todos los CSV de Torca en `data/processed/` vienen de esa ejecución corregida. Se retiraron el informe LaTeX y las presentaciones del 19-oct, que usaban esas cifras; el informe final los reemplaza.

## Datos que no están en el repositorio

Estos archivos superan el límite de 100 MB de GitHub (ver `.gitignore`):

| Archivo | Tamaño | Origen |
|---|---|---|
| `data/raw/CR2MET_pr_v2.0_mon_1979_2019_005deg.nc` | 346 MB | CR2MET v2.0, producto grillado mensual del CR2: https://www.cr2.cl |
| `data/raw/CR2MET_tmax_v2.0_mon_1979_2019_005deg.nc` | 346 MB | ídem |
| `data/raw/CR2MET_tmin_v2.0_mon_1979_2019_005deg.nc` | 346 MB | ídem |

Solo hacen falta para volver a extraer el clima de otra ubicación. Para las tres lagunas ya está extraído en `data/processed/clima_cr2met_laguna_*_mensual_1979_2019.csv`.

## Fuentes de datos

- Imágenes Landsat 5, 7, 8 y 9 (USGS), procesadas en Google Earth Engine (Gorelick et al., 2017).
- JRC Global Surface Water, Monthly History v1.4 y v1.5 (Pekel et al., 2016): segunda fuente de superficie para Miscanti.
- CR2MET v2.0 y Explorador de Simulaciones Climáticas (CMIP5, RCP 2.6 y 8.5) del Centro de Ciencia del Clima y la Resiliencia (CR2).
- ERA5-Land (Copernicus): clima de 2020-2024 para la prueba independiente de Miscanti, ajustado a CR2MET.

El uso de inteligencia artificial en el proyecto está declarado en el Anexo E del informe final.
