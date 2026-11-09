# -*- coding: utf-8 -*-
"""
Proyecto de consultoría estadística: superficie de agua de la Laguna Torca (Maule, Chile)
con la Laguna de Aculeo como caso de contraste.

Los datos son reales: salen de los CSV de data/processed generados por
Laguna_Torca_v2_CR2MET.ipynb (secciones 0-12) y Laguna_Aculeo_v1.ipynb. Este script NO modifica
los originales: trabaja sobre una copia en ./datos y escribe resultados en ./salidas y ./figs.

Orden: 0 configuración · 1 respaldo y hashes · 2 lectura y formato · 3 revisión de errores ·
4 reproducción de la curación · 5 faltantes y cobertura · 6 atípicos (1,5·IQR) · 7 EDA ·
8 H1 tendencia · 9 H3 efecto del sensor · 10 H2 asociación con el clima · 11 autocorrelación y
potencia · 12 validación predictiva · 13 extrapolación de temperatura · 14 Aculeo · 15 MICE · 16 exportación.
"""
import hashlib
import io
import json
import re
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from scipy import stats
from matplotlib.ticker import FuncFormatter
import statsmodels.api as sm
import statsmodels.formula.api as smf
from statsmodels.stats.anova import anova_lm
from statsmodels.stats.multicomp import pairwise_tukeyhsd
from statsmodels.stats.stattools import durbin_watson
from statsmodels.tsa.stattools import acf
from statsmodels.graphics.tsaplots import plot_acf

warnings.filterwarnings("ignore")

# ============================== 0. Configuración ==============================
BASE = Path(__file__).resolve().parents[1]
RESP = BASE / "respaldo" / "data"          # copia intacta del original (solo lectura)
DATA = BASE / "datos" / "processed"        # copia de trabajo
RAW = BASE / "datos" / "raw"
# Resultados del notebook Laguna_Torca_v2_CR2MET.ipynb (ejecución usada en este informe, con el clima CR2MET de Torca).
CORR = BASE / "ejecucion_torca" / "data" / "processed"
OUT = BASE / "salidas"
FIG = BASE / "figs"
OUT.mkdir(exist_ok=True)
FIG.mkdir(exist_ok=True)

SEMILLA = 42
RNG = np.random.default_rng(SEMILLA)
COL = "area_hectareas"
QUIEBRE_SENSOR = pd.Timestamp("2012-06-01")
FALLA_SLC_L7 = pd.Timestamp("2003-05-31")
AREA_MIN_HA = 44.0
FECHAS_AUDITADAS = pd.to_datetime(["2011-10-07", "2018-05-12", "2018-06-13",
                                   "2019-05-22", "2019-06-07", "2018-07-22"])
ALFA, POTENCIA = 0.05, 0.80
BLOQUE_ANUAL = 3      # largo de bloque (años) para el bootstrap por bloques (≈ n^(1/3) con n ≈ 37)
N_BOOT = 5000

# Paleta (categórica validada, en orden fijo): azul = L5/L7, naranjo = L8/L9, aqua = tercer grupo
C1, C2, C3, GRIS, TINTA = "#2a78d6", "#eb6834", "#1baf7a", "#8a8984", "#2b2b2a"
plt.rcParams.update({"figure.dpi": 150, "savefig.dpi": 200, "font.size": 9, "axes.spines.top": False,
                     "axes.spines.right": False, "axes.grid": True, "grid.color": "#e6e5e1", "grid.linewidth": 0.6,
                     "axes.edgecolor": "#9a9993", "axes.labelcolor": TINTA, "xtick.color": TINTA,
                     "ytick.color": TINTA, "legend.frameon": False})

R = {}            # todos los resultados numéricos que citan los documentos
TABLAS = {}       # tablas que se exportan a CSV


def _coma(texto):
    """Decimales con coma (español de Chile) en todo texto de la figura."""
    return re.sub(r"(\d)\.(\d)", r"\1,\2", texto).replace("RCP 2,6", "RCP 2.6").replace("RCP 8,5", "RCP 8.5")


def guardar_fig(fig, nombre):
    for ax in fig.axes:
        for eje in (ax.xaxis, ax.yaxis):
            if type(eje.get_major_formatter()).__name__ == "ScalarFormatter":
                eje.set_major_formatter(FuncFormatter(lambda v, _p: _coma(f"{v:g}").replace("-", "−")))
        ax.set_title(_coma(ax.get_title()))
        ax.set_xlabel(_coma(ax.get_xlabel())); ax.set_ylabel(_coma(ax.get_ylabel()))
        for t in ax.texts:
            t.set_text(_coma(t.get_text()))
        if ax.get_legend():
            for t in ax.get_legend().get_texts():
                t.set_text(_coma(t.get_text()))
    fig.tight_layout()
    fig.savefig(FIG / nombre, bbox_inches="tight")
    plt.close(fig)


def estacion(m):
    """Estaciones del hemisferio sur."""
    return np.select([np.isin(m, [12, 1, 2]), np.isin(m, [3, 4, 5]), np.isin(m, [6, 7, 8])],
                     ["Verano (dic-feb)", "Otoño (mar-may)", "Invierno (jun-ago)"], "Primavera (sep-nov)")


ORDEN_EST = ["Verano (dic-feb)", "Otoño (mar-may)", "Invierno (jun-ago)", "Primavera (sep-nov)"]


# ============================== 1. Respaldo y hashes ==============================
def sha256(ruta):
    h = hashlib.sha256()
    with open(ruta, "rb") as f:
        for bloque in iter(lambda: f.read(1 << 20), b""):
            h.update(bloque)
    return h.hexdigest()


filas = []
for ruta in sorted((DATA).glob("*.csv")):
    orig = RESP / "processed" / ruta.name
    filas.append((ruta.name, ruta.stat().st_size, sha256(orig), sha256(ruta) == sha256(orig)))
for ruta in sorted(RAW.glob("*.csv")):
    orig = RESP / "raw" / ruta.name
    filas.append(("raw/" + ruta.name, ruta.stat().st_size, sha256(orig), sha256(ruta) == sha256(orig)))
for ruta in sorted((BASE / "datos" / "miscanti").glob("*.csv")):      # Laguna Miscanti (control natural), copiados el 8-nov
    orig = BASE / "respaldo" / "miscanti" / ruta.name
    filas.append(("miscanti/" + ruta.name, ruta.stat().st_size, sha256(orig), sha256(ruta) == sha256(orig)))
hashes = pd.DataFrame(filas, columns=["archivo", "bytes", "sha256_original", "copia_identica"])
TABLAS["t01_hashes"] = hashes
R["respaldo"] = {"n_archivos": len(hashes), "todas_identicas": bool(hashes["copia_identica"].all()),
                 "fecha_respaldo_utc": (OUT / "fecha_respaldo.txt").read_text().strip()}

# ============================== 2. Lectura y formato ==============================
formato = []
for ruta in sorted(DATA.glob("*.csv")) + sorted(RAW.glob("proyeccin*.csv")):
    crudo = ruta.read_bytes()[:4000]
    bom = crudo.startswith(b"\xef\xbb\xbf")
    texto = crudo.decode("utf-8-sig", errors="replace")
    primera = texto.splitlines()[0]
    sep = ";" if primera.count(";") > primera.count(",") else ","
    try:
        crudo.decode("ascii")
        cod = "ASCII"
    except UnicodeDecodeError:
        cod = "UTF-8" + (" con BOM" if bom else "")
    cols = [c.strip('"') for c in primera.split(sep)]
    sin_nombre = sum(1 for c in cols if c == "")
    formato.append((ruta.name, cod, sep, "coma" if (sep == ";") else "punto", len(cols), sin_nombre))
TABLAS["t02_formato"] = pd.DataFrame(formato, columns=["archivo", "codificacion", "separador", "decimal",
                                                       "n_columnas", "columnas_sin_nombre"])

# ---- Lecturas principales
bruta = pd.read_csv(DATA / "serie_laguna_torca_bruta.csv", parse_dates=["fecha"])
base_m = pd.read_csv(DATA / "base_mensual_torca_v2.csv", parse_dates=["fecha"], index_col="fecha")
clima = pd.read_csv(DATA / "clima_cr2met_laguna_torca_mensual_1979_2019.csv", parse_dates=["fecha"], index_col="fecha")
base_m = pd.read_csv(CORR / "base_mensual_torca_v2.csv", parse_dates=["fecha"], index_col="fecha")
anual12 = pd.read_csv(CORR / "anual12_base_anual.csv", index_col="fecha")
var_anual = pd.read_csv(CORR / "variacion_anual_v2.csv", index_col="fecha")
pares = pd.read_csv(CORR / "anual12_pares_L5_L7.csv", parse_dates=["fecha"]).drop(columns=["Unnamed: 1"])
corr12 = pd.read_csv(CORR / "anual12_correlaciones.csv")
val12 = pd.read_csv(CORR / "anual12_validacion.csv")
saltos = pd.read_csv(CORR / "anual12_saltos_por_fuente.csv")
corr_m = pd.read_csv(CORR / "correlaciones_v2.csv")
cv_m = pd.read_csv(CORR / "cv_nivel_area_v2.csv", index_col=0)
cv_sar = pd.read_csv(CORR / "cv_sarimax_v2.csv", index_col=0)
corr_ac = pd.read_csv(DATA / "correlaciones_aculeo_v1.csv", index_col="variable")
cal_ac = pd.read_csv(DATA / "calibracion_sensores_aculeo_v1.csv")
cv_ac = pd.read_csv(DATA / "cv_nivel_area_aculeo_v1.csv", index_col=0)
var_ac = pd.read_csv(DATA / "variacion_anual_aculeo_v1.csv")
circ = pd.read_csv(CORR / "diagnostico_circularidad_mice_v2.csv")
mice_val = pd.read_csv(CORR / "modelo_anual_mice_validacion_v2.csv")
comp_esc = pd.read_csv(CORR / "comparacion_diferencia_escenarios_v2.csv")
proy12 = pd.read_csv(CORR / "anual12_resumen_proyecciones.csv")
proy_m = pd.read_csv(CORR / "resumen_proyecciones_v2.csv")
proy_ac = pd.read_csv(DATA / "resumen_proyecciones_aculeo_v1.csv")

R["formato"] = {
    "n_escenas_brutas": int(len(bruta)), "fecha_min": str(bruta["fecha"].min().date()),
    "fecha_max": str(bruta["fecha"].max().date()),
    "escenas_por_sensor": bruta["satelite"].value_counts().to_dict(),
    "fechas_no_parseadas": int(bruta["fecha"].isna().sum()),
    "pares_columna_indice_sin_nombre": True,
}

# ============================== 3. Revisión de errores ==============================
err = {}
# 3a. Variable derivada: área en ha = área en m² / 10 000
err["max_dif_ha_vs_m2"] = float((bruta[COL] - bruta["area_m2"] / 1e4).abs().max())
# 3b. Rango físico: el área no puede ser negativa ni superar el polígono de estudio
COORD = np.array([[-72.0410805, -34.7577721], [-72.0367228, -34.7582617], [-72.0387713, -34.7682978],
                  [-72.0319554, -34.7684813], [-72.0291981, -34.7765296], [-72.0353063, -34.7784569],
                  [-72.0422712, -34.777172], [-72.0449901, -34.7868999], [-72.0516198, -34.7914269],
                  [-72.0545995, -34.786961], [-72.0502045, -34.7824032], [-72.0491989, -34.7747858],
                  [-72.0469641, -34.7673474], [-72.0410805, -34.7577721]])
lat0 = np.radians(COORD[:, 1].mean())
x = np.radians(COORD[:, 0]) * 6371000 * np.cos(lat0)
y = np.radians(COORD[:, 1]) * 6371000
poligono_ha = 0.5 * abs(np.dot(x[:-1], y[1:]) - np.dot(x[1:], y[:-1])) / 1e4
err["poligono_ha"] = round(float(poligono_ha), 1)
err["n_area_negativa"] = int((bruta[COL] < 0).sum())
err["n_area_cero"] = int((bruta[COL] == 0).sum())
err["n_area_sobre_poligono"] = int((bruta[COL] > poligono_ha).sum())
err["area_min"], err["area_max"] = float(bruta[COL].min()), float(bruta[COL].max())
err["n_bajo_44ha"] = int((bruta[COL] < AREA_MIN_HA).sum())
# 3c. Categorías del sensor
err["niveles_sensor"] = sorted(bruta["satelite"].unique().tolist())
# 3d. Duplicados por fecha y sensor, y por fecha
err["dup_fecha_sensor"] = int(bruta.duplicated(["fecha", "satelite"]).sum())
dup_f = bruta[bruta.duplicated("fecha", keep=False)].sort_values("fecha")
err["dup_fecha_distinto_sensor"] = int(bruta.duplicated("fecha").sum())
dupt = dup_f.pivot_table(index="fecha", columns="satelite", values=COL).reset_index()
TABLAS["t03_duplicados_fecha"] = dupt
err["dup_pares_sensores"] = dup_f.groupby("fecha")["satelite"].apply(lambda s: "+".join(sorted(s))).value_counts().to_dict()
# 3e. Escenas Landsat 7 por período (SLC-off)
l7 = bruta[bruta["satelite"] == "LANDSAT_7"]
err["l7_antes_slc"] = int((l7["fecha"] <= FALLA_SLC_L7).sum())
err["l7_slc_off"] = int((l7["fecha"] > FALLA_SLC_L7).sum())
# 3f. Clima CR2MET
err["clima_meses"] = int(len(clima))
err["clima_faltantes"] = int(clima.isna().sum().sum())
err["clima_precip_negativa"] = int((clima["precip_mm"] < 0).sum())
err["clima_tmax_menor_tmin"] = int((clima["tmax_c"] <= clima["tmin_c"]).sum())
err["clima_rangos"] = {c: [round(float(clima[c].min()), 2), round(float(clima[c].max()), 2)] for c in clima.columns}
lluvia_anual = clima["precip_mm"].resample("YS").sum()
err["lluvia_anual_min_max"] = [round(float(lluvia_anual.min()), 0), round(float(lluvia_anual.max()), 0)]
err["lluvia_anual_media_1985_2019"] = round(float(lluvia_anual.loc["1985":"2019"].mean()), 0)
# 3g. Variables derivadas del archivo mensual: p12 debe ser la suma móvil de 12 meses de la lluvia
p12_rec = clima["precip_mm"].rolling(12).sum().reindex(base_m.index)
err["max_dif_p12"] = float((p12_rec - base_m["p12"]).abs().max())
epoca_rec = (base_m.index > QUIEBRE_SENSOR).astype(int)
err["epoca_inconsistente"] = int((epoca_rec != base_m["epoca"]).sum())
# 3h. Rangos de la tabla anual
err["anual12_min_mayor_que_media"] = int((anual12["min_verano"] > anual12["media_anual"] + 1e-9).sum())
# 3i. La lluvia de la base mensual coincide con el CSV de CR2MET de Torca
j_c = base_m[["precip"]].join(clima[["precip_mm"]]).dropna()
err["clima_base_vs_cr2met_maxdif_mm"] = float((j_c["precip"] - j_c["precip_mm"]).abs().max())
R["errores"] = err

# ============================== 4. Reproducción de la curación ==============================
def curar_serie(df):
    """Las 5 reglas de la sección 2 del notebook v2 (se reproducen para registrar cada dato anulado)."""
    s = df.set_index("fecha").copy()
    s["motivo"] = ""

    def anular(mascara, nombre):
        m = np.asarray(mascara) & s[COL].notna().values
        s.loc[m, "motivo"] = nombre
        s.loc[m, COL] = np.nan

    anular(s["satelite"].str.contains("LANDSAT_7") & (s.index > FALLA_SLC_L7), "1. Landsat 7 con falla SLC (después de 2003-05-31)")
    anular(s[COL] < AREA_MIN_HA, "2. Área < 44 ha (nube total)")
    mediana = s[COL].rolling(window=7, center=True, min_periods=3).median()
    anular(s[COL] < mediana * 0.85, "3. Caída > 15 % frente a la mediana móvil (nube parcial)")
    v = s[s[COL].notna()]
    f = v.index.to_series()
    dp, dn = f.diff().dt.days, (-f.diff(-1)).dt.days
    vp, vn = v[COL].pct_change(fill_method=None), v[COL].pct_change(periods=-1, fill_method=None)
    valle = (vp < -0.15) & (vn < -0.15) & (dp <= 45) & (dn <= 45)
    anular(s.index.isin(v.index[valle.values]), "4. Valle abrupto de corta duración")
    anular(s.index.isin(FECHAS_AUDITADAS), "5. Fecha auditada visualmente en GEE (nube)")
    return s


curada = curar_serie(bruta)
registro = curada[curada["motivo"] != ""].reset_index()[["fecha", "satelite", "motivo"]]
registro = registro.merge(bruta[["fecha", "satelite", COL]], on=["fecha", "satelite"], how="left")
TABLAS["t04_registro_datos_anulados"] = registro
resumen_cur = pd.crosstab(registro["motivo"], registro["satelite"].str.replace("LANDSAT_", "L"), margins=True,
                          margins_name="Total")
TABLAS["t05_curacion_por_sensor"] = resumen_cur.reset_index()
validas = curada[curada[COL].notna()].copy()
validas["sensor"] = validas["satelite"].str.replace("LANDSAT_", "L")
R["curacion"] = {"n_validas": int(len(validas)), "n_anuladas": int(len(registro)),
                 "por_motivo": registro["motivo"].value_counts().sort_index().to_dict(),
                 "validas_por_sensor": validas["sensor"].value_counts().to_dict()}
mensual = curada[COL].resample("MS").mean()
comp = pd.concat([mensual.rename("mio"), base_m[COL].rename("csv")], axis=1)
R["curacion"]["meses_con_dato_mio"] = int(mensual.notna().sum())
R["curacion"]["meses_con_dato_csv"] = int(base_m[COL].notna().sum())
R["curacion"]["max_dif_mensual_vs_csv"] = float((comp["mio"] - comp["csv"]).abs().max())
R["curacion"]["meses_discrepantes"] = int((comp["mio"].notna() != comp["csv"].notna()).sum())

# ============================== 5. Faltantes y cobertura ==============================
meses = pd.DataFrame({"area": base_m[COL]})
meses["dato"] = meses["area"].notna()
meses["anio"] = meses.index.year
meses["estacion"] = estacion(meses.index.month)
meses["periodo"] = np.select([meses.index <= FALLA_SLC_L7, meses.index < QUIEBRE_SENSOR],
                             ["1985-2003 (L5 + L7)", "2003-2012 (solo L5)"], "2012-2024 (L8/L9)")
R["faltantes"] = {"meses_totales": int(len(meses)), "meses_con_dato": int(meses["dato"].sum()),
                  "pct_con_dato": round(100 * meses["dato"].mean(), 1),
                  "meses_con_dato_2004_2023": int(meses.loc["2004":"2023", "dato"].sum()),
                  "meses_2004_2023": int(len(meses.loc["2004":"2023"])),
                  "anios_sin_dato": meses.groupby("anio")["dato"].sum().loc[lambda s: s == 0].index.tolist()}
por_anio = meses.groupby("anio")["dato"].agg(meses_con_dato="sum", meses="count")
por_anio["pct_faltante"] = (100 * (1 - por_anio["meses_con_dato"] / por_anio["meses"])).round(1)
TABLAS["t06_faltantes_por_anio"] = por_anio.reset_index()

# Proporción de escenas descartadas por sensor (dato bruto → válido)
por_sensor = pd.DataFrame({"escenas_brutas": bruta["satelite"].str.replace("LANDSAT_", "L").value_counts(),
                           "escenas_validas": validas["sensor"].value_counts()}).fillna(0).astype(int)
por_sensor["pct_descartado"] = (100 * (1 - por_sensor["escenas_validas"] / por_sensor["escenas_brutas"])).round(1)
TABLAS["t07_escenas_por_sensor"] = por_sensor.reset_index(names="sensor")


def tabla_contingencia(fila, col, nombre):
    """Chi-cuadrado solo si ≤ 20 % de las celdas tiene esperado < 5; si no, Fisher (2x2) o se combinan niveles."""
    t = pd.crosstab(fila, col)
    chi2, p, gl, esp = stats.chi2_contingency(t, correction=False)
    pct_bajo5 = 100 * (esp < 5).mean()
    res = {"tabla": t.to_dict(), "chi2": round(float(chi2), 2), "gl": int(gl), "p": float(p),
           "pct_celdas_esperado_menor_5": round(float(pct_bajo5), 1), "n": int(t.values.sum()),
           "esperado_min": round(float(esp.min()), 1)}
    if pct_bajo5 > 20 and t.shape == (2, 2):
        res["fisher_p"] = float(stats.fisher_exact(t)[1])
    # V de Cramér como tamaño del efecto
    res["v_cramer"] = round(float(np.sqrt(chi2 / (t.values.sum() * (min(t.shape) - 1)))), 3)
    pct = (100 * t.div(t.sum(axis=1), axis=0)).round(1)
    TABLAS[nombre] = pd.concat([t.add_prefix("n_"), pct.add_prefix("pct_")], axis=1).reset_index()
    return res


cob = meses.copy()
cob["Dato"] = np.where(cob["dato"], "Con dato", "Sin dato")
R["cobertura_estacion"] = tabla_contingencia(pd.Categorical(cob["estacion"], ORDEN_EST), cob["Dato"], "t08_cobertura_estacion")
R["cobertura_periodo"] = tabla_contingencia(cob["periodo"], cob["Dato"], "t09_cobertura_periodo")

# ============================== 6. Atípicos (1,5·IQR) ==============================
def limites_iqr(s):
    q1, q3 = s.quantile([0.25, 0.75])
    return q1 - 1.5 * (q3 - q1), q3 + 1.5 * (q3 - q1)


# 6a. Escenas brutas (todas juntas): cuántas marcaría la regla y cuántas ya quitó la curación
li, ls_ = limites_iqr(bruta[COL])
marca_bruta = bruta[(bruta[COL] < li) | (bruta[COL] > ls_)]
R["atipicos"] = {"bruta_lim": [round(li, 1), round(ls_, 1)], "bruta_n_marcadas": int(len(marca_bruta))}
marcadas_set = set(zip(marca_bruta["fecha"], marca_bruta["satelite"]))
anuladas_set = set(zip(registro["fecha"], registro["satelite"]))
R["atipicos"]["bruta_marcadas_ya_anuladas"] = int(len(marcadas_set & anuladas_set))
# 6b. Meses curados, separado por época (el nivel cambia con el sensor)
filas = []
for ep, g in base_m[COL].dropna().groupby(base_m["epoca"].reindex(base_m[COL].dropna().index)):
    li, ls_ = limites_iqr(g)
    for f, v in g[(g < li) | (g > ls_)].items():
        filas.append((f.date(), "L5/L7" if ep == 0 else "L8/L9", round(v, 1), round(li, 1), round(ls_, 1),
                      round(base_m.loc[f, "area_anom"], 1)))
at = pd.DataFrame(filas, columns=["mes", "epoca", "area_ha", "lim_inf", "lim_sup", "anomalia_ha"])
TABLAS["t10_atipicos_mensuales_iqr"] = at
R["atipicos"]["mensual_n"] = int(len(at))
R["atipicos"]["mensual_lista"] = at.to_dict("records")
an = base_m["area_anom"].dropna()
li, ls_ = limites_iqr(an)
R["atipicos"]["anomalia_lim"] = [round(li, 1), round(ls_, 1)]
R["atipicos"]["anomalia_n"] = int(((an < li) | (an > ls_)).sum())
R["atipicos"]["anomalia_extremas"] = {str(k.date()): round(v, 1) for k, v in an.sort_values().head(3).items()}

# ============================== 7. EDA ==============================
area_m = base_m[COL].dropna()
epoca_m = np.where(area_m.index > QUIEBRE_SENSOR, "L8/L9", "L5/L7")
desc = area_m.groupby(epoca_m).describe().round(1)
desc.loc["Total"] = area_m.describe().round(1)
TABLAS["t11_descriptivos_mensual"] = desc.reset_index(names="epoca")
R["eda"] = {"descriptivos": desc.to_dict("index"),
            "media_2004_2023": round(float(base_m.loc["2004":"2023", COL].mean()), 1),
            "min_2004_2023": round(float(base_m.loc["2004":"2023", COL].min()), 1),
            "max_2004_2023": round(float(base_m.loc["2004":"2023", COL].max()), 1)}
# Ancho de clase de Freedman-Diaconis y de Scott
n = len(area_m)
iqr = area_m.quantile(.75) - area_m.quantile(.25)
R["eda"]["h_fd"] = round(float(2 * iqr * n ** (-1 / 3)), 2)
R["eda"]["h_scott"] = round(float(3.49 * area_m.std() * n ** (-1 / 3)), 2)
R["eda"]["n_meses"] = int(n)
# Frecuencias por sensor y por estación (con n y %)
frec_sensor = validas["sensor"].value_counts().rename("n").to_frame()
frec_sensor["%"] = (100 * frec_sensor["n"] / frec_sensor["n"].sum()).round(1)
TABLAS["t12_frecuencia_sensor"] = frec_sensor.reset_index(names="sensor")
validas["estacion"] = estacion(validas.index.month)
frec_est = validas["estacion"].value_counts().reindex(ORDEN_EST).rename("n").to_frame()
frec_est["%"] = (100 * frec_est["n"] / frec_est["n"].sum()).round(1)
TABLAS["t13_frecuencia_estacion"] = frec_est.reset_index(names="estacion")
# Estacionalidad (sin el nivel de época): Kruskal-Wallis entre meses
adj = area_m - area_m.groupby(epoca_m).transform("mean")
kw = stats.kruskal(*[g.values for _, g in adj.groupby(adj.index.month)])
R["eda"]["kw_meses_H"], R["eda"]["kw_meses_p"] = round(float(kw.statistic), 1), float(kw.pvalue)
est_media = adj.groupby(estacion(adj.index.month)).mean().reindex(ORDEN_EST).round(1)
R["eda"]["media_por_estacion_sin_epoca"] = est_media.to_dict()

# ---- Figura 1: serie mensual observada
fig, ax = plt.subplots(figsize=(7.2, 3.0))
for ep, c in (("L5/L7", C1), ("L8/L9", C2)):
    s = area_m[epoca_m == ep]
    ax.plot(s.index, s.values, "o", ms=2.6, color=c, label=f"Landsat {ep[1:].replace('/L', ' y ')} (n = {len(s)} meses)")
ax.axvline(QUIEBRE_SENSOR, color=GRIS, ls="--", lw=1)
ax.text(QUIEBRE_SENSOR, ax.get_ylim()[1] - 6, "  cambio de satélite (2012)", color=TINTA, fontsize=7.5)
ax.set(xlabel="Año", ylabel="Superficie de agua (ha)", title="Superficie mensual observada de la Laguna Torca, 1985-2024")
ax.legend(loc="lower left", fontsize=7.5)
guardar_fig(fig, "f01_serie_mensual.png")

# ---- Figura 2: histograma (Freedman-Diaconis) por época
fig, ax = plt.subplots(figsize=(7.2, 2.8))
bins = np.arange(np.floor(area_m.min()), area_m.max() + R["eda"]["h_fd"], R["eda"]["h_fd"])
for ep, c in (("L5/L7", C1), ("L8/L9", C2)):
    s = area_m[epoca_m == ep]
    ax.hist(s, bins=bins, color=c, alpha=0.75, weights=np.ones(len(s)) * 100 / len(s),
            label=f"{ep} (n = {len(s)})", edgecolor="white", linewidth=0.6)
ax.set(xlabel="Superficie mensual (ha)", ylabel="% de los meses de cada época",
       title=f"Distribución de la superficie mensual (ancho de clase Freedman-Diaconis = {R['eda']['h_fd']} ha)")
ax.legend(fontsize=7.5)
guardar_fig(fig, "f02_histograma.png")

# ---- Figura 3: boxplot por sensor y por estación (escenas válidas)
fig, axes = plt.subplots(1, 2, figsize=(7.2, 2.9), sharey=True)
orden_s = ["L5", "L7", "L8", "L9"]
datos_s = [validas.loc[validas["sensor"] == s, COL].values for s in orden_s]
b = axes[0].boxplot(datos_s, tick_labels=[f"{s}\n(n = {len(d)})" for s, d in zip(orden_s, datos_s)], patch_artist=True,
                    widths=0.55, medianprops=dict(color=TINTA), flierprops=dict(ms=3, markeredgecolor=GRIS))
for p, c in zip(b["boxes"], [C1, C1, C2, C2]):
    p.set_facecolor(c); p.set_alpha(0.6)
axes[0].set(ylabel="Superficie por escena (ha)", title="Por satélite")
datos_e = [validas.loc[validas["estacion"] == e, COL].values for e in ORDEN_EST]
b = axes[1].boxplot(datos_e, tick_labels=[f"{e.split(' ')[0]}\n(n = {len(d)})" for e, d in zip(ORDEN_EST, datos_e)],
                    patch_artist=True, widths=0.55, medianprops=dict(color=TINTA), flierprops=dict(ms=3, markeredgecolor=GRIS))
for p in b["boxes"]:
    p.set_facecolor(C3); p.set_alpha(0.5)
axes[1].set(title="Por estación del año")
guardar_fig(fig, "f03_boxplots.png")

# ---- Figura 4: cobertura (% de meses con dato) por año
fig, ax = plt.subplots(figsize=(7.2, 2.6))
pct = 100 * por_anio["meses_con_dato"] / por_anio["meses"]
cols_ = [C1 if a < 2012 else (GRIS if a == 2012 else C2) for a in pct.index]
ax.bar(pct.index, pct.values, color=cols_, width=0.8)
ax.set(xlabel="Año", ylabel="% de meses con dato", ylim=(0, 105),
       title="Meses con al menos una imagen válida, por año (2012: ninguna)")
guardar_fig(fig, "f04_cobertura.png")

# ============================== 8. H1: tendencia de la superficie anual ==============================
an12 = anual12.copy()
an12 = an12[an12["media_anual"].notna()]
ya = an12["media_anual"]
R["h1"] = {"n_anios": int(len(ya)), "anio_min": int(ya.index.min()), "anio_max": int(ya.index.max()),
           "anios_faltantes": sorted(set(range(ya.index.min(), ya.index.max() + 1)) - set(ya.index))}
d_h1 = pd.DataFrame({"y": ya.values, "anio": ya.index.values, "epoca": an12["epoca"].values})
m_sin = smf.ols("y ~ anio", d_h1).fit()
m_con = smf.ols("y ~ anio + epoca", d_h1).fit()


def resumen_modelo(m, var="anio"):
    ci = m.conf_int().loc[var]
    return {"pendiente": float(m.params[var]), "ic_inf": float(ci[0]), "ic_sup": float(ci[1]), "p": float(m.pvalues[var]),
            "r2": float(m.rsquared), "r2_aj": float(m.rsquared_adj), "aic": float(m.aic),
            "shapiro_p": float(stats.shapiro(m.resid).pvalue), "dw": float(durbin_watson(m.resid)),
            "r1_resid": float(pd.Series(m.resid).autocorr(1)),
            "cook_max": float(m.get_influence().cooks_distance[0].max()),
            "n_resid_std_fuera_2": int((np.abs(m.get_influence().resid_studentized_internal) > 2).sum())}


R["h1"]["sin_epoca"] = resumen_modelo(m_sin)
R["h1"]["con_epoca"] = resumen_modelo(m_con)
R["h1"]["con_epoca"]["coef_epoca"] = float(m_con.params["epoca"])
R["h1"]["con_epoca"]["ic_epoca"] = [float(v) for v in m_con.conf_int().loc["epoca"]]
R["h1"]["con_epoca"]["p_epoca"] = float(m_con.pvalues["epoca"])
R["h1"]["con_epoca"]["vif_anio"] = float(1 / (1 - smf.ols("anio ~ epoca", d_h1).fit().rsquared))


def bootstrap_bloques(df, estadistico, largo=BLOQUE_ANUAL, n_boot=N_BOOT, rng=RNG):
    """Bootstrap por bloques móviles: remuestrea tramos de `largo` años consecutivos (conserva la autocorrelación)."""
    n = len(df)
    inicios = np.arange(n - largo + 1)
    k = int(np.ceil(n / largo))
    vals = []
    for _ in range(n_boot):
        idx = np.concatenate([np.arange(i, i + largo) for i in rng.choice(inicios, k)])[:n]
        try:
            vals.append(estadistico(df.iloc[idx]))
        except Exception:
            continue
    vals = np.array(vals, dtype=float)
    vals = vals[np.isfinite(vals)]
    return np.percentile(vals, [2.5, 97.5]), vals


for clave, formula in (("sin_epoca", "y ~ anio"), ("con_epoca", "y ~ anio + epoca")):
    ic, vals = bootstrap_bloques(d_h1, lambda d, f=formula: smf.ols(f, d).fit().params["anio"]
                                 if d["epoca"].nunique() > 1 or "epoca" not in f else np.nan)
    R["h1"][clave]["ic_boot_bloques"] = [float(ic[0]), float(ic[1])]
    R["h1"][clave]["p_boot_aprox"] = float(min(1, 2 * min((vals <= 0).mean(), (vals >= 0).mean())))

# Alternativa no paramétrica: Theil-Sen + Mann-Kendall (Kendall tau con el año), dentro de cada época
for ep_nombre, g in (("todos", d_h1), ("L5/L7", d_h1[d_h1["epoca"] == 0]), ("L8/L9", d_h1[d_h1["epoca"] == 1])):
    ts = stats.theilslopes(g["y"], g["anio"])
    kt = stats.kendalltau(g["anio"], g["y"])
    R["h1"][f"theil_sen_{ep_nombre}"] = {"pendiente": float(ts.slope), "ic_inf": float(ts.low_slope),
                                         "ic_sup": float(ts.high_slope), "tau": float(kt.statistic),
                                         "p": float(kt.pvalue), "n": int(len(g))}
# Sensibilidad 1: serie sin corregir Landsat 7 (variacion_anual_v2), mismo modelo
d_s1 = pd.DataFrame({"y": var_anual["area_media_obs"], "anio": var_anual.index,
                     "epoca": (var_anual["epoca"] == "L8/L9").astype(int)})
R["h1"]["sens_sin_correccion_L7"] = {k: resumen_modelo(smf.ols(f, d_s1).fit()) for k, f in
                                     (("sin_epoca", "y ~ anio"), ("con_epoca", "y ~ anio + epoca"))}
# Sensibilidad 3: sin los 4 meses atípicos (1,5·IQR por época). Base: promedio de meses observados sin corrección L7.
def anual_desde_mensual(serie):
    a = serie.groupby(serie.index.year).agg(["mean", "count"])
    a = a[a["count"] >= 3]
    return pd.DataFrame({"y": a["mean"], "anio": a.index, "epoca": (a.index > 2012).astype(int)})


mens = base_m[COL].dropna()
excl = pd.to_datetime([r["mes"] for r in R["atipicos"]["mensual_lista"]])
d_con, d_sin_at = anual_desde_mensual(mens), anual_desde_mensual(mens.drop(excl))
R["h1"]["sens_atipicos"] = {"meses_excluidos": [str(e.date()) for e in excl],
                            "pend_con_epoca_todos": float(smf.ols("y ~ anio + epoca", d_con).fit().params["anio"]),
                            "pend_con_epoca_sin_atipicos": float(smf.ols("y ~ anio + epoca", d_sin_at).fit().params["anio"]),
                            "p_con_epoca_sin_atipicos": float(smf.ols("y ~ anio + epoca", d_sin_at).fit().pvalues["anio"]),
                            "pend_sin_epoca_sin_atipicos": float(smf.ols("y ~ anio", d_sin_at).fit().params["anio"])}
# Sensibilidad 2: periodo de la propuesta (2004-2023)
d_s2 = d_h1[(d_h1["anio"] >= 2004) & (d_h1["anio"] <= 2023)]
R["h1"]["sens_2004_2023"] = {k: resumen_modelo(smf.ols(f, d_s2).fit()) for k, f in
                             (("sin_epoca", "y ~ anio"), ("con_epoca", "y ~ anio + epoca"))}
R["h1"]["sens_2004_2023"]["n"] = int(len(d_s2))
# Tamaño del efecto: cambio total implícito en el periodo
R["h1"]["cambio_total_sin_epoca_ha"] = float(R["h1"]["sin_epoca"]["pendiente"] * (d_h1["anio"].max() - d_h1["anio"].min()))
R["h1"]["cambio_total_con_epoca_ha"] = float(R["h1"]["con_epoca"]["pendiente"] * (d_h1["anio"].max() - d_h1["anio"].min()))
R["h1"]["media_por_epoca"] = d_h1.groupby("epoca")["y"].mean().round(2).to_dict()
TABLAS["t14_serie_anual_h1"] = d_h1

# ---- Figura 5: serie anual con ajuste (antes de ajustar se grafica; aquí se muestran ambos ajustes)
fig, ax = plt.subplots(figsize=(7.2, 3.0))
for ep, c, lab in ((0, C1, "Landsat 5 y 7"), (1, C2, "Landsat 8 y 9")):
    g = d_h1[d_h1["epoca"] == ep]
    ax.plot(g["anio"], g["y"], "o", color=c, ms=4.5, label=f"{lab} (n = {len(g)} años)")
xx = np.arange(d_h1["anio"].min(), d_h1["anio"].max() + 1)
ax.plot(xx, m_sin.params["Intercept"] + m_sin.params["anio"] * xx, color=GRIS, lw=1.3, ls="--",
        label=f"Sin separar satélites: {m_sin.params['anio']:+.2f} ha por año")
for ep, c in ((0, C1), (1, C2)):
    xs = xx[(xx <= 2011) if ep == 0 else (xx >= 2013)]
    ax.plot(xs, m_con.params["Intercept"] + m_con.params["anio"] * xs + m_con.params["epoca"] * ep, color=c, lw=1.6)
ax.plot([], [], color=TINTA, lw=1.6, label=f"Separando satélites: {m_con.params['anio']:+.2f} ha por año")
ax.set(xlabel="Año", ylabel="Superficie media anual (ha)", title="Superficie media anual y tendencia, 1986-2024")
ax.legend(fontsize=7, loc="upper left")
guardar_fig(fig, "f05_tendencia_anual.png")

# ---- Figura A1: diagnóstico del modelo con época (residuos, QQ, Cook, ACF)
infl = m_con.get_influence()
fig, axes = plt.subplots(2, 2, figsize=(7.2, 5.2))
rs = infl.resid_studentized_internal
axes[0, 0].scatter(m_con.fittedvalues, rs, s=14, color=C1)
for h in (-2, 0, 2):
    axes[0, 0].axhline(h, color=GRIS, ls=":" if h else "-", lw=0.8)
axes[0, 0].set(xlabel="Valor ajustado (ha)", ylabel="Residuo estandarizado", title="Residuos vs ajustados")
sm.qqplot(m_con.resid, line="s", ax=axes[0, 1], markersize=3.5, markerfacecolor=C1, markeredgecolor=C1)
axes[0, 1].get_lines()[1].set_color(GRIS)
axes[0, 1].set(title=f"QQ normal (Shapiro-Wilk p = {R['h1']['con_epoca']['shapiro_p']:.2f})",
               xlabel="Cuantil teórico", ylabel="Cuantil observado (ha)")
cook = infl.cooks_distance[0]
axes[1, 0].vlines(d_h1["anio"], 0, cook, color=C1)
axes[1, 0].axhline(4 / len(d_h1), color=GRIS, ls=":", lw=0.8)
axes[1, 0].set(xlabel="Año", ylabel="Distancia de Cook", title="Influencia (línea: 4/n)")
plot_acf(m_con.resid, ax=axes[1, 1], lags=10, color=C1, vlines_kwargs={"colors": C1})
axes[1, 1].set(title="Autocorrelación de los residuos", xlabel="Rezago (años)", ylabel="ACF")
guardar_fig(fig, "fA1_diagnostico_tendencia.png")

# ============================== 9. H3: efecto del sensor ==============================
dif = pares["dif_L7_menos_L5"]
sw = stats.shapiro(dif)
tt = stats.ttest_rel(pares["L7"], pares["L5"])
wx = stats.wilcoxon(pares["L7"], pares["L5"])
boot_med = [np.median(RNG.choice(dif.values, len(dif))) for _ in range(N_BOOT)]
R["h3"] = {"n_pares": int(len(pares)), "mediana": float(dif.median()), "media": float(dif.mean()),
           "de": float(dif.std()), "shapiro_p": float(sw.pvalue), "t": float(tt.statistic), "t_p": float(tt.pvalue),
           "t_ic": [float(v) for v in tt.confidence_interval()],
           "wilcoxon_p": float(wx.pvalue), "ic_mediana_boot": [float(v) for v in np.percentile(boot_med, [2.5, 97.5])],
           "dz": float(dif.mean() / dif.std()), "pares_negativos": int((dif < 0).sum()),
           "periodo": [str(pares["fecha"].min().date()), str(pares["fecha"].max().date())]}
li, ls_ = limites_iqr(dif)
R["h3"]["pares_atipicos_iqr"] = pares.loc[(dif < li) | (dif > ls_), ["fecha", "dif_L7_menos_L5"]].assign(
    fecha=lambda d: d["fecha"].dt.date.astype(str)).round(1).to_dict("records")
R["h3"]["salto_csv"] = saltos.iloc[0].to_dict()

# ANOVA de dos factores (sensor × estación), escenas válidas. Tipo III por desbalance; interacción primero.
an_df = validas[validas["sensor"].isin(["L5", "L7", "L8", "L9"])].copy()
an_df["grupo"] = an_df["sensor"].replace({"L8": "L8/L9", "L9": "L8/L9"})
an_df["est"] = an_df["estacion"]
an_df = an_df.rename(columns={COL: "area"})
m_int = smf.ols("area ~ C(grupo, Sum) * C(est, Sum)", an_df).fit()
tab_int = anova_lm(m_int, typ=3)
m_adit = smf.ols("area ~ C(grupo, Sum) + C(est, Sum)", an_df).fit()
tab_adit = anova_lm(m_adit, typ=3)
TABLAS["t15_anova_interaccion"] = tab_int.reset_index(names="termino").round(4)
TABLAS["t16_anova_aditivo"] = tab_adit.reset_index(names="termino").round(4)
ss_tot = ((an_df["area"] - an_df["area"].mean()) ** 2).sum()
R["h3"]["anova"] = {"n": int(len(an_df)), "n_por_grupo": an_df["grupo"].value_counts().to_dict(),
                    "p_interaccion": float(tab_int.loc["C(grupo, Sum):C(est, Sum)", "PR(>F)"]),
                    "F_interaccion": float(tab_int.loc["C(grupo, Sum):C(est, Sum)", "F"]),
                    "p_sensor": float(tab_adit.loc["C(grupo, Sum)", "PR(>F)"]),
                    "F_sensor": float(tab_adit.loc["C(grupo, Sum)", "F"]),
                    "p_estacion": float(tab_adit.loc["C(est, Sum)", "PR(>F)"]),
                    "eta2_parcial_sensor": float(tab_adit.loc["C(grupo, Sum)", "sum_sq"] /
                                                 (tab_adit.loc["C(grupo, Sum)", "sum_sq"] + tab_adit.loc["Residual", "sum_sq"])),
                    "levene_p": float(stats.levene(*[g["area"] for _, g in an_df.groupby("grupo")]).pvalue),
                    "shapiro_resid_p": float(stats.shapiro(m_adit.resid).pvalue),
                    "medias": an_df.groupby("grupo")["area"].mean().round(1).to_dict()}
medias_ge = an_df.pivot_table(index="est", columns="grupo", values="area", aggfunc=["mean", "count"]).reindex(ORDEN_EST)
medias_ge.columns = [f"{a}_{b}" for a, b in medias_ge.columns]
TABLAS["t15b_medias_sensor_estacion"] = medias_ge.round(1).reset_index()
R["h3"]["medias_sensor_estacion"] = medias_ge.round(1).to_dict("index")
tk = pairwise_tukeyhsd(an_df["area"], an_df["grupo"])
tk_df = pd.DataFrame(tk.summary().data[1:], columns=tk.summary().data[0])
TABLAS["t17_tukey"] = tk_df
R["h3"]["tukey"] = tk_df.to_dict("records")


def welch_anova(grupos):
    k = len(grupos)
    n = np.array([len(g) for g in grupos]); m = np.array([g.mean() for g in grupos]); v = np.array([g.var(ddof=1) for g in grupos])
    w = n / v; mw = (w * m).sum() / w.sum()
    a = (w * (m - mw) ** 2).sum() / (k - 1)
    tmp = ((1 - w / w.sum()) ** 2 / (n - 1)).sum()
    b = 1 + 2 * (k - 2) / (k ** 2 - 1) * tmp
    F = a / b
    gl2 = (k ** 2 - 1) / (3 * tmp)
    return F, k - 1, gl2, stats.f.sf(F, k - 1, gl2)


# Welch sobre el área sin el ciclo estacional (resta la media de cada estación)
an_df["area_des"] = an_df["area"] - an_df.groupby("est")["area"].transform("mean")
F, g1, g2, p = welch_anova([g["area_des"].values for _, g in an_df.groupby("grupo")])
R["h3"]["welch"] = {"F": float(F), "gl1": int(g1), "gl2": float(g2), "p": float(p)}
R["h3"]["kruskal_p"] = float(stats.kruskal(*[g["area_des"].values for _, g in an_df.groupby("grupo")]).pvalue)

# ---- Figura 6: pares L7 vs L5
fig, axes = plt.subplots(1, 2, figsize=(7.2, 2.9))
lim = [pares[["L5", "L7"]].min().min() - 2, pares[["L5", "L7"]].max().max() + 2]
axes[0].plot(lim, lim, color=GRIS, ls="--", lw=1, label="Igual medición (L7 = L5)")
axes[0].scatter(pares["L5"], pares["L7"], s=18, color=C1, label=f"Pares de imágenes (n = {len(pares)})")
axes[0].set(xlabel="Superficie según Landsat 5 (ha)", ylabel="Superficie según Landsat 7 (ha)",
            title="Misma laguna, dos satélites (≤ 8 días)")
axes[0].legend(fontsize=7)
axes[1].hist(dif, bins=np.arange(-24, 3, 2), color=C1, edgecolor="white")
axes[1].axvline(R["h3"]["mediana"], color=TINTA, lw=1.4)
axes[1].axvline(0, color=GRIS, ls="--", lw=1)
axes[1].text(R["h3"]["mediana"] - 0.5, axes[1].get_ylim()[1] * 0.9, f"mediana {R['h3']['mediana']:+.1f} ha",
             ha="right", fontsize=7.5)
axes[1].set(xlabel="Diferencia Landsat 7 − Landsat 5 (ha)", ylabel="N.º de pares", title="Diferencia en cada par")
guardar_fig(fig, "f06_pares_L7_L5.png")

# ============================== 10. H2: asociación con el clima (anual) ==============================
def n_efectivo(x, y):
    """n efectivo para la correlación de dos series autocorrelacionadas (aprox. de Bartlett, rezago 1)."""
    r1x, r1y = pd.Series(x).autocorr(1), pd.Series(y).autocorr(1)
    n = len(x)
    f = (1 - r1x * r1y) / (1 + r1x * r1y)
    return float(min(n, n * f)), float(r1x), float(r1y)


def r_minimo(n, alfa=ALFA, potencia=POTENCIA):
    """Menor |r| detectable (bilateral) con n observaciones independientes (transformación z de Fisher)."""
    z = stats.norm.ppf(1 - alfa / 2) + stats.norm.ppf(potencia)
    return float(np.tanh(z / np.sqrt(n - 3)))


def n_necesario(r, alfa=ALFA, potencia=POTENCIA):
    z = stats.norm.ppf(1 - alfa / 2) + stats.norm.ppf(potencia)
    return float((z / np.arctanh(r)) ** 2 + 3)


hasta = 2019
d_med = an12.loc[:hasta, ["media_anual", "epoca", "P12_med", "P24_med", "P36_med", "SPEI12_med", "SPEI24_med", "Tx_verano"]].dropna()
d_med["y_sin_epoca"] = d_med["media_anual"] - d_med.groupby("epoca")["media_anual"].transform("mean")
d_min = anual12.loc[:hasta, ["min_verano", "epoca", "P12_min", "P24_min", "P36_min", "SPEI12_min", "SPEI24_min", "Tx_verano"]].dropna()
d_min["y_sin_epoca"] = d_min["min_verano"] - d_min.groupby("epoca")["min_verano"].transform("mean")
filas = []
for obj, d, suf in (("Promedio anual", d_med, "_med"), ("Mínimo de verano", d_min, "_min")):
    for v in ["P12", "P24", "P36", "SPEI12", "SPEI24", "Tx_verano"]:
        col = v + suf if v != "Tx_verano" else v
        rho, p = stats.spearmanr(d["y_sin_epoca"], d[col])
        rc, pc = stats.spearmanr(d.iloc[:, 0], d[col])
        r, pr = stats.pearsonr(d["y_sin_epoca"], d[col])
        ne, r1x, r1y = n_efectivo(d["y_sin_epoca"].values, d[col].values)
        ic, _ = bootstrap_bloques(d[["y_sin_epoca", col]].reset_index(drop=True),
                                  lambda b, c=col: stats.spearmanr(b["y_sin_epoca"], b[c])[0], n_boot=2000)
        filas.append((obj, v, len(d), round(rc, 3), round(pc, 3), round(rho, 3), round(p, 3), round(ic[0], 2),
                      round(ic[1], 2), round(r, 3), round(pr, 3), round(rho ** 2 * 100, 1), round(ne, 1),
                      round(r_minimo(max(ne, 4)), 2)))
corr_tab = pd.DataFrame(filas, columns=["objetivo", "variable", "n_anios", "rho_cruda", "p_cruda", "rho_sin_epoca",
                                        "p_sin_epoca", "ic95_boot_inf", "ic95_boot_sup", "pearson_r_sin_epoca",
                                        "p_pearson", "pct_variabilidad_rho2", "n_efectivo", "r_min_detectable"])
TABLAS["t18_correlaciones_anuales"] = corr_tab
# Coincidencia con el CSV del notebook
m_ = corr_tab.assign(variable=corr_tab["variable"]).merge(
    corr12.replace({"objetivo": {"media_anual": "Promedio anual", "min_verano": "Mínimo de verano"}}),
    on=["objetivo", "variable"], suffixes=("", "_csv"))
R["h2"] = {"max_dif_rho_vs_csv": float((m_["rho_sin_epoca"] - m_["rho_sin_epoca_csv"]).abs().max()),
           "n_anios": int(len(d_med)), "n_anios_L8L9": int((d_med["epoca"] == 1).sum()),
           "r_min_n31": r_minimo(len(d_med)), "n_para_r03": n_necesario(0.3), "n_para_r027": n_necesario(0.27),
           "mejor": corr_tab.loc[corr_tab["rho_sin_epoca"].abs().idxmax()].to_dict()}
# Regresión con y sin época: promedio anual ~ P36
for clave, f in (("P36_sin_epoca", "media_anual ~ P36_med"), ("P36_con_epoca", "media_anual ~ P36_med + epoca")):
    m = smf.ols(f, d_med).fit()
    R["h2"][clave] = {"coef_100mm": float(m.params["P36_med"] * 100), "ic_100mm": [float(v * 100) for v in m.conf_int().loc["P36_med"]],
                      "p": float(m.pvalues["P36_med"]), "r2": float(m.rsquared), "r2_aj": float(m.rsquared_adj)}
    if "epoca" in f:
        R["h2"][clave]["coef_epoca"] = float(m.params["epoca"])
# Ejemplo de años lluviosos con superficie baja (2000-2003)
ej = pd.DataFrame({"lluvia_mm": lluvia_anual.loc["1997":"2004"].values}, index=range(1997, 2005))
ej["area_media"] = an12["media_anual"].reindex(ej.index)
TABLAS["t19_ejemplo_lluvia_area"] = ej.reset_index(names="anio").round(1)
R["h2"]["ejemplo_2000_2003"] = ej.loc[2000:2003].round(1).to_dict("index")

# Correlaciones mensuales del notebook (208 meses) con su n efectivo
base_c = base_m.loc[:"2019"].dropna(subset=[COL, "p12", "tmax"])
an_mens = base_c["area_anom"]
r1_area = float(base_m["area_anom"].autocorr(1))
r1_p12 = float(clima["precip_mm"].rolling(12).sum().autocorr(1))
r1_tmax_anom = float((clima["tmax_c"] - clima.groupby(clima.index.month)["tmax_c"].transform("mean")).autocorr(1))
R["h2"]["mensual"] = {"n": int(len(base_c)), "r1_area_anom": r1_area, "r1_p12": r1_p12, "r1_tmax_anom": r1_tmax_anom,
                      "n_ef_p12": float(len(base_c) * (1 - r1_area * r1_p12) / (1 + r1_area * r1_p12)),
                      "n_ef_tmax": float(len(base_c) * (1 - r1_area * r1_tmax_anom) / (1 + r1_area * r1_tmax_anom)),
                      "csv": corr_m.set_index("variable")[["rho_anom_sin_epoca", "p_anom_sin_epoca", "n", "R2_aprox"]].to_dict("index")}
for k in ("n_ef_p12", "n_ef_tmax"):
    R["h2"]["mensual"]["r_min_" + k] = r_minimo(R["h2"]["mensual"][k])
# Valor p corregido por autocorrelación para cada correlación mensual (prueba t con n efectivo)
cl_anom = clima.rename(columns={"precip_mm": "precip", "tmax_c": "tmax", "tmin_c": "tmin"})
feat = pd.DataFrame({"precip": cl_anom["precip"], "p1": cl_anom["precip"].shift(1), "p2": cl_anom["precip"].shift(2),
                     "p3": cl_anom["precip"].rolling(3).sum(), "p12": cl_anom["precip"].rolling(12).sum(),
                     "tmax": cl_anom["tmax"], "tmin": cl_anom["tmin"]})
feat = feat - feat.groupby(feat.index.month).transform("mean")      # sin ciclo estacional, igual que el área
filas = []
for v_, fila in corr_m.set_index("variable").iterrows():
    r1v = float(feat[v_].autocorr(1))
    ne = len(base_c) * (1 - r1_area * r1v) / (1 + r1_area * r1v)
    rho = fila["rho_anom_sin_epoca"]
    t_ = rho * np.sqrt((ne - 2) / (1 - rho ** 2))
    filas.append((v_, rho, fila["p_anom_sin_epoca"], int(fila["n"]), round(r1v, 2), round(ne, 0),
                  round(float(2 * stats.t.sf(abs(t_), ne - 2)), 3), round(100 * rho ** 2, 1)))
TABLAS["t18b_correlaciones_mensuales_n_efectivo"] = pd.DataFrame(
    filas, columns=["variable", "rho_sin_epoca", "p_clasico", "n_meses", "autocorr_clima_r1", "n_efectivo", "p_con_n_efectivo", "pct_variabilidad"])
R["h2"]["mensual_n_ef"] = TABLAS["t18b_correlaciones_mensuales_n_efectivo"].to_dict("records")

# ---- Figura 7: dispersión Torca (sin época) vs P36, y Aculeo como contraste
fig, axes = plt.subplots(1, 2, figsize=(7.2, 3.0))
for ep, c, lab in ((0, C1, "Landsat 5 y 7"), (1, C2, "Landsat 8 y 9")):
    g = d_med[d_med["epoca"] == ep]
    axes[0].scatter(g["P36_med"], g["y_sin_epoca"], s=18, color=c, label=lab)
rr = corr_tab.query("objetivo == 'Promedio anual' and variable == 'P36'").iloc[0]
axes[0].axhline(0, color=GRIS, lw=0.8)
axes[0].set(xlabel="Lluvia de los 3 años anteriores (mm)", ylabel="Superficie del año respecto de lo\nnormal para su satélite (ha)",
            title=f"Laguna Torca: sin relación clara ({int(rr['n_anios'])} años)")
axes[0].legend(fontsize=7)
ac = var_ac.rename(columns={var_ac.columns[0]: "anio"})
axes[1].plot(ac["anio"], ac["area_media_obs"] if "area_media_obs" in ac else ac.iloc[:, 1], "o-", color=C3, ms=3, lw=1)
axes[1].set(xlabel="Año", ylabel="Superficie media anual (ha)",
            title="Laguna de Aculeo (comparación): se secó con la sequía")
guardar_fig(fig, "f07_clima_torca_aculeo.png")

# ============================== 11. Autocorrelación ==============================
acf_anual = acf(d_h1["y"].values, nlags=5)
acf_m = acf(base_m["area_anom"].values, nlags=24, missing="conservative")
R["autocorr"] = {"acf_anual_rezago1": float(acf_anual[1]), "acf_mensual_rezago1": float(acf_m[1]),
                 "acf_mensual_rezago12": float(acf_m[12]),
                 "banda95_anual": float(1.96 / np.sqrt(len(d_h1))),
                 "banda95_mensual": float(1.96 / np.sqrt(base_m["area_anom"].notna().sum()))}
fig, axes = plt.subplots(1, 2, figsize=(7.2, 2.7))
plot_acf(d_h1["y"].values, ax=axes[0], lags=10, color=C1, vlines_kwargs={"colors": C1})
axes[0].set(title="Superficie media anual", xlabel="Rezago (años)", ylabel="Autocorrelación")
plot_acf(base_m["area_anom"].values, ax=axes[1], lags=24, missing="conservative", color=C1, vlines_kwargs={"colors": C1})
axes[1].set(title="Anomalía mensual (sin estación ni época)", xlabel="Rezago (meses)", ylabel="Autocorrelación")
guardar_fig(fig, "f08_autocorrelacion.png")

# ============================== 12. Validación predictiva ==============================
# 12a. Se reproduce la validación un año hacia adelante (referencias y modelos lineales) desde la tabla anual
def cv_un_anio(d, y, xs, desde=2000, min_entren=12):
    pred = {}
    for anio in d.index[d.index >= desde]:
        tr = d[d.index < anio]
        if len(tr) < min_entren:
            continue
        X = np.column_stack([np.ones(len(tr))] + [tr[c] for c in xs])
        beta = np.linalg.lstsq(X, tr[y], rcond=None)[0]
        x0 = np.r_[1, [d.loc[anio, c] for c in xs]]
        pred[anio] = (d.loc[anio, y], float(x0 @ beta), float(tr[y].iloc[-1]))
    p = pd.DataFrame(pred, index=["obs", "modelo", "persistencia"]).T
    return p


val_rep = {}
for obj, d, suf in (("media_anual", d_med, "_med"), ("min_verano", d_min, "_min")):
    for v in ["P12", "P24", "P36", "SPEI12", "SPEI24"]:
        p = cv_un_anio(d, obj, ["epoca", v + suf])
        val_rep[(obj, f"Lineal: época + {v}")] = float(np.sqrt(((p["modelo"] - p["obs"]) ** 2).mean()))
    p = cv_un_anio(d, obj, ["epoca"])
    val_rep[(obj, "Media de época (referencia)")] = float(np.sqrt(((p["modelo"] - p["obs"]) ** 2).mean()))
    val_rep[(obj, "Persistencia: año anterior (referencia)")] = float(np.sqrt(((p["persistencia"] - p["obs"]) ** 2).mean()))
    # Comparación pareada con bootstrap: media de época vs el mejor modelo lineal con clima de este objetivo
    mejores = {k[1]: v for k, v in val_rep.items() if k[0] == obj and k[1].startswith("Lineal")}
    mejor_cl = min(mejores, key=mejores.get)
    pe, pc = cv_un_anio(d, obj, ["epoca"]), cv_un_anio(d, obj, ["epoca", mejor_cl.split("+ ")[1] + suf])
    e_ref, e_cl = (pe["modelo"] - pe["obs"]).values, (pc["modelo"] - pc["obs"]).values
    idx = RNG.integers(0, len(e_ref), size=(N_BOOT, len(e_ref)))
    difb = np.sqrt((e_ref[idx] ** 2).mean(1)) - np.sqrt((e_cl[idx] ** 2).mean(1))
    R[f"validacion_dif_boot_{obj}"] = {"mejor_clima": mejor_cl, "ic": [float(v) for v in np.percentile(difb, [2.5, 97.5])]}
    R["validacion_n_anios_prueba"] = int(len(pe))
vr = pd.Series(val_rep).rename("RMSE_reproducido").reset_index()
vr.columns = ["objetivo", "modelo", "RMSE_reproducido"]
vcomp = val12.merge(vr, on=["objetivo", "modelo"], how="left")
TABLAS["t20_validacion_anual"] = vcomp
R["validacion"] = {"max_dif_rmse_reproducido": float((vcomp["RMSE"] - vcomp["RMSE_reproducido"]).abs().max()),
                   "tabla": vcomp.to_dict("records"), "mensual_csv": cv_m.reset_index(names="modelo").to_dict("records"),
                   "sarimax_csv": cv_sar.reset_index(names="modelo").to_dict("records")}

# ---- Figura 9: RMSE de validación (promedio anual)
v = val12[val12["objetivo"] == "media_anual"].sort_values("RMSE").copy()
NOMBRES_SIMPLES = {"Media de época (referencia)": "Promedio de su período de satélite (referencia)",
                   "Persistencia: año anterior (referencia)": "Repetir el año anterior (referencia)",
                   "Ridge: época + todas": "Ridge: período + todas las variables",
                   "Random Forest: época + todas": "Bosque aleatorio: período + todas las variables"}
v["modelo"] = v["modelo"].map(lambda m: NOMBRES_SIMPLES.get(m, m.replace("Lineal: época + ", "Recta: período + ")
                                                             .replace("SPEI", "SPEI ").replace("P12", "lluvia 12 m")
                                                             .replace("P24", "lluvia 24 m").replace("P36", "lluvia 36 m")))
fig, ax = plt.subplots(figsize=(7.2, 2.9))
colores = [GRIS if "referencia" in m else C1 for m in v["modelo"]]
ax.barh(v["modelo"], v["RMSE"], color=colores, height=0.6)
for i, (m, r_) in enumerate(zip(v["modelo"], v["RMSE"])):
    ax.text(r_ + 0.05, i, f"{r_:.2f}", va="center", fontsize=7.5)
ax.invert_yaxis()
ax.set(xlabel="Error típico del pronóstico a un año (hectáreas; menor es mejor)",
       title="Pronóstico de cada año 2000-2019 (gris = reglas simples, azul = modelos con clima)")
guardar_fig(fig, "f09_validacion.png")

# ============================== 13. Extrapolación de temperatura (RCP 8.5) ==============================
def leer_cr2(nombre):
    t = (RAW / nombre).read_bytes().decode("utf-8-sig", errors="ignore")
    d = pd.read_csv(io.StringIO(t), sep=";", decimal=",", thousands=".")
    d["DateTime"] = pd.to_datetime(d["DateTime"])
    return d.set_index("DateTime")


tx = leer_cr2("proyeccin-de-la-temperat max.csv")
deltas = []
for c in tx.columns:
    if c.endswith("_tasmax rcp85"):
        g = c[: -len("_tasmax rcp85")]
        h = f"{g}_tasmax historical"
        if h in tx.columns:
            fut = tx[c].loc["2071":"2095"]
            his = tx[h].loc["1986":"2005"]
            fut, his = fut[fut.index.month.isin([1, 2, 3, 4])], his[his.index.month.isin([1, 2, 3, 4])]
            if fut.notna().all() and his.notna().all() and len(fut) and len(his):
                deltas.append(fut.mean() - his.mean())
tx_obs = an12.loc[:2019, "Tx_verano"] if "Tx_verano" in an12 else anual12.loc[:2019, "Tx_verano"]
txv = anual12.loc[1985:2019, "Tx_verano"]
base_tx = anual12.loc[1986:2005, "Tx_verano"].mean()
R["extrapolacion"] = {"n_gcm": len(deltas), "delta_mediana": float(np.median(deltas)),
                      "delta_p10": float(np.percentile(deltas, 10)), "delta_p90": float(np.percentile(deltas, 90)),
                      "tx_verano_obs_min": float(txv.min()), "tx_verano_obs_max": float(txv.max()),
                      "tx_verano_base_1986_2005": float(base_tx),
                      "tx_verano_proyectada": float(base_tx + np.median(deltas))}
R["extrapolacion"]["exceso_sobre_max_obs"] = R["extrapolacion"]["tx_verano_proyectada"] - R["extrapolacion"]["tx_verano_obs_max"]

# ============================== 14. Aculeo (contraste) ==============================
R["aculeo"] = {"p36": corr_ac.loc["p36"].to_dict(), "p24": corr_ac.loc["p24"].to_dict(),
               "calibracion": cal_ac.to_dict("records"), "cv": cv_ac.reset_index(names="modelo").to_dict("records"),
               "proy": proy_ac.to_dict("records"),
               "r_min_n_ef": r_minimo(corr_ac.loc["p36", "n_efectivo"])}

# ============================== 15. Imputación MICE (solo comparación) ==============================
R["mice"] = {"circularidad": circ.to_dict("records"), "validacion": mice_val.to_dict("records"),
             "dif_escenarios": comp_esc.to_dict("records")}
R["proyecciones"] = {"anual12": proy12.to_dict("records"), "mensual": proy_m.to_dict("records")}

# ============================== 15b. Laguna Miscanti (control natural sin intervención) ==============================
# Resultados del notebook Laguna_Miscanti_v1.ipynb (mismo método que Torca), leídos desde sus CSV. Se recalcula aquí
# la tendencia JRC para verificarla de forma independiente.
MIS = BASE / "datos" / "miscanti"
jrc = pd.read_csv(MIS / "serie_laguna_miscanti_jrc.csv", parse_dates=["fecha"])
jrc = jrc[(jrc["version"] == "v1.4") & jrc["fecha"].dt.month.isin([10, 11, 12, 1, 2, 3, 4])
          & (jrc["sin_dato_ha"] < 0.05 * jrc["poligono_ha"])]
anio_hidro = jrc["fecha"].dt.year + (jrc["fecha"].dt.month >= 7).astype(int)       # año hidrológico jul-jun
ja = jrc.groupby(anio_hidro)["agua_ha"].agg(["mean", "count"])
ja = ja[ja["count"] >= 3]["mean"].loc[:2021]          # JRC v1.4 (clasificador único) hasta 2021, como en el notebook de Miscanti
ts_m, kt_m = stats.theilslopes(ja.values, ja.index.values), stats.kendalltau(ja.index.values, ja.values)
comp_tm = pd.read_csv(MIS / "comparacion_torca_vs_miscanti_v1.csv", index_col=0)
dm_m = pd.read_csv(MIS / "dm_hln_mensual_miscanti_v1.csv")
ho_m = pd.read_csv(MIS / "holdout_mensual_miscanti_v1.csv", index_col=0)
cv_mi = pd.read_csv(MIS / "cv_nivel_area_miscanti_v1.csv", index_col=0)
sal_m = pd.read_csv(MIS / "anual12_saltos_por_fuente_miscanti_v1.csv")
pro_m = pd.read_csv(MIS / "anual12_resumen_proyecciones_miscanti_v1.csv")
cor_n = pd.read_csv(MIS / "anual12_correlaciones_miscanti_v1.csv")
cor_j = pd.read_csv(MIS / "anual12_correlaciones_jrc_miscanti_v1.csv")
val_tabs = {"NDWI": pd.read_csv(MIS / "anual12_validacion_miscanti_v1.csv"), "JRC": pd.read_csv(MIS / "anual12_validacion_jrc_miscanti_v1.csv")}
filas = []
for fuente, v_ in val_tabs.items():
    for obj, g in v_.groupby("objetivo", sort=False):
        ref = g[g["modelo"].str.contains("referencia")].sort_values("RMSE").iloc[0]
        cl = g[~g["modelo"].str.contains("referencia")].sort_values("RMSE").iloc[0]
        filas.append((fuente, obj, cl["modelo"], cl["RMSE"], ref["modelo"], ref["RMSE"], int(cl["n_años"])))
TABLAS["t21_validacion_anual_miscanti"] = pd.DataFrame(filas, columns=["fuente", "objetivo", "mejor_con_clima", "RMSE_clima",
                                                                        "mejor_referencia", "RMSE_referencia", "n_anios"])
dmr = dm_m.set_index("comparacion")
lp = pro_m[pro_m["horizonte"].str.startswith("Largo")]
# Variación entre años de cada laguna (desviación estándar del promedio anual)
torca_dev = d_h1["y"] - d_h1.groupby("epoca")["y"].transform("mean")
ac_anual = var_ac.set_index(var_ac.columns[0])["area_media_obs"]
R["miscanti"] = {
    "area_media_ndwi": float(str(comp_tm.loc["Área media observada (ha)", "Miscanti"]).replace(",", ".")),
    "jrc_media": float(ja.mean()), "jrc_de_anual": float(ja.std()), "jrc_de_pct": float(100 * ja.std() / ja.mean()),
    "jrc_n_anios": int(len(ja)), "jrc_anio_min": int(ja.index.min()), "jrc_anio_max": int(ja.index.max()),
    "jrc_pendiente": float(ts_m.slope), "jrc_pend_ic": [float(ts_m.low_slope), float(ts_m.high_slope)],
    "jrc_tau": float(kt_m.statistic), "jrc_p": float(kt_m.pvalue), "jrc_pend_pct": float(100 * ts_m.slope / ja.mean()),
    "rho_max_ndwi": float(cor_n["rho_sin_epoca"].abs().max()), "rho_max_jrc": float(cor_j["rho_sin_epoca"].abs().max()),
    "n_corr_p05": int((cor_n["p_sin_epoca"] < 0.05).sum() + (cor_j["p_sin_epoca"] < 0.05).sum()),
    "n_corr_total": int(len(cor_n) + len(cor_j)),
    "dm_con_vs_sin_p": float(dmr.loc["Ridge con_clima vs sin_clima", "p_h1"]),
    "dm_con_vs_sin_p_sin_pliegue2": float(dmr.loc["Ridge [sin pliegues [2]] con_clima vs sin_clima", "p_h1"]),
    "cv_ridge_clima": float(cv_mi.loc["Ridge (+ clima)", "RMSE"]), "cv_ridge_sin": float(cv_mi.loc["Ridge (estacionalidad)", "RMSE"]),
    "holdout_con": float(ho_m.loc["Ridge (+ clima)", "RMSE"]), "holdout_sin": float(ho_m.loc["Ridge (estacionalidad)", "RMSE"]),
    "holdout_n": int(ho_m.loc["Ridge (+ clima)", "n_meses"]),
    "salto_jrc_mismos_meses": -15.5, "desfase_ndwi_jrc": -18.9,      # tabla comparacion_torca_vs_miscanti_v1.csv
    "sesgo_L7_L5": float(sal_m.loc[sal_m["fuente"] == "original", "sesgo_L7_vs_L5_ha"].iloc[0]),
    "proy_largo": lp[["objetivo", "RCP85_menos_RCP26_ha", "RMSE_validacion_ha"]].to_dict("records"),
    "torca_dm_con_vs_sin_p": float(comp_tm.loc["Mensual: DM-HLN con vs sin clima, mismo modelo (p)", "Torca"]),
    "torca_de_anual_sin_epoca": float(torca_dev.std()), "torca_de_pct": float(100 * torca_dev.std() / d_h1["y"].mean()),
    "aculeo_min": float(ac_anual.min()), "aculeo_max": float(ac_anual.max()),
    "torca_caida_max_detectable": float(-R["h1"]["con_epoca"]["ic_inf"]),
    "torca_caida_max_detectable_pct": float(-100 * R["h1"]["con_epoca"]["ic_inf"] / d_h1["y"].mean()),
}
# verificación: los valores fijos del salto se leen de la tabla de comparación
assert "-15.5 / -18.9" in str(comp_tm.loc["Salto según JRC, mismos meses (ha) / desfase NDWI − JRC", "Miscanti"])
TABLAS["t22_comparacion_torca_miscanti"] = comp_tm.reset_index(names="indicador")

# ---- Figura 10: tres lagunas, cada una respecto de su propio promedio (%)
fig, axes = plt.subplots(1, 3, figsize=(7.4, 2.7))
for ep, c in ((0, C1), (1, C2)):
    g = d_h1[d_h1["epoca"] == ep].set_index("anio")["y"]
    g = g.reindex(range(g.index.min(), g.index.max() + 1))          # deja huecos en los años sin dato
    axes[0].plot(g.index, 100 * (g / d_h1["y"].mean() - 1), "o-", color=c, ms=3, lw=1)
axes[0].set(title="Torca", ylabel="Diferencia con su promedio (%)", xlabel="Año")
axes[1].plot(ja.index, 100 * (ja / ja.mean() - 1), "o-", color=C3, ms=3, lw=1)
axes[1].set(title="Miscanti (sin intervención)", xlabel="Año")
axes[2].plot(ac_anual.index, 100 * (ac_anual / ac_anual.mean() - 1), "o-", color=GRIS, ms=3, lw=1)
axes[2].set(title="Aculeo (se secó en 2019)", xlabel="Año")
for a in axes:
    a.axhline(0, color=GRIS, lw=0.8)
guardar_fig(fig, "f10_tres_lagunas.png")

# ============================== 15c. Proyección 2025-2095 (análisis de sensibilidad) ==============================
# Curvas por GCM generadas por el notebook de Torca: (A) modelo mensual Ridge + clima (sección 8), promedio anual;
# (B) modelo anual del mínimo de verano, "Lineal: época + P36" (sección 12.7). Ambos en la escala de Landsat 8/9 (época = 1).
pm = {e: pd.read_csv(CORR / f"proyeccion_anual_{e}_v2.csv", index_col=0, parse_dates=True) for e in ("rcp26", "rcp85")}
pm_ctrl = pd.read_csv(CORR / "proyeccion_anual_control_v2.csv", index_col=0, parse_dates=True)["control"]
for e in pm:
    pm[e].index = pm[e].index.year
pm_ctrl.index = pm_ctrl.index.year
pa = {e: pd.read_csv(CORR / f"anual12_proyeccion_{e}.csv", index_col=0) for e in ("rcp26", "rcp85")}
rmse_m = float(cv_m.loc["Ridge (+ clima)", "RMSE"])
rmse_a = float(val12.query("objetivo == 'min_verano' and modelo == 'Lineal: época + P36'")["RMSE"].iloc[0])


def resumen_esc(d, desde, hasta):
    t = d.loc[desde:hasta].mean()          # promedio del periodo para cada GCM
    return {"mediana": float(t.median()), "p10": float(t.quantile(0.1)), "p90": float(t.quantile(0.9)), "n_gcm": int(d.shape[1])}


R["proyeccion"] = {
    "mensual": {e: resumen_esc(pm[e], 2071, 2095) for e in pm},
    "mensual_control": float(pm_ctrl.loc[2071:2095].mean()), "mensual_rmse": rmse_m,
    "anual": {e: resumen_esc(pa[e], 2071, 2095) for e in pa}, "anual_rmse": rmse_a,
    "obs_L8L9_media_anual": float(anual12.loc[2013:2024, "media_anual"].mean()),
    "obs_L8L9_min_verano": float(anual12.loc[2013:2024, "min_verano"].mean()),
}
fig, axes = plt.subplots(2, 1, figsize=(7.2, 5.6), sharex=True)
paneles = ((axes[0], pm, pm_ctrl, anual12["media_anual"], rmse_m,
            "A. Promedio anual: modelo mensual con lluvia y temperatura"),
           (axes[1], pa, None, anual12["min_verano"], rmse_a,
            "B. Mínimo de verano (ene-abr): modelo anual con la lluvia de 3 años"))
for ax, dd, ctrl, obs, rmse, tit in paneles:
    obs = obs.dropna()
    for ep, c, lab in ((0, C1, "Observado, Landsat 5 y 7"), (1, C2, "Observado, Landsat 8 y 9")):
        o = obs[(obs.index > 2012) if ep else (obs.index < 2012)]
        ax.plot(o.index, o.values, "o", ms=2.8, color=c, label=lab)
    base = (ctrl if ctrl is not None else dd["rcp26"].median(axis=1))
    ax.fill_between(base.index, base - rmse, base + rmse, color=GRIS, alpha=0.15, lw=0,
                    label=f"± error típico del modelo ({rmse:.1f} ha)")
    for e, c, lab in (("rcp26", C3, "RCP 2.6 (bajas emisiones)"), ("rcp85", "#c0392b", "RCP 8.5 (altas emisiones)")):
        q = dd[e].quantile([0.1, 0.5, 0.9], axis=1).T
        ax.fill_between(q.index, q[0.1], q[0.9], color=c, alpha=0.25, lw=0)
        ax.plot(q.index, q[0.5], color=c, lw=1.4, label=f"{lab}: mediana y p10-p90 de {dd[e].shape[1]} modelos")
    ax.set(title=tit, ylabel="Superficie (ha)")
    ax.legend(fontsize=6.3, loc="upper left", ncol=2)
    ax.set_ylim(top=ax.get_ylim()[1] + 8)
axes[1].set(xlabel="Año")
guardar_fig(fig, "f11_proyeccion.png")

# ============================== 16. Exportación ==============================
for nombre, t in TABLAS.items():
    t.to_csv(OUT / f"{nombre}.csv", index=False)


def a_json(o):
    if isinstance(o, (np.integer,)):
        return int(o)
    if isinstance(o, (np.floating,)):
        return float(o)
    if isinstance(o, (pd.Timestamp,)):
        return str(o.date())
    return str(o)


(OUT / "resultados.json").write_text(json.dumps(R, ensure_ascii=False, indent=1, default=a_json), encoding="utf-8")
print("Listo:", len(TABLAS), "tablas,", len(list(FIG.glob('*.png'))), "figuras")
