# -*- coding: utf-8 -*-
"""
Dashboard Streamlit para consultar el Censo Argentino publicado en Source Cooperative
sin usar el plugin de QGIS.

Versión robusta:
- El selector de variables se arma desde las variables REALES presentes en census-data.parquet.
- Si una variable común de totales, como POB_TOT_P o VIV_TOT_P, no aparece en la tabla larga,
  el dashboard hace fallback a radios.parquet, donde esas columnas existen para 2022.
- Incluye diagnóstico cuando una consulta no devuelve filas.

Ejecutar:
    streamlit run app_censo_streamlit.py
"""

from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path

import numpy as np
from typing import Iterable

import altair as alt
import duckdb
import pandas as pd
import streamlit as st
import requests

try:
    import geopandas as gpd
    import pydeck as pdk
except Exception:  # pragma: no cover
    gpd = None
    pdk = None


# -----------------------------------------------------------------------------
# Configuración general
# -----------------------------------------------------------------------------

st.set_page_config(
    page_title="Censo Argentino · DuckDB + Streamlit",
    page_icon="🇦🇷",
    layout="wide",
)

VALID_YEARS = [1991, 2001, 2010, 2022]
BASE_URL = "https://data.source.coop/nlebovits/censo-argentino/{year}/{filename}"
CACHE_DIR = Path(os.getenv("CENSO_CACHE_DIR", "/tmp/censo_argentino_cache"))
FILES = {
    "census": "census-data.parquet",
    "metadata": "metadata.parquet",
    "radios": "radios.parquet",
}
JOIN_COL_BY_YEAR = {
    1991: "COD_1991",
    2001: "COD_2001",
    2010: "COD_2010",
    2022: "COD_2022",
}

# Estas columnas figuran en radios.parquet para 2022 y suelen ser útiles como consulta inicial.
# En caso de que no estén en census-data.parquet, se consultan directamente desde radios.parquet.
RADIOS_NUMERIC_VARIABLES = {
    "POB_TOT_P": "Población total",
    "VIV_TOT_P": "Viviendas totales",
}

# Catálogo mínimo de provincias como fallback local.
# Los nombres de departamentos se intentan resolver primero con GeoRef.
PROVINCE_NAMES = {
    "02": "Ciudad Autónoma de Buenos Aires",
    "06": "Buenos Aires",
    "10": "Catamarca",
    "14": "Córdoba",
    "18": "Corrientes",
    "22": "Chaco",
    "26": "Chubut",
    "30": "Entre Ríos",
    "34": "Formosa",
    "38": "Jujuy",
    "42": "La Pampa",
    "46": "La Rioja",
    "50": "Mendoza",
    "54": "Misiones",
    "58": "Neuquén",
    "62": "Río Negro",
    "66": "Salta",
    "70": "San Juan",
    "74": "San Luis",
    "78": "Santa Cruz",
    "82": "Santa Fe",
    "86": "Santiago del Estero",
    "90": "Tucumán",
    "94": "Tierra del Fuego, Antártida e Islas del Atlántico Sur",
}

GEOREF_API_BASE = "https://apis.datos.gob.ar/georef/api"

GROUP_MAP_LONG = {
    "Provincia": ["valor_provincia", "etiqueta_provincia"],
    "Departamento": [
        "valor_provincia",
        "etiqueta_provincia",
        "valor_departamento",
        "etiqueta_departamento",
    ],
    "Categoría": ["codigo_variable", "valor_categoria", "etiqueta_categoria"],
    "Radio censal": ["id_geo"],
}

GROUP_MAP_RADIOS = {
    "Provincia": ["PROV"],
    "Departamento": ["PROV", "DEPTO"],
    "Radio censal": [],  # se completa dinámicamente con COD_YYYY
}

# Búsquedas temáticas para descubrir variables reales desde metadata/census-data.
# No se hardcodean códigos porque pueden variar por año/procesamiento.
VARIABLE_THEMES = {
    "General": ["pob", "viv", "hogar"],
    "Población y demografía": ["sexo", "edad", "parentesco", "nacimiento"],
    "Vivienda": ["vivienda", "tipo", "material", "techo", "piso", "pared"],
    "Hogar": ["hogar", "hacinamiento", "habit", "cuarto", "baño"],
    "Servicios básicos": ["agua", "cloaca", "gas", "electric", "combustible", "desague"],
    "Educación": ["educ", "asistencia", "nivel", "alfabet", "sabe leer"],
    "Trabajo y actividad": ["trabajo", "actividad", "ocup", "empleo", "jubil"],
    "Migración": ["nacimiento", "residencia", "migr", "pais"],
    "Discapacidad / dificultad": ["dificultad", "limitacion", "discap", "caminar", "recordar"],
    "Tecnología": ["internet", "computadora", "celular", "telefono"],
}

MEASURE_MODES = [
    "Conteo absoluto",
    "% dentro de la variable",
    "% sobre población total",
    "% sobre viviendas totales",
]


BASE_MAP_STYLES = {
    "Calles y etiquetas (CARTO Voyager)": {
        "provider": "carto",
        "style": "https://basemaps.cartocdn.com/gl/voyager-gl-style/style.json",
        "desc": "Más contexto urbano: calles, rutas, barrios y etiquetas."
    },
    "Claro detallado (CARTO Positron)": {
        "provider": "carto",
        "style": "https://basemaps.cartocdn.com/gl/positron-gl-style/style.json",
        "desc": "Fondo claro, prolijo y útil para coropléticos."
    },
    "Oscuro detallado (CARTO Dark Matter)": {
        "provider": "carto",
        "style": "https://basemaps.cartocdn.com/gl/dark-matter-gl-style/style.json",
        "desc": "Fondo oscuro con buen contraste para capas luminosas."
    },
    "Calles simple": {
        "provider": "carto",
        "style": "road",
        "desc": "Estilo de calles provisto por el proveedor del mapa."
    },
    "Claro sin etiquetas": {
        "provider": "carto",
        "style": "light_no_labels",
        "desc": "Base limpia para mirar sólo la capa censal."
    },
    "Oscuro sin etiquetas": {
        "provider": "carto",
        "style": "dark_no_labels",
        "desc": "Base oscura limpia, sin etiquetas que compitan con el dato."
    },
    "Tema Streamlit": {
        "provider": "carto",
        "style": None,
        "desc": "Deja que Streamlit adapte el estilo al tema activo."
    },
    "Sin mapa base": {
        "provider": None,
        "style": None,
        "desc": "Sólo capas censales. Útil si la base carga lento."
    },
}


# -----------------------------------------------------------------------------
# Utilidades SQL / DuckDB
# -----------------------------------------------------------------------------


def source_url(year: int, table: str) -> str:
    return BASE_URL.format(year=int(year), filename=FILES[table])


def sql_quote(value: object) -> str:
    if value is None:
        return "NULL"
    text = str(value)
    return "'" + text.replace("'", "''") + "'"


@st.cache_resource(show_spinner=False)
def get_connection() -> duckdb.DuckDBPyConnection:
    con = duckdb.connect(database=":memory:")
    # httpfs permite leer Parquet remoto por HTTPS/S3.
    con.execute("INSTALL httpfs;")
    con.execute("LOAD httpfs;")
    con.execute("SET threads TO 4;")
    return con


@st.cache_data(show_spinner=False, ttl=60 * 60)
def run_sql(sql: str) -> pd.DataFrame:
    con = get_connection()
    return con.execute(sql).df()


def ensure_dataframe(value: object, context: str = "") -> pd.DataFrame:
    """Convierte respuestas inesperadas a DataFrame vacío para no romper la UI."""
    if isinstance(value, pd.DataFrame):
        return value
    if value is None:
        return pd.DataFrame()
    try:
        return pd.DataFrame(value)
    except Exception:
        if context:
            st.warning(f"No pude convertir `{context}` a tabla. Se omite ese filtro.")
        return pd.DataFrame()


def build_where_long(
    variable: str | None = None,
    provincia_code: str | None = None,
    departamento_code: str | None = None,
    categoria_value: str | None = None,
) -> str:
    filters: list[str] = []

    if variable:
        filters.append(f"codigo_variable = {sql_quote(variable)}")

    if provincia_code and provincia_code != "__ALL__":
        filters.append(f"valor_provincia = {sql_quote(provincia_code)}")

    if departamento_code and departamento_code != "__ALL__":
        filters.append(f"valor_departamento = {sql_quote(departamento_code)}")

    if categoria_value and categoria_value != "__ALL__":
        filters.append(f"valor_categoria = {sql_quote(categoria_value)}")

    return " AND ".join(filters) if filters else "1 = 1"


def build_where_radios(provincia_code: str | None = None, departamento_code: str | None = None) -> str:
    filters: list[str] = []

    if provincia_code and provincia_code != "__ALL__":
        filters.append(f"{sql_norm_code('PROV', 2)} = LPAD({sql_quote(provincia_code)}, 2, '0')")

    if departamento_code and departamento_code != "__ALL__":
        filters.append(f"{sql_norm_code('DEPTO', 3)} = LPAD({sql_quote(departamento_code)}, 3, '0')")

    return " AND ".join(filters) if filters else "1 = 1"


def infer_col(df: pd.DataFrame, candidates: list[str], contains_any: list[str] | None = None) -> str | None:
    lower_map = {c.lower(): c for c in df.columns}
    for cand in candidates:
        if cand.lower() in lower_map:
            return lower_map[cand.lower()]

    if contains_any:
        for col in df.columns:
            col_lower = col.lower()
            if all(token.lower() in col_lower for token in contains_any):
                return col
    return None


def format_number(value: float | int | None) -> str:
    if value is None or pd.isna(value):
        return "-"
    return f"{float(value):,.0f}".replace(",", ".")


def normalize_code_series(series: pd.Series, width: int) -> pd.Series:
    """Normaliza códigos geográficos que pueden venir como int, string o float serializado."""
    numeric = pd.to_numeric(series, errors="coerce")
    numeric_text = numeric.astype("Int64").astype(str).replace("<NA>", pd.NA)
    text = (
        series.astype(str)
        .str.strip()
        .str.replace(r"\.0$", "", regex=True)
        .replace({"nan": pd.NA, "None": pd.NA, "": pd.NA})
    )
    out = numeric_text.fillna(text).fillna("")
    return out.astype(str).str.zfill(width)


def sql_norm_code(col: str, width: int) -> str:
    """Expresión DuckDB para normalizar códigos geográficos con/sin ceros a la izquierda."""
    return (
        f"CASE "
        f"WHEN TRY_CAST({col} AS BIGINT) IS NOT NULL "
        f"THEN LPAD(CAST(TRY_CAST({col} AS BIGINT) AS VARCHAR), {int(width)}, '0') "
        f"ELSE LPAD(CAST({col} AS VARCHAR), {int(width)}, '0') "
        f"END"
    )


def get_label_column(df: pd.DataFrame) -> str | None:
    candidates = [
        "etiqueta_departamento",
        "etiqueta_provincia",
        "etiqueta_categoria",
        "id_geo",
        "valor_departamento",
        "valor_provincia",
    ]
    return next((c for c in candidates if c in df.columns), None)


def enrich_result(df: pd.DataFrame) -> pd.DataFrame:
    if df.empty or "conteo" not in df.columns:
        return df.copy()
    out = df.copy()
    out["conteo"] = pd.to_numeric(out["conteo"], errors="coerce").fillna(0)
    out = out.sort_values("conteo", ascending=False).reset_index(drop=True)
    total = out["conteo"].sum()
    out["ranking"] = np.arange(1, len(out) + 1)
    out["pct_total"] = np.where(total > 0, out["conteo"] / total, 0)
    out["conteo_acum"] = out["conteo"].cumsum()
    out["pct_acum"] = np.where(total > 0, out["conteo_acum"] / total, 0)
    return out


def _clean_text(value: object) -> str:
    if value is None:
        return ""
    text = str(value).strip()
    return "" if text.lower() in {"", "nan", "none", "<na>"} else text


def humanize_area_labels(
    df: pd.DataFrame,
    group_label: str,
    year: int,
    provincia_code: str | None,
) -> pd.DataFrame:
    """Crea una columna `area_label` legible: 'Nombre (código)' en vez de un código pelado.

    Resuelve nombres faltantes o numéricos cruzando con los catálogos de provincias y
    departamentos (GeoRef + fallback local). Para 'Radio censal' y 'Categoría' arma una
    etiqueta clara sin inventar nombres.
    """
    out = df.copy()
    if out.empty:
        out["area_label"] = pd.Series(dtype=str)
        return out

    if group_label == "Radio censal" and "id_geo" in out.columns:
        out["area_label"] = out["id_geo"].map(_clean_text)
        return out

    if group_label == "Categoría":
        cats = out["etiqueta_categoria"] if "etiqueta_categoria" in out.columns else pd.Series([None] * len(out))
        vals = out["valor_categoria"] if "valor_categoria" in out.columns else pd.Series([None] * len(out))
        labels = []
        for name_raw, code_raw in zip(cats, vals):
            name, code = _clean_text(name_raw), _clean_text(code_raw)
            labels.append(name or (f"Categoría {code}" if code else "Sin categoría"))
        out["area_label"] = labels
        return out

    if group_label == "Provincia" and "valor_provincia" in out.columns:
        prov_df = get_provincias(year)
        prov_names = dict(zip(prov_df["valor_provincia"].astype(str), prov_df["etiqueta_provincia"]))
        labels = []
        for _, row in out.iterrows():
            code = _clean_text(row.get("valor_provincia")).zfill(2)
            name = _clean_text(row.get("etiqueta_provincia"))
            if not name or name.isdigit():
                name = _clean_text(prov_names.get(code)) or PROVINCE_NAMES.get(code, f"Provincia {code}")
            labels.append(f"{name} ({code})")
        out["area_label"] = labels
        return out

    if group_label == "Departamento" and "valor_departamento" in out.columns:
        dept_names: dict[str, str] = {}
        if provincia_code and provincia_code != "__ALL__":
            dd = get_departamentos(year, provincia_code)
            if not dd.empty:
                dept_names = dict(
                    zip(dd["valor_departamento"].astype(str).str.zfill(3), dd["etiqueta_departamento"])
                )
        labels = []
        for _, row in out.iterrows():
            code = _clean_text(row.get("valor_departamento")).zfill(3)
            name = _clean_text(row.get("etiqueta_departamento"))
            if not name or name.isdigit():
                name = _clean_text(dept_names.get(code)) or f"Departamento {code}"
            labels.append(f"{name} ({code})")
        out["area_label"] = labels
        return out

    fallback_col = get_label_column(out)
    out["area_label"] = out[fallback_col].map(_clean_text) if fallback_col else ""
    return out


def measure_value_meta(measure_mode: str) -> tuple[str, str, str]:
    """Título de eje, formato Altair y esquema de color según el modo de medición.

    En los modos porcentuales la columna ``conteo`` ya viene multiplicada por 100
    (p. ej. 23.5 significa 23,5%), por eso se formatea con un decimal y sufijo %.
    """
    if measure_mode == "Conteo absoluto":
        return "Valor", ",.0f", "blues"
    return "Porcentaje (%)", ",.1f", "teals"


# Tooltip homogéneo: evita volcar columnas crudas y respeta el formato de cada métrica.
def _value_tooltips(label_col: str, value_title: str, value_fmt: str) -> list:
    return [
        alt.Tooltip(f"{label_col}:N", title="Área"),
        alt.Tooltip("conteo:Q", title=value_title, format=value_fmt),
        alt.Tooltip("pct_total:Q", title="% del total", format=".1%"),
        alt.Tooltip("ranking:Q", title="Ranking"),
    ]


def build_bar_chart(df: pd.DataFrame, label_col: str, top_n: int, measure_mode: str = "Conteo absoluto"):
    chart_df = enrich_result(df).head(int(top_n))
    value_title, value_fmt, scheme = measure_value_meta(measure_mode)
    base = alt.Chart(chart_df).encode(
        y=alt.Y(f"{label_col}:N", sort="-x", title=""),
        x=alt.X("conteo:Q", title=value_title, axis=alt.Axis(format=value_fmt)),
    )
    bars = base.mark_bar().encode(
        color=alt.Color("conteo:Q", scale=alt.Scale(scheme=scheme), legend=None),
        tooltip=_value_tooltips(label_col, value_title, value_fmt),
    )
    labels = base.mark_text(align="left", dx=4, baseline="middle", color="#333").encode(
        text=alt.Text("conteo:Q", format=value_fmt),
    )
    return (bars + labels).properties(height=max(300, min(850, 26 * len(chart_df))))


def build_pareto_chart(df: pd.DataFrame, label_col: str, top_n: int, measure_mode: str = "Conteo absoluto"):
    chart_df = enrich_result(df).head(int(top_n))
    value_title, value_fmt, scheme = measure_value_meta(measure_mode)
    base = alt.Chart(chart_df).encode(x=alt.X(f"{label_col}:N", sort="-y", title=""))
    bars = base.mark_bar(opacity=0.8).encode(
        y=alt.Y("conteo:Q", title=value_title, axis=alt.Axis(format=value_fmt)),
        color=alt.Color("conteo:Q", scale=alt.Scale(scheme=scheme), legend=None),
        tooltip=_value_tooltips(label_col, value_title, value_fmt)
        + [alt.Tooltip("pct_acum:Q", title="% acumulado", format=".1%")],
    )
    line = base.mark_line(point=True, color="#d62728").encode(
        y=alt.Y("pct_acum:Q", title="% acumulado", axis=alt.Axis(format="%")),
        tooltip=[
            alt.Tooltip(f"{label_col}:N", title="Área"),
            alt.Tooltip("pct_acum:Q", title="% acumulado", format=".1%"),
        ],
    )
    return (bars + line).resolve_scale(y="independent").properties(height=420)


def build_distribution_chart(df: pd.DataFrame, measure_mode: str = "Conteo absoluto"):
    chart_df = enrich_result(df)
    value_title, value_fmt, _ = measure_value_meta(measure_mode)
    return (
        alt.Chart(chart_df)
        .mark_bar(color="#2c7fb8")
        .encode(
            x=alt.X("conteo:Q", bin=alt.Bin(maxbins=35), title=value_title, axis=alt.Axis(format=value_fmt)),
            y=alt.Y("count():Q", title="Cantidad de áreas"),
            tooltip=[
                alt.Tooltip("conteo:Q", bin=alt.Bin(maxbins=35), title=value_title, format=value_fmt),
                alt.Tooltip("count():Q", title="Áreas"),
            ],
        )
        .properties(height=360)
    )


def build_share_chart(df: pd.DataFrame, label_col: str, top_n: int):
    chart_df = enrich_result(df).head(int(top_n)).copy()
    base = alt.Chart(chart_df).encode(
        y=alt.Y(f"{label_col}:N", sort="-x", title=""),
        x=alt.X("pct_total:Q", title="Participación", axis=alt.Axis(format="%")),
    )
    bars = base.mark_bar().encode(
        color=alt.Color("pct_total:Q", scale=alt.Scale(scheme="oranges"), legend=None),
        tooltip=[
            alt.Tooltip(f"{label_col}:N", title="Área"),
            alt.Tooltip("conteo:Q", title="Valor", format=",.1f"),
            alt.Tooltip("pct_total:Q", title="% del total", format=".1%"),
        ],
    )
    labels = base.mark_text(align="left", dx=4, baseline="middle", color="#333").encode(
        text=alt.Text("pct_total:Q", format=".1%"),
    )
    return (bars + labels).properties(height=max(300, min(850, 26 * len(chart_df))))


def prepare_levels_by_area(
    df: pd.DataFrame,
    group_label: str,
    year: int,
    provincia_code: str | None,
    top_n_areas: int,
) -> tuple[pd.DataFrame, list[str]]:
    """Prepara el dataframe área×nivel: agrega `area_label`, `nivel`, % por área y recorta al top N de áreas por total."""
    if df.empty:
        return df.assign(area_label=pd.Series(dtype=str), nivel=pd.Series(dtype=str)), []

    out = humanize_area_labels(df, group_label, year, provincia_code)
    cats = out["etiqueta_categoria"] if "etiqueta_categoria" in out.columns else pd.Series([None] * len(out))
    vals = out["valor_categoria"] if "valor_categoria" in out.columns else pd.Series([None] * len(out))
    out["nivel"] = [
        _clean_text(c) or (f"Categoría {_clean_text(v)}" if _clean_text(v) else "Sin nivel")
        for c, v in zip(cats, vals)
    ]
    out["conteo"] = pd.to_numeric(out["conteo"], errors="coerce").fillna(0)

    totals = out.groupby("area_label")["conteo"].sum().sort_values(ascending=False)
    area_order = list(totals.head(int(top_n_areas)).index)
    out = out[out["area_label"].isin(area_order)].copy()
    out["area_total"] = out["area_label"].map(totals)
    out["pct_area"] = np.where(out["area_total"] > 0, out["conteo"] / out["area_total"], 0)
    return out, area_order


def build_grouped_bar_chart(
    df: pd.DataFrame,
    area_order: list[str],
    measure_mode: str = "Conteo absoluto",
    value_col: str = "conteo",
):
    """Barras agrupadas: cada área abre una barra por nivel, lado a lado."""
    value_title, value_fmt, _ = measure_value_meta(measure_mode)
    height = max(320, min(950, 30 * max(1, len(area_order)) * 1 + 18 * max(1, df["nivel"].nunique())))
    return (
        alt.Chart(df)
        .mark_bar()
        .encode(
            y=alt.Y("area_label:N", sort=area_order, title=""),
            yOffset=alt.YOffset("nivel:N"),
            x=alt.X(f"{value_col}:Q", title=value_title, axis=alt.Axis(format=value_fmt)),
            color=alt.Color("nivel:N", title="Nivel", scale=alt.Scale(scheme="tableau10")),
            tooltip=[
                alt.Tooltip("area_label:N", title="Área"),
                alt.Tooltip("nivel:N", title="Nivel"),
                alt.Tooltip(f"{value_col}:Q", title=value_title, format=value_fmt),
                alt.Tooltip("pct_area:Q", title="% del área", format=".1%"),
            ],
        )
        .properties(height=int(height))
    )


def build_grouped_share_chart(df: pd.DataFrame, area_order: list[str]):
    """Barras agrupadas de composición: % de cada nivel dentro de su área."""
    height = max(320, min(950, 30 * max(1, len(area_order)) + 18 * max(1, df["nivel"].nunique())))
    return (
        alt.Chart(df)
        .mark_bar()
        .encode(
            y=alt.Y("area_label:N", sort=area_order, title=""),
            yOffset=alt.YOffset("nivel:N"),
            x=alt.X("pct_area:Q", title="% dentro del área", axis=alt.Axis(format="%")),
            color=alt.Color("nivel:N", title="Nivel", scale=alt.Scale(scheme="tableau10")),
            tooltip=[
                alt.Tooltip("area_label:N", title="Área"),
                alt.Tooltip("nivel:N", title="Nivel"),
                alt.Tooltip("pct_area:Q", title="% del área", format=".1%"),
                alt.Tooltip("conteo:Q", title="Valor", format=",.0f"),
            ],
        )
        .properties(height=int(height))
    )


def summarize_result(df: pd.DataFrame, label_col: str | None) -> dict[str, object]:
    enriched = enrich_result(df)
    if enriched.empty or "conteo" not in enriched.columns:
        return {
            "total": 0,
            "n_areas": 0,
            "top_label": "-",
            "top_value": 0,
            "top_share": 0,
            "mean_value": 0,
            "median_value": 0,
        }
    top_row = enriched.iloc[0]
    return {
        "total": float(enriched["conteo"].sum()),
        "n_areas": int(len(enriched)),
        "top_label": str(top_row[label_col]) if label_col and label_col in enriched.columns else "Top 1",
        "top_value": float(top_row["conteo"]),
        "top_share": float(top_row["pct_total"]),
        "mean_value": float(enriched["conteo"].mean()),
        "median_value": float(enriched["conteo"].median()),
    }


# -----------------------------------------------------------------------------
# Carga de catálogos
# -----------------------------------------------------------------------------


@st.cache_data(show_spinner=False, ttl=60 * 60)
def get_schema(year: int, table: str) -> pd.DataFrame:
    url = source_url(year, table)
    return run_sql(f"DESCRIBE SELECT * FROM read_parquet({sql_quote(url)})")


@st.cache_data(show_spinner=False, ttl=60 * 60)
def get_metadata(year: int) -> pd.DataFrame:
    url = source_url(year, "metadata")
    return run_sql(f"SELECT * FROM read_parquet({sql_quote(url)})")


@st.cache_data(show_spinner=False, ttl=60 * 60)
def get_census_variables(year: int, search: str | None = None, limit: int = 500) -> pd.DataFrame:
    """Devuelve variables realmente presentes en census-data.parquet, enriquecidas con metadata."""
    census_url = source_url(year, "census")
    metadata_url = source_url(year, "metadata")
    term = (search or "").strip()

    where = ""
    if term:
        quoted_like = sql_quote(f"%{term}%")
        where = f"""
        WHERE lower(v.codigo_variable) LIKE lower({quoted_like})
           OR lower(COALESCE(m.etiqueta_variable, '')) LIKE lower({quoted_like})
        """

    sql = f"""
    WITH vars AS (
        SELECT DISTINCT codigo_variable
        FROM read_parquet({sql_quote(census_url)})
        WHERE codigo_variable IS NOT NULL
    ), meta AS (
        SELECT
            codigo_variable,
            any_value(etiqueta_variable) AS etiqueta_variable
        FROM read_parquet({sql_quote(metadata_url)})
        WHERE codigo_variable IS NOT NULL
        GROUP BY 1
    )
    SELECT
        v.codigo_variable,
        COALESCE(m.etiqueta_variable, '') AS etiqueta_variable,
        'census-data' AS fuente
    FROM vars v
    LEFT JOIN meta m USING (codigo_variable)
    {where}
    ORDER BY v.codigo_variable
    LIMIT {int(limit)}
    """

    df = run_sql(sql)

    # Agregamos variables útiles desde radios.parquet, para que POB_TOT_P/VIV_TOT_P funcionen
    # aunque no aparezcan en la tabla larga del año elegido.
    radios_rows = []
    for code, label in RADIOS_NUMERIC_VARIABLES.items():
        if not term or term.lower() in code.lower() or term.lower() in label.lower():
            if code not in set(df.get("codigo_variable", pd.Series(dtype=str)).astype(str)):
                radios_rows.append(
                    {
                        "codigo_variable": code,
                        "etiqueta_variable": label,
                        "fuente": "radios.parquet",
                    }
                )
    if radios_rows:
        df = pd.concat([pd.DataFrame(radios_rows), df], ignore_index=True)

    return df


@st.cache_data(show_spinner=False, ttl=60 * 60)
def get_georef_provincias() -> pd.DataFrame:
    """Catálogo de provincias desde GeoRef, con fallback local."""
    try:
        resp = requests.get(
            f"{GEOREF_API_BASE}/provincias",
            params={"campos": "id,nombre", "max": 100},
            timeout=12,
        )
        resp.raise_for_status()
        rows = resp.json().get("provincias", [])
        df = pd.DataFrame(rows)
        if not df.empty and {"id", "nombre"}.issubset(df.columns):
            df["valor_provincia"] = df["id"].astype(str).str.zfill(2)
            df["etiqueta_provincia"] = df["nombre"].astype(str).str.title()
            return df[["valor_provincia", "etiqueta_provincia"]].drop_duplicates()
    except Exception:
        pass

    return pd.DataFrame(
        [
            {"valor_provincia": code, "etiqueta_provincia": name}
            for code, name in PROVINCE_NAMES.items()
        ]
    )


@st.cache_data(show_spinner=False, ttl=60 * 60)
def get_georef_departamentos(provincia_code: str | None) -> pd.DataFrame:
    """Departamentos oficiales desde GeoRef. Devuelve DEPTO de 3 dígitos para cruzar con radios.parquet."""
    if not provincia_code or provincia_code == "__ALL__":
        return pd.DataFrame(columns=["prov_norm", "depto_norm", "valor_departamento", "etiqueta_departamento"])

    prov_norm = str(provincia_code).zfill(2)
    try:
        resp = requests.get(
            f"{GEOREF_API_BASE}/departamentos",
            params={"provincia": prov_norm, "campos": "id,nombre,provincia", "max": 1000},
            timeout=15,
        )
        resp.raise_for_status()
        rows = resp.json().get("departamentos", [])
        df = pd.DataFrame(rows)
        if not df.empty and {"id", "nombre"}.issubset(df.columns):
            df["id"] = df["id"].astype(str).str.zfill(5)
            df["prov_norm"] = df["id"].str[:2]
            df["depto_norm"] = df["id"].str[-3:]
            df["valor_departamento"] = df["depto_norm"]
            df["etiqueta_departamento"] = df["nombre"].astype(str).str.title()
            return df[["prov_norm", "depto_norm", "valor_departamento", "etiqueta_departamento"]].drop_duplicates()
    except Exception:
        pass

    return pd.DataFrame(columns=["prov_norm", "depto_norm", "valor_departamento", "etiqueta_departamento"])


@st.cache_data(show_spinner=False, ttl=60 * 60)
def get_provincias(year: int) -> pd.DataFrame:
    """Provincias presentes en radios.parquet, enriquecidas con nombres de GeoRef/fallback."""
    radios_url = source_url(year, "radios")
    try:
        codes = run_sql(
            f"""
            SELECT DISTINCT {sql_norm_code('PROV', 2)} AS valor_provincia
            FROM read_parquet({sql_quote(radios_url)})
            WHERE PROV IS NOT NULL
            ORDER BY 1
            """
        )
    except Exception:
        codes = pd.DataFrame({"valor_provincia": list(PROVINCE_NAMES.keys())})

    names = get_georef_provincias()
    out = codes.merge(names, on="valor_provincia", how="left")
    out["etiqueta_provincia"] = out["etiqueta_provincia"].fillna(
        out["valor_provincia"].map(PROVINCE_NAMES)
    )
    out["etiqueta_provincia"] = out["etiqueta_provincia"].fillna("Provincia " + out["valor_provincia"].astype(str))
    return out[["valor_provincia", "etiqueta_provincia"]].drop_duplicates().sort_values("etiqueta_provincia")


@st.cache_data(show_spinner=False, ttl=60 * 60)
def get_departamentos(year: int, provincia_code: str | None) -> pd.DataFrame:
    """Departamentos presentes en radios.parquet, con nombres de GeoRef cuando están disponibles."""
    if not provincia_code or provincia_code == "__ALL__":
        return pd.DataFrame(columns=["valor_departamento", "etiqueta_departamento"])

    radios_url = source_url(year, "radios")
    where = build_where_radios(provincia_code=provincia_code)
    try:
        codes = run_sql(
            f"""
            SELECT DISTINCT
                {sql_norm_code('PROV', 2)} AS prov_norm,
                {sql_norm_code('DEPTO', 3)} AS depto_norm
            FROM read_parquet({sql_quote(radios_url)})
            WHERE {where}
              AND DEPTO IS NOT NULL
            ORDER BY 1, 2
            """
        )
    except Exception:
        codes = pd.DataFrame(columns=["prov_norm", "depto_norm"])

    georef = get_georef_departamentos(provincia_code)
    if codes.empty:
        out = georef.copy()
    else:
        out = codes.merge(georef, on=["prov_norm", "depto_norm"], how="left")

    if out.empty:
        return pd.DataFrame(columns=["valor_departamento", "etiqueta_departamento"])

    out["valor_departamento"] = out.get("valor_departamento", out["depto_norm"]).fillna(out["depto_norm"])
    out["etiqueta_departamento"] = out.get("etiqueta_departamento", pd.Series(index=out.index, dtype=str))
    out["etiqueta_departamento"] = out["etiqueta_departamento"].fillna("Departamento " + out["depto_norm"].astype(str))

    return (
        out[["valor_departamento", "etiqueta_departamento"]]
        .drop_duplicates()
        .sort_values("etiqueta_departamento")
        .reset_index(drop=True)
    )


@st.cache_data(show_spinner=False, ttl=60 * 60)
def get_geo_names(year: int) -> pd.DataFrame:
    """Dimensión geográfica con etiquetas de provincia/departamento desde census-data.parquet."""
    url = source_url(year, "census")
    sql = f"""
    SELECT
        {sql_norm_code('valor_provincia', 2)} AS prov_norm,
        any_value(valor_provincia) AS valor_provincia,
        any_value(etiqueta_provincia) AS etiqueta_provincia,
        {sql_norm_code('valor_departamento', 3)} AS depto_norm,
        any_value(valor_departamento) AS valor_departamento,
        any_value(etiqueta_departamento) AS etiqueta_departamento
    FROM read_parquet({sql_quote(url)})
    WHERE valor_provincia IS NOT NULL
      AND etiqueta_provincia IS NOT NULL
      AND valor_departamento IS NOT NULL
      AND etiqueta_departamento IS NOT NULL
    GROUP BY 1, 4
    """
    return run_sql(sql)


@st.cache_data(show_spinner=False, ttl=60 * 60)
def get_categorias(year: int, variable: str | None) -> pd.DataFrame:
    if not variable:
        return pd.DataFrame(columns=["valor_categoria", "etiqueta_categoria"])

    # Las variables provenientes de radios.parquet no tienen categorías.
    if variable in RADIOS_NUMERIC_VARIABLES:
        return pd.DataFrame(columns=["valor_categoria", "etiqueta_categoria"])

    url = source_url(year, "census")
    sql = f"""
    SELECT DISTINCT
        valor_categoria,
        etiqueta_categoria
    FROM read_parquet({sql_quote(url)})
    WHERE codigo_variable = {sql_quote(variable)}
      AND valor_categoria IS NOT NULL
    ORDER BY valor_categoria, etiqueta_categoria
    """
    return run_sql(sql)


@st.cache_data(show_spinner=False, ttl=60 * 60)
def get_variable_label(year: int, variable: str | None) -> str:
    """Descripción legible de la variable (etiqueta_variable de metadata.parquet)."""
    if not variable:
        return ""
    if variable in RADIOS_NUMERIC_VARIABLES:
        return RADIOS_NUMERIC_VARIABLES[variable]

    url = source_url(year, "metadata")
    try:
        df = run_sql(
            f"""
            SELECT any_value(etiqueta_variable) AS etiqueta
            FROM read_parquet({sql_quote(url)})
            WHERE codigo_variable = {sql_quote(variable)}
            """
        )
    except Exception:
        return ""
    if df.empty:
        return ""
    value = df.iloc[0].get("etiqueta")
    return "" if pd.isna(value) else str(value).strip()


# -----------------------------------------------------------------------------
# Consultas principales
# -----------------------------------------------------------------------------


@st.cache_data(show_spinner=True, ttl=60 * 30)
def query_censo_long(
    year: int,
    variable: str,
    provincia_code: str | None,
    departamento_code: str | None,
    categoria_value: str | None,
    group_label: str,
    limit: int,
) -> pd.DataFrame:
    url = source_url(year, "census")
    group_cols = GROUP_MAP_LONG[group_label]
    where = build_where_long(
        variable=variable,
        provincia_code=provincia_code,
        departamento_code=departamento_code,
        categoria_value=categoria_value,
    )

    select_cols = ",\n        ".join(group_cols)
    group_by = ", ".join(str(i + 1) for i in range(len(group_cols)))
    limit_clause = f"LIMIT {int(limit)}" if limit else ""

    sql = f"""
    SELECT
        {select_cols},
        SUM(conteo) AS conteo
    FROM read_parquet({sql_quote(url)})
    WHERE {where}
    GROUP BY {group_by}
    ORDER BY conteo DESC
    {limit_clause}
    """
    return run_sql(sql)


# Geografía por nivel: columnas usadas para abrir cada área en sus categorías.
LEVEL_GEO_COLS = {
    "Provincia": ["valor_provincia", "etiqueta_provincia"],
    "Departamento": ["valor_provincia", "etiqueta_provincia", "valor_departamento", "etiqueta_departamento"],
    "Radio censal": ["id_geo"],
}


@st.cache_data(show_spinner=True, ttl=60 * 30)
def query_censo_levels_by_area(
    year: int,
    variable: str,
    provincia_code: str | None,
    departamento_code: str | None,
    group_label: str,
) -> pd.DataFrame:
    """Devuelve área + nivel (categoría) + conteo para gráficos agrupados por nivel.

    No aplica filtro de categoría: trae todos los niveles. El recorte de 'top N áreas'
    se resuelve después en pandas para no cortar niveles a la mitad.
    """
    geo_cols = LEVEL_GEO_COLS.get(group_label)
    if not geo_cols:
        return pd.DataFrame()

    url = source_url(year, "census")
    cols = geo_cols + ["valor_categoria", "etiqueta_categoria"]
    select_cols = ",\n        ".join(cols)
    group_by = ", ".join(str(i + 1) for i in range(len(cols)))
    where = build_where_long(
        variable=variable,
        provincia_code=provincia_code,
        departamento_code=departamento_code,
    )
    sql = f"""
    SELECT
        {select_cols},
        SUM(conteo) AS conteo
    FROM read_parquet({sql_quote(url)})
    WHERE {where}
      AND valor_categoria IS NOT NULL
    GROUP BY {group_by}
    ORDER BY conteo DESC
    """
    return run_sql(sql)


@st.cache_data(show_spinner=True, ttl=60 * 30)
def query_censo_radios(
    year: int,
    variable: str,
    provincia_code: str | None,
    departamento_code: str | None,
    group_label: str,
    limit: int,
) -> pd.DataFrame:
    """Consulta variables que viven como columnas en radios.parquet y agrega nombres legibles con GeoRef."""
    if variable not in RADIOS_NUMERIC_VARIABLES:
        return pd.DataFrame()
    if group_label == "Categoría":
        return pd.DataFrame()

    radios_url = source_url(year, "radios")
    join_col = JOIN_COL_BY_YEAR[year]
    where = build_where_radios(provincia_code, departamento_code)
    limit_clause = f"LIMIT {int(limit)}" if limit else ""

    if group_label == "Provincia":
        sql = f"""
        SELECT
            {sql_norm_code('PROV', 2)} AS valor_provincia,
            SUM({variable}) AS conteo
        FROM read_parquet({sql_quote(radios_url)})
        WHERE {where}
        GROUP BY 1
        ORDER BY conteo DESC
        {limit_clause}
        """
        df = run_sql(sql)
        names = get_georef_provincias()
        df = df.merge(names, on="valor_provincia", how="left")
        df["etiqueta_provincia"] = df["etiqueta_provincia"].fillna(
            df["valor_provincia"].map(PROVINCE_NAMES)
        )
        df["etiqueta_provincia"] = df["etiqueta_provincia"].fillna("Provincia " + df["valor_provincia"].astype(str))
        return df[["valor_provincia", "etiqueta_provincia", "conteo"]]

    if group_label == "Departamento":
        sql = f"""
        SELECT
            {sql_norm_code('PROV', 2)} AS prov_norm,
            {sql_norm_code('DEPTO', 3)} AS depto_norm,
            SUM({variable}) AS conteo
        FROM read_parquet({sql_quote(radios_url)})
        WHERE {where}
        GROUP BY 1, 2
        ORDER BY conteo DESC
        {limit_clause}
        """
        df = run_sql(sql)
        prov_names = get_georef_provincias().rename(
            columns={"valor_provincia": "prov_norm", "etiqueta_provincia": "etiqueta_provincia"}
        )
        dept_names = get_georef_departamentos(provincia_code)
        df = df.merge(prov_names, on="prov_norm", how="left")
        if not dept_names.empty:
            df = df.merge(dept_names, on=["prov_norm", "depto_norm"], how="left")
        else:
            df["valor_departamento"] = df["depto_norm"]
            df["etiqueta_departamento"] = None

        df["valor_provincia"] = df["prov_norm"]
        df["etiqueta_provincia"] = df["etiqueta_provincia"].fillna(df["prov_norm"].map(PROVINCE_NAMES))
        df["etiqueta_provincia"] = df["etiqueta_provincia"].fillna("Provincia " + df["prov_norm"].astype(str))
        df["valor_departamento"] = df["valor_departamento"].fillna(df["depto_norm"])
        df["etiqueta_departamento"] = df["etiqueta_departamento"].fillna("Departamento " + df["depto_norm"].astype(str))
        return df[["valor_provincia", "etiqueta_provincia", "valor_departamento", "etiqueta_departamento", "conteo"]]

    if group_label == "Radio censal":
        sql = f"""
        SELECT
            {join_col} AS id_geo,
            SUM({variable}) AS conteo
        FROM read_parquet({sql_quote(radios_url)})
        WHERE {where}
        GROUP BY 1
        ORDER BY conteo DESC
        {limit_clause}
        """
        return run_sql(sql)

    return pd.DataFrame()


def _common_key_columns(left: pd.DataFrame, right: pd.DataFrame) -> list[str]:
    excluded = {"conteo", "conteo_abs", "denominador", "pct_variable", "valor_medido"}
    return [c for c in left.columns if c in right.columns and c not in excluded]


def _apply_measure_from_denominator(
    df_base: pd.DataFrame,
    denom_df: pd.DataFrame | None,
    denom_scalar: float | None,
    measure_mode: str,
) -> pd.DataFrame:
    """Convierte conteos absolutos a la medición elegida manteniendo trazabilidad."""
    out = df_base.copy()
    if out.empty or "conteo" not in out.columns:
        return out

    out["conteo_abs"] = pd.to_numeric(out["conteo"], errors="coerce").fillna(0)

    if measure_mode == "Conteo absoluto":
        out["valor_medido"] = out["conteo_abs"]
        out["tipo_medicion"] = measure_mode
        return out

    if denom_df is not None and not denom_df.empty:
        denom = denom_df.copy()
        denom["denominador"] = pd.to_numeric(denom["conteo"], errors="coerce").fillna(0)
        keys = _common_key_columns(out, denom)
        if keys:
            out = out.merge(denom[keys + ["denominador"]], on=keys, how="left")
        else:
            out["denominador"] = float(denom["denominador"].sum())
    else:
        out["denominador"] = float(denom_scalar or 0)

    out["denominador"] = pd.to_numeric(out["denominador"], errors="coerce").fillna(0)
    out["pct_variable"] = np.where(out["denominador"] > 0, out["conteo_abs"] / out["denominador"], 0)
    out["valor_medido"] = out["pct_variable"] * 100
    out["conteo"] = out["valor_medido"]
    out["tipo_medicion"] = measure_mode
    return out


@st.cache_data(show_spinner=True, ttl=60 * 30)
def query_denominator_radios_scalar(
    year: int,
    denom_variable: str,
    provincia_code: str | None,
    departamento_code: str | None,
) -> float:
    url = source_url(year, "radios")
    where = build_where_radios(provincia_code, departamento_code)
    sql = f"""
    SELECT SUM({denom_variable}) AS denominador
    FROM read_parquet({sql_quote(url)})
    WHERE {where}
    """
    df = run_sql(sql)
    if df.empty:
        return 0.0
    value = pd.to_numeric(df.iloc[0].get("denominador", 0), errors="coerce")
    return 0.0 if pd.isna(value) else float(value)


def query_with_measure_mode(
    year: int,
    variable: str,
    provincia_code: str | None,
    departamento_code: str | None,
    categoria_value: str | None,
    group_label: str,
    limit: int,
    measure_mode: str,
    source_kind: str,
) -> pd.DataFrame:
    """Ejecuta la consulta base y agrega métricas porcentuales si corresponde."""
    if source_kind == "radios":
        df_base = query_censo_radios(year, variable, provincia_code, departamento_code, group_label, limit)
    else:
        df_base = query_censo_long(year, variable, provincia_code, departamento_code, categoria_value, group_label, limit)

    if df_base.empty or measure_mode == "Conteo absoluto":
        return _apply_measure_from_denominator(df_base, None, None, measure_mode)

    if measure_mode == "% dentro de la variable":
        if source_kind == "radios":
            # Para variables directas de radios, el porcentaje dentro de sí misma no agrega información.
            return _apply_measure_from_denominator(df_base, df_base, None, measure_mode)
        if group_label == "Categoría":
            denom_total = float(query_censo_long(year, variable, provincia_code, departamento_code, "__ALL__", "Categoría", 0)["conteo"].sum())
            return _apply_measure_from_denominator(df_base, None, denom_total, measure_mode)
        denom_df = query_censo_long(year, variable, provincia_code, departamento_code, "__ALL__", group_label, 0)
        return _apply_measure_from_denominator(df_base, denom_df, None, measure_mode)

    if measure_mode in {"% sobre población total", "% sobre viviendas totales"}:
        denom_var = "POB_TOT_P" if measure_mode == "% sobre población total" else "VIV_TOT_P"
        if group_label == "Categoría":
            denom_total = query_denominator_radios_scalar(year, denom_var, provincia_code, departamento_code)
            return _apply_measure_from_denominator(df_base, None, denom_total, measure_mode)
        denom_df = query_censo_radios(year, denom_var, provincia_code, departamento_code, group_label, 0)
        return _apply_measure_from_denominator(df_base, denom_df, None, measure_mode)

    return _apply_measure_from_denominator(df_base, None, None, "Conteo absoluto")


@st.cache_data(show_spinner=True, ttl=60 * 30)
def query_censo_resilient(
    year: int,
    variable: str,
    provincia_code: str | None,
    departamento_code: str | None,
    categoria_value: str | None,
    group_label: str,
    limit: int,
    measure_mode: str = "Conteo absoluto",
) -> tuple[pd.DataFrame, str]:
    """Consulta principal con soporte para conteos y porcentajes."""
    # Para variables de radios, conviene ir directo al archivo geográfico si no hay categoría.
    if variable in RADIOS_NUMERIC_VARIABLES and (not categoria_value or categoria_value == "__ALL__"):
        df_radios = query_with_measure_mode(
            year, variable, provincia_code, departamento_code, categoria_value, group_label, limit, measure_mode, "radios"
        )
        if not df_radios.empty:
            return df_radios, f"radios.parquet · {measure_mode}"

    df_long = query_with_measure_mode(
        year, variable, provincia_code, departamento_code, categoria_value, group_label, limit, measure_mode, "census"
    )
    if not df_long.empty:
        return df_long, f"census-data.parquet · {measure_mode}"

    # Fallback final para POB_TOT_P / VIV_TOT_P si la primera consulta fue vacía.
    if variable in RADIOS_NUMERIC_VARIABLES and (not categoria_value or categoria_value == "__ALL__"):
        df_radios = query_with_measure_mode(
            year, variable, provincia_code, departamento_code, categoria_value, group_label, limit, measure_mode, "radios"
        )
        if not df_radios.empty:
            return df_radios, f"radios.parquet · {measure_mode}"

    return df_long, f"census-data.parquet · {measure_mode}"


@st.cache_data(show_spinner=True, ttl=60 * 30)
def query_radio_counts(
    year: int,
    variable: str,
    provincia_code: str | None,
    departamento_code: str | None,
    categoria_value: str | None,
    measure_mode: str = "Conteo absoluto",
) -> tuple[pd.DataFrame, str]:
    """Conteo o indicador a nivel radio para mapa."""
    source_kind = "census"
    if variable in RADIOS_NUMERIC_VARIABLES and (not categoria_value or categoria_value == "__ALL__"):
        url = source_url(year, "radios")
        join_col = JOIN_COL_BY_YEAR[year]
        where = build_where_radios(provincia_code, departamento_code)
        sql = f"""
        SELECT
            {join_col} AS id_geo,
            SUM({variable}) AS conteo
        FROM read_parquet({sql_quote(url)})
        WHERE {where}
        GROUP BY 1
        """
        df_base = run_sql(sql)
        source_kind = "radios"
        source_name = "radios.parquet"
    else:
        url = source_url(year, "census")
        where = build_where_long(
            variable=variable,
            provincia_code=provincia_code,
            departamento_code=departamento_code,
            categoria_value=categoria_value,
        )
        sql = f"""
        SELECT
            id_geo,
            SUM(conteo) AS conteo
        FROM read_parquet({sql_quote(url)})
        WHERE {where}
        GROUP BY 1
        """
        df_base = run_sql(sql)
        source_name = "census-data.parquet"

    if df_base.empty or measure_mode == "Conteo absoluto":
        return _apply_measure_from_denominator(df_base, None, None, measure_mode), f"{source_name} · {measure_mode}"

    if measure_mode == "% dentro de la variable":
        if source_kind == "radios":
            return _apply_measure_from_denominator(df_base, df_base, None, measure_mode), f"{source_name} · {measure_mode}"
        denom_df = query_censo_long(year, variable, provincia_code, departamento_code, "__ALL__", "Radio censal", 0)
        return _apply_measure_from_denominator(df_base, denom_df, None, measure_mode), f"{source_name} · {measure_mode}"

    if measure_mode in {"% sobre población total", "% sobre viviendas totales"}:
        denom_var = "POB_TOT_P" if measure_mode == "% sobre población total" else "VIV_TOT_P"
        denom_df = query_censo_radios(year, denom_var, provincia_code, departamento_code, "Radio censal", 0)
        return _apply_measure_from_denominator(df_base, denom_df, None, measure_mode), f"{source_name} · {measure_mode}"

    return _apply_measure_from_denominator(df_base, None, None, "Conteo absoluto"), source_name


@st.cache_data(show_spinner=False, ttl=60 * 30)
def diagnose_query(
    year: int,
    variable: str,
    provincia_code: str | None,
    departamento_code: str | None,
    categoria_value: str | None,
) -> dict[str, pd.DataFrame]:
    census_url = source_url(year, "census")
    where_geo = build_where_long(
        provincia_code=provincia_code,
        departamento_code=departamento_code,
        categoria_value=categoria_value,
    )

    exact_sql = f"""
    SELECT
        COUNT(*) AS filas,
        SUM(conteo) AS conteo_total
    FROM read_parquet({sql_quote(census_url)})
    WHERE codigo_variable = {sql_quote(variable)}
      AND {where_geo}
    """

    token = variable.strip()[:12]
    like = sql_quote(f"%{token}%") if token else sql_quote("%%")
    suggestions_sql = f"""
    SELECT DISTINCT codigo_variable
    FROM read_parquet({sql_quote(census_url)})
    WHERE lower(codigo_variable) LIKE lower({like})
    ORDER BY codigo_variable
    LIMIT 30
    """

    return {
        "exacto": run_sql(exact_sql),
        "sugerencias": run_sql(suggestions_sql),
    }


# -----------------------------------------------------------------------------
# Mapa
# -----------------------------------------------------------------------------


def find_column_case_insensitive(columns: Iterable[str], wanted: str) -> str | None:
    wanted_lower = wanted.lower()
    for col in columns:
        if col.lower() == wanted_lower:
            return col
    return None


def unique_existing_columns(candidate_cols: Iterable[str], available_cols: Iterable[str]) -> list[str]:
    """Devuelve columnas existentes, sin repetir.

    GeoPandas falla al exportar GeoJSON si el GeoDataFrame seleccionado
    contiene nombres duplicados. Esto puede pasar cuando metric_col == value_col
    o después de algunos merges.
    """
    available = set(available_cols)
    seen = set()
    out = []
    for col in candidate_cols:
        if col in available and col not in seen:
            out.append(col)
            seen.add(col)
    return out


def drop_duplicated_columns_keep_first(df):
    """Elimina columnas duplicadas preservando la primera ocurrencia."""
    if len(df.columns) == len(set(df.columns)):
        return df
    return df.loc[:, ~pd.Index(df.columns).duplicated()].copy()


@st.cache_resource(show_spinner=False)
def download_remote_parquet(url: str) -> str:
    """Descarga un Parquet remoto a /tmp para que GeoPandas/PyArrow no dependan de HTTPS FS."""
    CACHE_DIR.mkdir(parents=True, exist_ok=True)
    filename = hashlib.sha1(url.encode("utf-8")).hexdigest() + ".parquet"
    path = CACHE_DIR / filename
    tmp_path = CACHE_DIR / (filename + ".tmp")

    if path.exists() and path.stat().st_size > 0:
        return str(path)

    with requests.get(url, stream=True, timeout=(15, 300)) as response:
        response.raise_for_status()
        with open(tmp_path, "wb") as fh:
            for chunk in response.iter_content(chunk_size=1024 * 1024):
                if chunk:
                    fh.write(chunk)
    tmp_path.replace(path)
    return str(path)


@st.cache_data(show_spinner=True, ttl=60 * 60)
def load_radios_geoparquet(year: int, provincia_code: str | None, departamento_code: str | None):
    if gpd is None:
        raise RuntimeError(
            "Faltan dependencias geográficas. Instalá: pip install geopandas pyogrio shapely pydeck"
        )

    url = source_url(year, "radios")
    local_path = download_remote_parquet(url)
    join_col = JOIN_COL_BY_YEAR[year]

    filters = []
    if provincia_code and provincia_code != "__ALL__":
        filters.append(("PROV", "=", str(provincia_code).zfill(2)))
    if departamento_code and departamento_code != "__ALL__":
        filters.append(("DEPTO", "=", str(departamento_code).zfill(3)))

    columns = [join_col, "PROV", "DEPTO", "geometry"]

    try:
        if filters:
            gdf = gpd.read_parquet(local_path, columns=columns, filters=filters)
        else:
            gdf = gpd.read_parquet(local_path, columns=columns)
    except Exception:
        # Fallback: lectura completa local. No volvemos a pedir el archivo por HTTPS.
        gdf = gpd.read_parquet(local_path)

    join_col_actual = find_column_case_insensitive(gdf.columns, join_col)
    if join_col_actual is None:
        raise RuntimeError(f"No encontré la columna de join {join_col} en radios.parquet")

    if provincia_code and provincia_code != "__ALL__" and "PROV" in gdf.columns:
        gdf = gdf[normalize_code_series(gdf["PROV"], 2) == str(provincia_code).zfill(2)]
    if departamento_code and departamento_code != "__ALL__" and "DEPTO" in gdf.columns:
        gdf = gdf[normalize_code_series(gdf["DEPTO"], 3) == str(departamento_code).zfill(3)]

    if join_col_actual != join_col:
        gdf = gdf.rename(columns={join_col_actual: join_col})

    gdf[join_col] = gdf[join_col].astype(str)
    return gdf



def _interpolate_rgb(c1: list[int], c2: list[int], t: float) -> list[int]:
    t = max(0.0, min(1.0, float(t)))
    return [int(round(c1[i] + (c2[i] - c1[i]) * t)) for i in range(3)]


def rgba_to_hex(color: list[int]) -> str:
    if not color or len(color) < 3:
        return "#000000"
    return "#" + "".join(f"{max(0, min(255, int(v))):02x}" for v in color[:3])


def palette_color(norm: float, palette: str = "Azul", alpha: int = 165) -> list[int]:
    """Devuelve colores RGBA para PyDeck sin depender de librerías externas."""
    palettes = {
        "Azul": ([232, 244, 255], [82, 151, 220], [7, 59, 138]),
        "Verde": ([235, 250, 238], [87, 181, 121], [12, 104, 61]),
        "Naranja/Rojo": ([255, 244, 224], [244, 151, 67], [175, 35, 35]),
        "Violeta": ([246, 238, 255], [148, 107, 214], [78, 42, 132]),
        "Gris/Azul": ([238, 241, 245], [116, 151, 182], [34, 66, 94]),
    }
    low, mid, high = palettes.get(palette, palettes["Azul"])
    x = max(0.0, min(1.0, float(norm) if pd.notna(norm) else 0.0))
    if x <= 0.5:
        rgb = _interpolate_rgb(low, mid, x * 2)
    else:
        rgb = _interpolate_rgb(mid, high, (x - 0.5) * 2)
    return rgb + [int(alpha)]


def compute_norm_series(values: pd.Series, method: str = "Continuo", clip_upper_pct: int = 98) -> pd.Series:
    """Normaliza valores para color, con opción de percentiles para reducir outliers."""
    s = pd.to_numeric(values, errors="coerce").fillna(0).astype(float)
    if s.empty:
        return s

    if method == "Cuantiles":
        return s.rank(method="average", pct=True).fillna(0).clip(0, 1)

    upper_pct = max(50, min(100, int(clip_upper_pct)))
    upper = float(s.quantile(upper_pct / 100)) if len(s) > 1 else float(s.max())
    lower = float(s.min())
    if upper <= lower:
        upper = float(s.max())
    denom = max(upper - lower, 1.0)
    return ((s.clip(lower=lower, upper=upper) - lower) / denom).fillna(0).clip(0, 1)


def get_base_map_config(base_map_style: str):
    """Devuelve proveedor y estilo de mapa base para PyDeck.

Los estilos de CARTO no requieren token en este uso. La opción "Sin mapa base"
renderiza sólo las capas censales.
    """
    cfg = BASE_MAP_STYLES.get(base_map_style, BASE_MAP_STYLES["Calles y etiquetas (CARTO Voyager)"])
    return cfg.get("provider"), cfg.get("style"), cfg.get("desc", "")


def format_small_float(value: float | int | None, decimals: int = 2) -> str:
    if value is None or pd.isna(value):
        return "-"
    text = f"{float(value):,.{decimals}f}"
    return text.replace(",", "X").replace(".", ",").replace("X", ".")


def compute_view_state(gdf, extruded: bool = False):
    """Centra el mapa usando bounds y calcula un zoom aproximado según extensión."""
    bounds = gdf.total_bounds  # minx, miny, maxx, maxy
    minx, miny, maxx, maxy = [float(x) for x in bounds]
    longitude = (minx + maxx) / 2
    latitude = (miny + maxy) / 2
    span = max(abs(maxx - minx), abs(maxy - miny), 0.001)

    if span < 0.03:
        zoom = 12
    elif span < 0.08:
        zoom = 11
    elif span < 0.18:
        zoom = 10
    elif span < 0.45:
        zoom = 9
    elif span < 1.0:
        zoom = 8
    elif span < 2.5:
        zoom = 7
    else:
        zoom = 5.5

    return pdk.ViewState(
        latitude=latitude,
        longitude=longitude,
        zoom=zoom,
        pitch=45 if extruded else 0,
        bearing=0,
    )


def build_choropleth_map(
    gdf,
    year: int,
    value_col: str = "conteo",
    simplify_tolerance: float = 0.0002,
    metric_mode: str = "Valor total",
    color_method: str = "Continuo",
    color_palette: str = "Azul",
    clip_upper_pct: int = 98,
    polygon_opacity: int = 165,
    border_opacity: int = 100,
    border_width: float = 0.4,
    show_polygons: bool = True,
    show_bubbles: bool = False,
    show_heatmap: bool = False,
    show_labels: bool = False,
    show_outline_layer: bool = True,
    show_centroid_reference: bool = False,
    label_top_n: int = 20,
    label_mode: str = "Valor",
    bubble_scale: float = 1.0,
    heatmap_radius: int = 45,
    extruded: bool = False,
    base_map_style: str = "Calles y etiquetas (CARTO Voyager)",
    map_height: int = 680,
    variable_label: str = "",
    nivel_label: str = "",
):
    if pdk is None:
        raise RuntimeError("Falta pydeck. Instalá: pip install pydeck")

    if gdf.empty:
        return None, pd.DataFrame(), pd.DataFrame()

    gdf = gdf.copy()
    gdf = drop_duplicated_columns_keep_first(gdf)

    # Asegura que siga siendo GeoDataFrame después de eliminar duplicados.
    if gpd is not None and "geometry" in gdf.columns and not isinstance(gdf, gpd.GeoDataFrame):
        gdf = gpd.GeoDataFrame(gdf, geometry="geometry")

    if gdf.crs is None:
        gdf = gdf.set_crs(4326, allow_override=True)
    else:
        gdf = gdf.to_crs(4326)

    # Métricas derivadas para capas temáticas.
    area_geom = gdf.to_crs(3857).geometry.area / 1_000_000
    gdf["area_km2"] = area_geom.replace(0, np.nan).fillna(0.000001)
    gdf[value_col] = pd.to_numeric(gdf[value_col], errors="coerce").fillna(0)
    gdf["valor_por_km2"] = gdf[value_col] / gdf["area_km2"]
    gdf["log_valor"] = np.log1p(gdf[value_col])

    if metric_mode == "Valor por km²":
        metric_col = "valor_por_km2"
        metric_title = "Valor por km²"
    elif metric_mode == "Log(valor + 1)":
        metric_col = "log_valor"
        metric_title = "Log(valor + 1)"
    else:
        metric_col = value_col
        metric_title = "Valor total"

    if simplify_tolerance and simplify_tolerance > 0:
        gdf["geometry"] = gdf.geometry.simplify(simplify_tolerance, preserve_topology=True)

    gdf["_norm"] = compute_norm_series(gdf[metric_col], method=color_method, clip_upper_pct=clip_upper_pct)
    gdf["fill_color"] = gdf["_norm"].apply(lambda x: palette_color(x, color_palette, polygon_opacity))
    gdf["color_hex"] = gdf["fill_color"].apply(rgba_to_hex)
    gdf["elevation"] = (gdf["_norm"] * 3500).fillna(0)
    gdf["ranking_mapa"] = gdf[value_col].rank(method="first", ascending=False).astype(int)
    gdf["percentil_mapa"] = gdf[value_col].rank(method="average", pct=True).fillna(0)
    gdf["percentil_mapa_pct"] = (gdf["percentil_mapa"] * 100).round(1).astype(str) + "%"
    gdf["valor_fmt"] = gdf[value_col].map(format_number)
    gdf["area_km2_fmt"] = gdf["area_km2"].map(lambda x: format_small_float(x, 2))
    gdf["valor_por_km2_fmt"] = gdf["valor_por_km2"].map(lambda x: format_small_float(x, 1))

    # Centroides en CRS proyectado para evitar distorsión y warnings.
    centroids = gdf.to_crs(3857).geometry.centroid.to_crs(4326)
    gdf["lon"] = centroids.x
    gdf["lat"] = centroids.y
    gdf["radio_burbuja"] = (70 + 1250 * np.sqrt(gdf["_norm"].clip(0, 1)) * float(bubble_scale)).fillna(70)
    join_col = JOIN_COL_BY_YEAR[year]
    if label_mode == "Código de radio":
        gdf["label"] = gdf[join_col].astype(str)
    elif label_mode == "Ranking":
        gdf["label"] = "#" + gdf["ranking_mapa"].astype(str)
    elif label_mode == "Percentil":
        gdf["label"] = gdf["percentil_mapa_pct"]
    else:
        gdf["label"] = gdf[value_col].round(0).astype("Int64").astype(str)

    export_cols = unique_existing_columns(
        [
            join_col,
            value_col,
            metric_col,
            "ranking_mapa",
            "percentil_mapa",
            "percentil_mapa_pct",
            "valor_fmt",
            "area_km2",
            "area_km2_fmt",
            "valor_por_km2",
            "valor_por_km2_fmt",
            "color_hex",
            "fill_color",
            "elevation",
            "geometry",
        ],
        gdf.columns,
    )
    geojson_gdf = gdf[export_cols].copy()
    geojson_gdf = drop_duplicated_columns_keep_first(geojson_gdf)
    if gpd is not None and "geometry" in geojson_gdf.columns and not isinstance(geojson_gdf, gpd.GeoDataFrame):
        geojson_gdf = gpd.GeoDataFrame(geojson_gdf, geometry="geometry", crs=gdf.crs)
    geojson = json.loads(geojson_gdf.to_json())

    layers = []
    if show_polygons:
        layers.append(
            pdk.Layer(
                "GeoJsonLayer",
                geojson,
                pickable=True,
                stroked=True,
                filled=True,
                extruded=extruded,
                get_fill_color="properties.fill_color",
                get_elevation="properties.elevation",
                get_line_color=[35, 35, 35, int(border_opacity)],
                line_width_min_pixels=float(border_width),
            )
        )

    if show_outline_layer:
        layers.append(
            pdk.Layer(
                "GeoJsonLayer",
                geojson,
                pickable=False,
                stroked=True,
                filled=False,
                get_line_color=[10, 10, 10, max(120, int(border_opacity))],
                line_width_min_pixels=max(0.7, float(border_width)),
            )
        )

    point_cols = unique_existing_columns(
        [
            join_col,
            value_col,
            metric_col,
            "ranking_mapa",
            "percentil_mapa",
            "percentil_mapa_pct",
            "valor_fmt",
            "area_km2",
            "area_km2_fmt",
            "valor_por_km2",
            "valor_por_km2_fmt",
            "lon",
            "lat",
            "radio_burbuja",
            "label",
            "color_hex",
        ],
        gdf.columns,
    )
    points_df = pd.DataFrame(drop_duplicated_columns_keep_first(gdf[point_cols]))

    if show_heatmap:
        layers.append(
            pdk.Layer(
                "HeatmapLayer",
                points_df,
                get_position="[lon, lat]",
                get_weight=metric_col,
                radius_pixels=int(heatmap_radius),
                intensity=1,
                threshold=0.03,
            )
        )

    if show_centroid_reference:
        layers.append(
            pdk.Layer(
                "ScatterplotLayer",
                points_df,
                pickable=False,
                get_position="[lon, lat]",
                get_radius=35,
                get_fill_color=[20, 20, 20, 90],
                get_line_color=[255, 255, 255, 120],
                line_width_min_pixels=0.5,
            )
        )

    if show_bubbles:
        layers.append(
            pdk.Layer(
                "ScatterplotLayer",
                points_df,
                pickable=True,
                get_position="[lon, lat]",
                get_radius="radio_burbuja",
                get_fill_color=[30, 90, 180, 115],
                get_line_color=[20, 20, 20, 130],
                line_width_min_pixels=1,
            )
        )

    if show_labels:
        labels_df = points_df.sort_values(value_col, ascending=False).head(int(label_top_n)).copy()
        layers.append(
            pdk.Layer(
                "TextLayer",
                labels_df,
                pickable=False,
                get_position="[lon, lat]",
                get_text="label",
                get_size=13,
                get_color=[20, 20, 20, 235],
                get_angle=0,
                get_text_anchor="middle",
                get_alignment_baseline="center",
            )
        )

    view_state = compute_view_state(gdf, extruded=extruded)

    # Encabezado del tooltip con la variable y el nivel (categoría) que se está mapeando.
    header_lines = ""
    if variable_label:
        header_lines += f"<div style='font-weight:600;margin-bottom:4px'>{variable_label}</div>"
    if nivel_label:
        header_lines += f"<div style='color:#555;margin-bottom:4px'>Nivel: {nivel_label}</div>"

    tooltip = {
        "html": (
            "<div style='min-width:210px'>"
            + header_lines
            + "<b>Radio:</b> {" + join_col + "}<br/>"
            "<b>Valor:</b> {valor_fmt}<br/>"
            "<b>Ranking:</b> {ranking_mapa}<br/>"
            "<b>Percentil:</b> {percentil_mapa_pct}<br/>"
            "<b>Km²:</b> {area_km2_fmt}<br/>"
            "<b>Valor/km²:</b> {valor_por_km2_fmt}<br/>"
            f"<b>Capa:</b> {metric_title}<br/>"
            f"<b>Escala:</b> {color_method}"
            "</div>"
        ),
        "style": {"backgroundColor": "rgba(255,255,255,0.96)", "color": "black"},
    }

    map_provider, map_style, _ = get_base_map_config(base_map_style)
    deck = pdk.Deck(
        layers=layers,
        initial_view_state=view_state,
        tooltip=tooltip,
        map_provider=map_provider,
        map_style=map_style,
        height=int(map_height),
    )
    summary_cols = unique_existing_columns(
        [join_col, value_col, metric_col, "ranking_mapa", "percentil_mapa", "percentil_mapa_pct", "valor_fmt", "area_km2", "area_km2_fmt", "valor_por_km2", "valor_por_km2_fmt", "lon", "lat", "color_hex"],
        gdf.columns,
    )
    map_summary = pd.DataFrame(drop_duplicated_columns_keep_first(gdf[summary_cols])).copy()

    q = pd.to_numeric(gdf[metric_col], errors="coerce").fillna(0)
    legend_df = pd.DataFrame(
        {
            "tramo": ["Bajo", "Medio", "Alto"],
            "referencia": [float(q.quantile(0.10)), float(q.quantile(0.50)), float(q.quantile(0.90))],
            "color": [
                rgba_to_hex(palette_color(0.10, color_palette, polygon_opacity)),
                rgba_to_hex(palette_color(0.50, color_palette, polygon_opacity)),
                rgba_to_hex(palette_color(0.90, color_palette, polygon_opacity)),
            ],
        }
    )
    return deck, map_summary, legend_df


# -----------------------------------------------------------------------------
# UI
# -----------------------------------------------------------------------------

st.title("Dashboard Censo Argentino")
st.caption(
    "Consulta directa a archivos Parquet/GeoParquet remotos de Source Cooperative usando DuckDB, sin QGIS."
)

with st.sidebar:
    st.header("Filtros")
    year_selected = st.selectbox("Año censal", VALID_YEARS, index=VALID_YEARS.index(2022))

    st.divider()
    st.subheader("Variable")

    def _sync_search_from_theme() -> None:
        """Al cambiar el tema, refresca el buscador con sus términos sugeridos."""
        theme = st.session_state.get("theme_selected", "General")
        st.session_state["variable_search"] = " OR ".join(VARIABLE_THEMES.get(theme, ["pob"])[:3])

    if "variable_search" not in st.session_state:
        _sync_search_from_theme()

    theme_selected = st.selectbox(
        "Tema de variables",
        list(VARIABLE_THEMES.keys()),
        index=0,
        key="theme_selected",
        on_change=_sync_search_from_theme,
        help="Preselecciona términos de búsqueda para descubrir variables reales del censo.",
    )
    variable_search = st.text_input(
        "Buscar variable",
        key="variable_search",
        help="Busca sobre las variables presentes en census-data.parquet. Podés escribir edad, sexo, vivienda, agua, educación, internet, etc. Separá términos con OR.",
    )

    # El buscador admite términos separados por OR para ampliar el descubrimiento temático.
    search_terms = [t.strip() for t in variable_search.split(" OR ") if t.strip()] or [variable_search]
    variables_parts = [get_census_variables(year_selected, term, limit=250) for term in search_terms]
    variables_df = pd.concat(variables_parts, ignore_index=True).drop_duplicates(subset=["codigo_variable", "fuente"]) if variables_parts else pd.DataFrame()
    if variables_df.empty:
        st.warning("No encontré variables con esa búsqueda. Probá con otro término o escribí el código exacto abajo.")

    variable_options = []
    for _, row in variables_df.iterrows():
        code = str(row["codigo_variable"])
        label = str(row.get("etiqueta_variable", "")) if pd.notna(row.get("etiqueta_variable", "")) else ""
        source = str(row.get("fuente", ""))
        shown = f"{code} · {label}" if label else code
        if source:
            shown = f"{shown} [{source}]"
        variable_options.append((code, shown))

    if not variable_options:
        variable_options = [("POB_TOT_P", "POB_TOT_P · Población total [radios.parquet]")]

    variable_label_to_code = {label: code for code, label in variable_options}
    option_labels = list(variable_label_to_code.keys())
    st.caption(f"{len(option_labels)} variable(s) coinciden con la búsqueda.")

    # Preserva la variable elegida cuando la búsqueda cambia pero la variable sigue disponible.
    prev_code = st.session_state.get("variable_selected_code")
    default_index = next((i for i, lbl in enumerate(option_labels) if variable_label_to_code[lbl] == prev_code), 0)
    selected_variable_label = st.selectbox(
        "Variable censal",
        option_labels,
        index=default_index,
    )
    variable_selected = variable_label_to_code.get(selected_variable_label, option_labels and variable_label_to_code[option_labels[0]] or "POB_TOT_P")
    st.session_state["variable_selected_code"] = variable_selected

    use_manual_variable = st.checkbox(
        "Escribir código manualmente",
        value=False,
        help="Activá esto solo si querés forzar un código que no aparece en el desplegable.",
    )
    if use_manual_variable:
        manual_variable = st.text_input(
            "Código exacto",
            value=variable_selected,
            help="Ejemplo: POB_TOT_P",
        )
        variable_selected = manual_variable.strip() or variable_selected

    st.divider()
    st.subheader("Geografía")

    provincias = ensure_dataframe(get_provincias(year_selected), "provincias")
    prov_options = {"Todas": "__ALL__"}
    required_prov_cols = {"etiqueta_provincia", "valor_provincia"}
    if provincias.empty or not required_prov_cols.issubset(set(provincias.columns)):
        st.warning("No pude cargar el catálogo de provincias. Usá 'Todas' o SQL avanzado.")
    else:
        for _, row in provincias.iterrows():
            label = f"{row['etiqueta_provincia']} ({row['valor_provincia']})"
            prov_options[label] = str(row["valor_provincia"])

    selected_prov_label = st.selectbox("Provincia", list(prov_options.keys()))
    provincia_code = prov_options[selected_prov_label]

    dept_options = {"Todos": "__ALL__"}
    if provincia_code != "__ALL__":
        departamentos = ensure_dataframe(get_departamentos(year_selected, provincia_code), "departamentos")
        required_dept_cols = {"etiqueta_departamento", "valor_departamento"}
        if departamentos.empty or not required_dept_cols.issubset(set(departamentos.columns)):
            st.info("No pude cargar nombres de departamentos para esta provincia. La consulta queda a nivel provincia.")
        else:
            for _, row in departamentos.iterrows():
                label = f"{row['etiqueta_departamento']} ({row['valor_departamento']})"
                dept_options[label] = str(row["valor_departamento"])

    selected_dept_label = st.selectbox("Departamento", list(dept_options.keys()))
    departamento_code = dept_options[selected_dept_label]

    # Contexto de la variable: define qué filtros tienen sentido aguas abajo.
    is_radios_var = variable_selected in RADIOS_NUMERIC_VARIABLES
    categorias = ensure_dataframe(get_categorias(year_selected, variable_selected), "categorías")
    has_categories = (not categorias.empty) and ("valor_categoria" in categorias.columns)

    if is_radios_var:
        st.info("ℹ️ Variable de `radios.parquet`: se mide como total geográfico, sin categorías.")
    elif has_categories:
        st.caption(f"Esta variable tiene {len(categorias)} categoría(s) disponibles.")
    else:
        st.caption("Esta variable no expone categorías en census-data.parquet.")

    st.divider()
    st.subheader("Categoría")
    cat_options = {"Todas": "__ALL__"}
    if has_categories:
        for _, row in categorias.iterrows():
            label = f"{row.get('etiqueta_categoria', '')} ({row.get('valor_categoria', '')})"
            cat_options[label] = str(row["valor_categoria"])
        selected_cat_label = st.selectbox("Categoría", list(cat_options.keys()))
        categoria_value = cat_options[selected_cat_label]
    else:
        st.selectbox("Categoría", ["Todas"], disabled=True, help="La variable seleccionada no tiene categorías para filtrar.")
        categoria_value = "__ALL__"
        selected_cat_label = "Todas"

    st.divider()
    st.subheader("Medición")
    # En variables de radios el "% dentro de la variable" da siempre 100%: se oculta.
    available_measures = [m for m in MEASURE_MODES if not (is_radios_var and m == "% dentro de la variable")]
    measure_mode = st.selectbox(
        "Cómo medir",
        available_measures,
        index=0,
        help="Usá porcentajes para comparar zonas de distinto tamaño. El conteo absoluto sirve para volumen/concentración.",
    )

    st.divider()
    # "Categoría" como agrupación sólo aplica si la variable realmente tiene categorías.
    group_options = [g for g in GROUP_MAP_LONG.keys() if not (g == "Categoría" and (is_radios_var or not has_categories))]
    default_group = "Departamento" if "Departamento" in group_options else group_options[0]
    group_selected = st.selectbox("Agrupar por", group_options, index=group_options.index(default_group))
    limit_rows = st.slider("Máximo de filas", min_value=10, max_value=5000, value=500, step=10)


variable_label_text = get_variable_label(year_selected, variable_selected)

col_a, col_b, col_c, col_d = st.columns(4)
col_a.metric("Año", str(year_selected))
col_b.metric("Variable", variable_selected, help=variable_label_text or "Sin descripción en metadata.parquet")
col_c.metric("Provincia", "Todas" if provincia_code == "__ALL__" else selected_prov_label.split(" (")[0])
col_d.metric("Medición", measure_mode)

if variable_label_text:
    st.caption(f"📋 **{variable_selected}** — {variable_label_text}")
else:
    st.caption(f"📋 **{variable_selected}** — sin descripción en metadata.parquet para este año.")

with st.expander("🔢 Niveles (categorías) de la variable", expanded=False):
    if is_radios_var:
        st.info("Esta variable proviene de `radios.parquet`: es un total geográfico, no tiene niveles.")
    elif not has_categories:
        st.info("Esta variable no expone categorías en census-data.parquet.")
    else:
        geo_txt = "todo el país" if provincia_code == "__ALL__" else selected_prov_label.split(" (")[0]
        if departamento_code != "__ALL__":
            geo_txt = f"{selected_dept_label.split(' (')[0]} ({geo_txt})"
        st.caption(
            f"Conteo de cada nivel para **{geo_txt}**. "
            "Para desagregar los gráficos por estos niveles, elegí «Agrupar por: Categoría»."
        )
        try:
            # group_label='Categoría' + categoría '__ALL__' devuelve todos los niveles con su conteo.
            niveles = enrich_result(
                query_censo_long(
                    year_selected, variable_selected, provincia_code, departamento_code,
                    "__ALL__", "Categoría", 0,
                )
            )
            if niveles.empty:
                st.warning("No hay datos de niveles para este recorte geográfico.")
            else:
                niveles_disp = humanize_area_labels(niveles, "Categoría", year_selected, provincia_code)
                tabla = pd.DataFrame({
                    "Nivel": niveles_disp["area_label"],
                    "Código": niveles_disp.get("valor_categoria", "").astype(str),
                    "Conteo": niveles_disp["conteo"].map(format_number),
                    "% del total": niveles_disp["pct_total"].map(lambda x: f"{x:.1%}"),
                })
                col_tabla, col_chart = st.columns([1, 1])
                with col_tabla:
                    st.dataframe(tabla, use_container_width=True, hide_index=True)
                with col_chart:
                    st.altair_chart(
                        build_bar_chart(niveles_disp, "area_label", min(30, len(niveles_disp)), "Conteo absoluto"),
                        use_container_width=True,
                    )
        except Exception as exc:
            st.caption(f"No pude calcular los niveles: {exc}")

try:
    df_result, query_source = query_censo_resilient(
        year=year_selected,
        variable=variable_selected,
        provincia_code=provincia_code,
        departamento_code=departamento_code,
        categoria_value=categoria_value,
        group_label=group_selected,
        limit=limit_rows,
        measure_mode=measure_mode,
    )
except Exception as exc:
    st.error("No pude ejecutar la consulta principal.")
    st.exception(exc)
    st.stop()

if df_result.empty:
    st.warning("La consulta no devolvió resultados. Revisá el código de variable o los filtros.")
    st.info(
        "Tip: el selector ahora lista variables que existen en census-data.parquet. "
        "Para población total o viviendas totales, usá POB_TOT_P o VIV_TOT_P, que pueden venir desde radios.parquet."
    )
    try:
        diag = diagnose_query(
            year_selected,
            variable_selected,
            provincia_code,
            departamento_code,
            categoria_value,
        )
        st.subheader("Diagnóstico")
        st.write("Chequeo exacto en census-data.parquet")
        st.dataframe(diag["exacto"], use_container_width=True)
        if not diag["sugerencias"].empty:
            st.write("Variables parecidas presentes en census-data.parquet")
            st.dataframe(diag["sugerencias"], use_container_width=True)
    except Exception as exc:
        st.caption(f"No pude generar diagnóstico automático: {exc}")
else:
    df_enriched = enrich_result(df_result)
    # Etiquetas legibles ("Nombre (código)") en lugar de códigos pelados como "497".
    df_enriched = humanize_area_labels(df_enriched, group_selected, year_selected, provincia_code)
    label_col = "area_label" if "area_label" in df_enriched.columns else get_label_column(df_enriched)
    summary = summarize_result(df_enriched, label_col)

    col_total, col_source = st.columns([1, 2])
    metric_title_main = "Total según consulta" if measure_mode == "Conteo absoluto" else "Suma de valores medidos"
    col_total.metric(metric_title_main, format_number(summary["total"]))
    col_source.info(f"Fuente usada para esta consulta: {query_source}")

    tab_resumen, tab_graficos, tab_mapa, tab_metadata, tab_sql = st.tabs(
        ["Resumen", "Gráficos", "Mapa y capas", "Metadata", "SQL avanzado"]
    )

    with tab_resumen:
        st.subheader("Resumen ejecutivo")
        kpi1, kpi2, kpi3, kpi4 = st.columns(4)
        kpi1.metric("Total" if measure_mode == "Conteo absoluto" else "Suma medición", format_number(summary["total"]))
        kpi2.metric("Áreas", format_number(summary["n_areas"]))
        kpi3.metric("Promedio por área", format_number(summary["mean_value"]))
        kpi4.metric("Mediana por área", format_number(summary["median_value"]))

        if measure_mode == "Conteo absoluto":
            st.caption(
                f"Área con mayor valor: **{summary['top_label']}** "
                f"con {format_number(summary['top_value'])} "
                f"({summary['top_share']:.1%} del total)."
            )
        else:
            st.caption(
                f"Área con mayor medición: **{summary['top_label']}** "
                f"con {format_small_float(summary['top_value'], 2)}%. "
                "Las columnas `conteo_abs`, `denominador` y `pct_variable` conservan la trazabilidad."
            )

        display_df = df_enriched.copy()
        if "pct_total" in display_df.columns:
            display_df["pct_total"] = display_df["pct_total"].map(lambda x: f"{x:.2%}")
        if "pct_acum" in display_df.columns:
            display_df["pct_acum"] = display_df["pct_acum"].map(lambda x: f"{x:.2%}")

        st.subheader("Tabla enriquecida")
        st.dataframe(display_df, use_container_width=True, hide_index=True)

        csv = df_enriched.to_csv(index=False).encode("utf-8-sig")
        st.download_button(
            "Descargar CSV enriquecido",
            data=csv,
            file_name=f"censo_{year_selected}_{variable_selected}_{group_selected}.csv".replace(" ", "_"),
            mime="text/csv",
        )

    with tab_graficos:
        st.subheader("Gráficos informativos")
        if label_col is None:
            st.info("No encontré una columna de etiqueta para graficar esta consulta.")
        else:
            top_n_chart = st.slider("Top N para gráficos", 5, 50, min(20, max(5, len(df_enriched))), 5)

            # ¿Podemos discriminar por niveles? (variable con categorías, sin nivel filtrado, área geográfica)
            discriminate = (
                has_categories
                and categoria_value == "__ALL__"
                and group_selected in ("Provincia", "Departamento", "Radio censal")
            )

            if discriminate:
                st.markdown("### Discriminado por niveles de la variable")
                try:
                    raw_levels = query_censo_levels_by_area(
                        year_selected, variable_selected, provincia_code, departamento_code, group_selected
                    )
                    lvl_df, area_order = prepare_levels_by_area(
                        raw_levels, group_selected, year_selected, provincia_code, top_n_chart
                    )
                    if lvl_df.empty:
                        st.info("No hay datos por nivel para este recorte; muestro los totales por área más abajo.")
                    else:
                        st.markdown(f"**Ranking por área, abierto por nivel** (top {len(area_order)} áreas)")
                        st.altair_chart(
                            build_grouped_bar_chart(lvl_df, area_order, measure_mode),
                            use_container_width=True,
                        )
                        st.markdown("**Composición: % de cada nivel dentro del área**")
                        st.altair_chart(
                            build_grouped_share_chart(lvl_df, area_order),
                            use_container_width=True,
                        )
                        st.caption(
                            "Cada área se abre en sus niveles (barras agrupadas). El segundo gráfico normaliza a % "
                            "dentro de cada área para comparar composición independientemente del tamaño."
                        )
                except Exception as exc:
                    st.warning(f"No pude graficar la apertura por niveles: {exc}")
                st.divider()
                st.markdown("### Totales por área (todos los niveles sumados)")

            chart_a, chart_b = st.columns(2)
            with chart_a:
                st.markdown("**Ranking de áreas**")
                st.altair_chart(build_bar_chart(df_enriched, label_col, top_n_chart, measure_mode), use_container_width=True)
            with chart_b:
                st.markdown("**Participación sobre el total**")
                st.altair_chart(build_share_chart(df_enriched, label_col, top_n_chart), use_container_width=True)

            st.markdown("**Pareto: acumulado del top seleccionado**")
            st.altair_chart(build_pareto_chart(df_enriched, label_col, top_n_chart, measure_mode), use_container_width=True)

            st.markdown("**Distribución de valores entre áreas**")
            st.altair_chart(build_distribution_chart(df_enriched, measure_mode), use_container_width=True)

            with st.expander("Lectura rápida"):
                st.write(
                    "Este bloque ayuda a detectar concentración territorial: si pocas áreas explican un porcentaje alto del total, "
                    "conviene revisar esas áreas en el mapa y eventualmente bajar a nivel radio censal."
                )

    with tab_mapa:
        st.subheader("Mapa de radios censales y capas")
        st.caption(
            "Para evitar tiempos de carga altos, conviene filtrar al menos una provincia y, si es Buenos Aires/CABA, idealmente un departamento/comuna."
        )

        # Nivel a mapear: el mapa pinta un valor por radio, así que se elige un nivel (o el total).
        # Por defecto respeta el nivel elegido en la barra lateral.
        map_categoria = categoria_value
        nivel_mapa_txt = "Todos los niveles (suma)" if categoria_value == "__ALL__" else selected_cat_label.split(" (")[0]

        if gpd is None or pdk is None:
            st.warning(
                "Para activar el mapa instalá dependencias geográficas: geopandas, pyogrio, shapely y pydeck."
            )
        elif provincia_code == "__ALL__":
            st.info("Seleccioná una provincia para generar el mapa sin cargar todo el país.")
        elif group_selected == "Categoría" and variable_selected in RADIOS_NUMERIC_VARIABLES:
            st.info("Las variables provenientes de radios.parquet no tienen categorías. Cambiá la agrupación para mapear.")
        else:
            with st.expander("Cómo leer las capas", expanded=False):
                st.write(
                    "Usá **Valor total** para ver concentración absoluta, **Valor por km²** para una aproximación de densidad "
                    "y **Log(valor + 1)** cuando un radio muy grande domina la escala. La escala por **cuantiles** reparte los colores "
                    "de forma más equilibrada cuando hay muchos outliers."
                )

            # Selector de nivel: permite recolorear el mapa por cada categoría de la variable.
            if has_categories:
                nivel_opts = {"Total (todos los niveles)": "__ALL__"}
                for _, r in categorias.iterrows():
                    etiqueta = _clean_text(r.get("etiqueta_categoria")) or "Nivel"
                    nivel_opts[f"{etiqueta} ({r.get('valor_categoria')})"] = str(r["valor_categoria"])
                nivel_keys = list(nivel_opts.keys())
                default_idx = next((i for i, v in enumerate(nivel_opts.values()) if v == categoria_value), 0)
                map_nivel_label = st.selectbox(
                    "Nivel a mapear",
                    nivel_keys,
                    index=default_idx,
                    help="El mapa pinta un valor por radio. Elegí qué nivel de la variable ver; «Total» suma todos los niveles.",
                )
                map_categoria = nivel_opts[map_nivel_label]
                nivel_mapa_txt = "Todos los niveles (suma)" if map_categoria == "__ALL__" else map_nivel_label.split(" (")[0]

            desc_mapa = f" — {variable_label_text}" if variable_label_text else ""
            st.info(
                f"🗺️ **Mapeando:** {variable_selected}{desc_mapa}  ·  "
                f"**Nivel:** {nivel_mapa_txt}  ·  **Medición:** {measure_mode}"
            )

            st.markdown("**Mapa base y contexto**")
            base1, base2, base3, base4 = st.columns(4)
            with base1:
                base_map_style = st.selectbox(
                    "Mapa base",
                    list(BASE_MAP_STYLES.keys()),
                    index=0,
                    help="Voyager suele ser el más informativo porque muestra calles, rutas y etiquetas."
                )
                st.caption(BASE_MAP_STYLES[base_map_style]["desc"])
            with base2:
                map_height = st.slider("Alto del mapa", 420, 900, 680, 20)
                show_outline_layer = st.checkbox("Contorno reforzado", value=True)
            with base3:
                show_centroid_reference = st.checkbox("Centroides de referencia", value=False)
                label_mode = st.selectbox("Texto de etiquetas", ["Valor", "Ranking", "Percentil", "Código de radio"], index=0)
            with base4:
                st.info("Para ver mejor calles y etiquetas, bajá la opacidad de polígonos a 90–130 o usá base Voyager.")

            cfg1, cfg2, cfg3, cfg4 = st.columns(4)
            with cfg1:
                metric_mode = st.selectbox(
                    "Métrica de color",
                    ["Valor total", "Valor por km²", "Log(valor + 1)"],
                    index=0,
                    help="Valor por km² sirve como aproximación de densidad para población/viviendas.",
                )
                color_method = st.selectbox(
                    "Escala de color",
                    ["Continuo", "Cuantiles"],
                    index=0,
                    help="Cuantiles mejora la lectura cuando hay outliers fuertes.",
                )
            with cfg2:
                color_palette = st.selectbox(
                    "Paleta",
                    ["Azul", "Verde", "Naranja/Rojo", "Violeta", "Gris/Azul"],
                    index=0,
                )
                clip_upper_pct = st.slider(
                    "Corte superior color",
                    min_value=80,
                    max_value=100,
                    value=98,
                    step=1,
                    help="Recorta la escala de color en el percentil elegido para que los outliers no apaguen el mapa.",
                )
            with cfg3:
                simplify_tolerance = st.slider(
                    "Simplificación geométrica",
                    min_value=0.0,
                    max_value=0.006,
                    value=0.0004,
                    step=0.0001,
                    help="Valores más altos alivian el mapa, pero reducen detalle de polígonos.",
                )
                max_radios_render = st.slider(
                    "Máximo radios a dibujar",
                    min_value=500,
                    max_value=15000,
                    value=6000,
                    step=500,
                    help="Si el filtro devuelve más radios, la app conserva los de mayor valor para proteger performance.",
                )
            with cfg4:
                polygon_opacity = st.slider("Opacidad polígonos", 40, 240, 165, 5)
                border_opacity = st.slider("Opacidad bordes", 0, 220, 90, 5)

            st.markdown("**Capas visibles**")
            layer1, layer2, layer3, layer4, layer5 = st.columns(5)
            with layer1:
                show_polygons = st.checkbox("Coroplético", value=True)
            with layer2:
                show_bubbles = st.checkbox("Burbujas", value=False)
            with layer3:
                show_heatmap = st.checkbox("Calor", value=False)
            with layer4:
                show_labels = st.checkbox("Etiquetas top", value=False)
            with layer5:
                extruded = st.checkbox("3D", value=False)

            adv1, adv2, adv3, adv4 = st.columns(4)
            with adv1:
                border_width = st.slider("Grosor borde", 0.1, 3.0, 0.4, 0.1)
            with adv2:
                bubble_scale = st.slider("Escala burbujas", 0.3, 3.0, 1.0, 0.1)
            with adv3:
                heatmap_radius = st.slider("Radio capa calor", 20, 140, 55, 5)
            with adv4:
                label_top_n = st.slider("Cantidad de etiquetas", 5, 100, 20, 5)

            st.markdown("**Filtro visual de radios**")
            f1, f2, f3 = st.columns(3)
            with f1:
                map_filter_mode = st.selectbox(
                    "Mostrar",
                    ["Todos los radios filtrados", "Top N por valor", "Percentil superior", "Valor mínimo"],
                    index=0,
                )
            with f2:
                top_n_map = st.number_input("Top N", min_value=50, max_value=20000, value=3000, step=50)
                percentile_min = st.slider("Desde percentil", 50, 99, 80, 1)
            with f3:
                min_value_map = st.number_input("Valor mínimo", min_value=0.0, value=0.0, step=10.0)

            if st.button("Generar mapa", type="primary"):
                try:
                    with st.spinner("Consultando radios y geometrías..."):
                        radio_counts, radio_source = query_radio_counts(
                            year_selected,
                            variable_selected,
                            provincia_code,
                            departamento_code,
                            map_categoria,
                            measure_mode,
                        )
                        if radio_counts.empty:
                            st.warning("No hay datos a nivel radio para mapear con esos filtros.")
                            st.stop()

                        join_col = JOIN_COL_BY_YEAR[year_selected]
                        gdf = load_radios_geoparquet(year_selected, provincia_code, departamento_code)
                        radio_counts["id_geo"] = radio_counts["id_geo"].astype(str)
                        gdf[join_col] = gdf[join_col].astype(str)
                        gdf_plot = gdf.merge(
                            radio_counts,
                            left_on=join_col,
                            right_on="id_geo",
                            how="inner",
                            suffixes=("", "_dato"),
                        )
                        gdf_plot = drop_duplicated_columns_keep_first(gdf_plot)
                        gdf_plot["conteo"] = pd.to_numeric(gdf_plot["conteo"], errors="coerce").fillna(0)

                        radios_originales = len(gdf_plot)
                        if map_filter_mode == "Top N por valor":
                            gdf_plot = gdf_plot.sort_values("conteo", ascending=False).head(int(top_n_map))
                        elif map_filter_mode == "Percentil superior":
                            threshold = gdf_plot["conteo"].quantile(float(percentile_min) / 100)
                            gdf_plot = gdf_plot[gdf_plot["conteo"] >= threshold]
                        elif map_filter_mode == "Valor mínimo":
                            gdf_plot = gdf_plot[gdf_plot["conteo"] >= float(min_value_map)]

                        if len(gdf_plot) > int(max_radios_render):
                            st.warning(
                                f"El filtro devolvió {len(gdf_plot):,} radios. Para cuidar performance se muestran los {int(max_radios_render):,} de mayor valor."
                                .replace(",", ".")
                            )
                            gdf_plot = gdf_plot.sort_values("conteo", ascending=False).head(int(max_radios_render))

                    if gdf_plot.empty:
                        st.warning("El join entre radios y datos censales quedó vacío o el filtro visual eliminó todos los radios.")
                    else:
                        st.caption(
                            f"Fuente del mapa: {radio_source}. Radios base: {format_number(radios_originales)} · Radios renderizados: {format_number(len(gdf_plot))}."
                        )
                        deck, map_summary, legend_df = build_choropleth_map(
                            gdf_plot,
                            year=year_selected,
                            value_col="conteo",
                            simplify_tolerance=simplify_tolerance,
                            metric_mode=metric_mode,
                            color_method=color_method,
                            color_palette=color_palette,
                            clip_upper_pct=clip_upper_pct,
                            polygon_opacity=polygon_opacity,
                            border_opacity=border_opacity,
                            border_width=border_width,
                            show_polygons=show_polygons,
                            show_bubbles=show_bubbles,
                            show_heatmap=show_heatmap,
                            show_labels=show_labels,
                            show_outline_layer=show_outline_layer,
                            show_centroid_reference=show_centroid_reference,
                            label_top_n=label_top_n,
                            label_mode=label_mode,
                            bubble_scale=bubble_scale,
                            heatmap_radius=heatmap_radius,
                            extruded=extruded,
                            base_map_style=base_map_style,
                            map_height=map_height,
                            variable_label=f"{variable_selected} — {variable_label_text}" if variable_label_text else variable_selected,
                            nivel_label="" if map_categoria == "__ALL__" else nivel_mapa_txt,
                        )
                        st.pydeck_chart(deck, use_container_width=True)

                        m1, m2, m3, m4 = st.columns(4)
                        m1.metric("Radios mapeados", format_number(len(gdf_plot)))
                        m2.metric("Área aproximada km²", format_number(map_summary["area_km2"].sum()))
                        m3.metric("Valor/km² promedio", format_number(map_summary["valor_por_km2"].mean()))
                        m4.metric("Máximo radio", format_number(map_summary["conteo"].max()))

                        with st.expander("Leyenda y escala", expanded=True):
                            st.markdown(f"**Variable:** {variable_selected}" + (f" — {variable_label_text}" if variable_label_text else ""))
                            st.markdown(f"**Nivel mapeado:** {nivel_mapa_txt}")
                            st.markdown(
                                f"**Métrica de color:** {metric_mode}  ·  **Escala:** {color_method}  ·  **Medición:** {measure_mode}"
                            )
                            # Barras de color con su valor de referencia (percentiles 10/50/90).
                            chips = "".join(
                                "<div style='display:flex;align-items:center;margin:2px 0'>"
                                f"<span style='display:inline-block;width:22px;height:14px;background:{row.color};"
                                "border:1px solid #999;margin-right:8px'></span>"
                                f"<span>{row.tramo} · ≈ {format_number(row.referencia)}</span>"
                                "</div>"
                                for row in legend_df.itertuples()
                            )
                            st.markdown(chips, unsafe_allow_html=True)
                            st.caption(
                                "Referencial: cada color corresponde aproximadamente a los percentiles 10, 50 y 90 "
                                "de la métrica usada para colorear."
                            )

                        with st.expander("Top radios mapeados"):
                            top_map = map_summary.sort_values("conteo", ascending=False).head(1000).copy()
                            if "percentil_mapa" in top_map.columns:
                                top_map["percentil_mapa"] = top_map["percentil_mapa"].map(lambda x: f"{x:.1%}")
                            st.dataframe(top_map, use_container_width=True, hide_index=True)
                            csv_map = map_summary.to_csv(index=False).encode("utf-8-sig")
                            st.download_button(
                                "Descargar radios mapeados CSV",
                                data=csv_map,
                                file_name=f"radios_mapa_{year_selected}_{variable_selected}.csv".replace(" ", "_"),
                                mime="text/csv",
                            )
                except Exception as exc:
                    st.error("No pude generar el mapa.")
                    st.exception(exc)

    with tab_metadata:
        st.subheader("Variables disponibles")
        st.caption("Esta tabla está basada en los códigos presentes en census-data.parquet y enriquecida con metadata.parquet.")
        st.dataframe(variables_df, use_container_width=True, hide_index=True)

        st.subheader("Explorador rápido por tema")
        theme_rows = []
        for tema, terms in VARIABLE_THEMES.items():
            theme_rows.append({"tema": tema, "buscar_por": ", ".join(terms)})
        st.dataframe(pd.DataFrame(theme_rows), use_container_width=True, hide_index=True)

        with st.expander("Ver metadata completa"):
            st.dataframe(get_metadata(year_selected), use_container_width=True)

        with st.expander("Ver esquemas de archivos"):
            schema_table = st.selectbox("Archivo", ["census", "metadata", "radios"])
            st.dataframe(get_schema(year_selected, schema_table), use_container_width=True)

    with tab_sql:
        st.subheader("SQL avanzado")
        st.caption(
            "Podés consultar los Parquet remotos directamente. Alias sugeridos: census-data, metadata y radios."
        )
        if variable_selected in RADIOS_NUMERIC_VARIABLES and (categoria_value == "__ALL__"):
            default_sql = f"""
SELECT
    {sql_norm_code('PROV', 2)} AS provincia_codigo,
    {sql_norm_code('DEPTO', 3)} AS departamento_codigo,
    {JOIN_COL_BY_YEAR[year_selected]} AS id_geo,
    {variable_selected} AS conteo
FROM read_parquet('{source_url(year_selected, 'radios')}')
WHERE {build_where_radios(provincia_code, departamento_code)}
ORDER BY conteo DESC
LIMIT 20
""".strip()
        else:
            default_sql = f"""
SELECT *
FROM read_parquet('{source_url(year_selected, 'census')}')
WHERE codigo_variable = '{variable_selected}'
  AND {build_where_long(provincia_code=provincia_code, departamento_code=departamento_code, categoria_value=categoria_value)}
LIMIT 20
""".strip()
        sql_custom = st.text_area("Consulta DuckDB", value=default_sql, height=220)
        if st.button("Ejecutar SQL"):
            try:
                st.dataframe(run_sql(sql_custom), use_container_width=True)
            except Exception as exc:
                st.error("Error ejecutando SQL.")
                st.exception(exc)
