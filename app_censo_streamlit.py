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


def build_bar_chart(df: pd.DataFrame, label_col: str, top_n: int):
    chart_df = enrich_result(df).head(int(top_n))
    return (
        alt.Chart(chart_df)
        .mark_bar()
        .encode(
            x=alt.X("conteo:Q", title="Valor"),
            y=alt.Y(f"{label_col}:N", sort="-x", title=""),
            tooltip=[c for c in chart_df.columns if c != "conteo_acum"],
        )
        .properties(height=max(300, min(850, 24 * len(chart_df))))
    )


def build_pareto_chart(df: pd.DataFrame, label_col: str, top_n: int):
    chart_df = enrich_result(df).head(int(top_n))
    base = alt.Chart(chart_df).encode(x=alt.X(f"{label_col}:N", sort="-y", title=""))
    bars = base.mark_bar(opacity=0.75).encode(
        y=alt.Y("conteo:Q", title="Valor"),
        tooltip=[label_col, "conteo", alt.Tooltip("pct_total:Q", format=".1%"), alt.Tooltip("pct_acum:Q", format=".1%")],
    )
    line = base.mark_line(point=True).encode(
        y=alt.Y("pct_acum:Q", title="% acumulado", axis=alt.Axis(format="%")),
        tooltip=[label_col, alt.Tooltip("pct_acum:Q", format=".1%")],
    )
    return (bars + line).resolve_scale(y="independent").properties(height=420)


def build_distribution_chart(df: pd.DataFrame):
    chart_df = enrich_result(df)
    return (
        alt.Chart(chart_df)
        .mark_bar()
        .encode(
            x=alt.X("conteo:Q", bin=alt.Bin(maxbins=35), title="Valor"),
            y=alt.Y("count():Q", title="Cantidad de áreas"),
            tooltip=[alt.Tooltip("count():Q", title="Áreas")],
        )
        .properties(height=360)
    )


def build_share_chart(df: pd.DataFrame, label_col: str, top_n: int):
    chart_df = enrich_result(df).head(int(top_n)).copy()
    chart_df["pct_label"] = chart_df["pct_total"]
    return (
        alt.Chart(chart_df)
        .mark_bar()
        .encode(
            x=alt.X("pct_label:Q", title="Participación", axis=alt.Axis(format="%")),
            y=alt.Y(f"{label_col}:N", sort="-x", title=""),
            tooltip=[label_col, "conteo", alt.Tooltip("pct_total:Q", format=".1%")],
        )
        .properties(height=max(300, min(850, 24 * len(chart_df))))
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


@st.cache_data(show_spinner=True, ttl=60 * 30)
def query_censo_resilient(
    year: int,
    variable: str,
    provincia_code: str | None,
    departamento_code: str | None,
    categoria_value: str | None,
    group_label: str,
    limit: int,
) -> tuple[pd.DataFrame, str]:
    """Consulta principal. Si la tabla larga no tiene el total solicitado, cae a radios.parquet."""
    # Para variables de radios, conviene ir directo al archivo geográfico si no hay categoría.
    if variable in RADIOS_NUMERIC_VARIABLES and (not categoria_value or categoria_value == "__ALL__"):
        df_radios = query_censo_radios(
            year, variable, provincia_code, departamento_code, group_label, limit
        )
        if not df_radios.empty:
            return df_radios, "radios.parquet"

    df_long = query_censo_long(
        year, variable, provincia_code, departamento_code, categoria_value, group_label, limit
    )
    if not df_long.empty:
        return df_long, "census-data.parquet"

    # Fallback final para POB_TOT_P / VIV_TOT_P si la primera consulta fue vacía.
    if variable in RADIOS_NUMERIC_VARIABLES and (not categoria_value or categoria_value == "__ALL__"):
        df_radios = query_censo_radios(
            year, variable, provincia_code, departamento_code, group_label, limit
        )
        if not df_radios.empty:
            return df_radios, "radios.parquet"

    return df_long, "census-data.parquet"


@st.cache_data(show_spinner=True, ttl=60 * 30)
def query_radio_counts(
    year: int,
    variable: str,
    provincia_code: str | None,
    departamento_code: str | None,
    categoria_value: str | None,
) -> tuple[pd.DataFrame, str]:
    """Conteo a nivel radio para mapa."""
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
        return run_sql(sql), "radios.parquet"

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
    return run_sql(sql), "census-data.parquet"


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


def build_choropleth_map(
    gdf,
    year: int,
    value_col: str = "conteo",
    simplify_tolerance: float = 0.0002,
    metric_mode: str = "Valor total",
    show_polygons: bool = True,
    show_bubbles: bool = False,
    show_labels: bool = False,
    label_top_n: int = 20,
    extruded: bool = False,
):
    if pdk is None:
        raise RuntimeError("Falta pydeck. Instalá: pip install pydeck")

    if gdf.empty:
        return None, pd.DataFrame()

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

    min_val = float(gdf[metric_col].min())
    max_val = float(gdf[metric_col].max())
    denom = max(max_val - min_val, 1.0)
    gdf["_norm"] = ((gdf[metric_col] - min_val) / denom).clip(0, 1)

    gdf["fill_color"] = gdf["_norm"].apply(
        lambda x: [int(55 + 180 * x), int(120 - 70 * x), int(220 - 120 * x), 150]
    )
    gdf["elevation"] = (gdf["_norm"] * 2500).fillna(0)

    centroids = gdf.geometry.centroid
    gdf["lon"] = centroids.x
    gdf["lat"] = centroids.y
    gdf["radio_burbuja"] = (80 + 900 * np.sqrt(gdf["_norm"].clip(0, 1))).fillna(80)
    gdf["label"] = gdf[value_col].round(0).astype("Int64").astype(str)

    latitude = float(gdf["lat"].mean())
    longitude = float(gdf["lon"].mean())

    join_col = JOIN_COL_BY_YEAR[year]
    export_cols = unique_existing_columns(
        [join_col, value_col, metric_col, "area_km2", "valor_por_km2", "fill_color", "elevation", "geometry"],
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
                get_line_color=[80, 80, 80, 80],
                line_width_min_pixels=0.2,
            )
        )

    point_cols = unique_existing_columns(
        [join_col, value_col, metric_col, "area_km2", "valor_por_km2", "lon", "lat", "radio_burbuja", "label"],
        gdf.columns,
    )
    points_df = pd.DataFrame(drop_duplicated_columns_keep_first(gdf[point_cols]))

    if show_bubbles:
        layers.append(
            pdk.Layer(
                "ScatterplotLayer",
                points_df,
                pickable=True,
                get_position="[lon, lat]",
                get_radius="radio_burbuja",
                get_fill_color=[30, 90, 180, 130],
                get_line_color=[20, 20, 20, 120],
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
                get_color=[20, 20, 20, 230],
                get_angle=0,
                get_text_anchor="middle",
                get_alignment_baseline="center",
            )
        )

    view_state = pdk.ViewState(
        latitude=latitude,
        longitude=longitude,
        zoom=8 if len(gdf) < 1500 else 6,
        pitch=35 if extruded else 0,
    )

    tooltip = {
        "html": (
            "<b>Radio:</b> {" + join_col + "}<br/>"
            "<b>Valor:</b> {" + value_col + "}<br/>"
            "<b>Km²:</b> {area_km2}<br/>"
            "<b>Valor/km²:</b> {valor_por_km2}<br/>"
            f"<b>Capa:</b> {metric_title}"
        ),
        "style": {"backgroundColor": "white", "color": "black"},
    }

    deck = pdk.Deck(layers=layers, initial_view_state=view_state, tooltip=tooltip)
    summary_cols = unique_existing_columns([join_col, value_col, "area_km2", "valor_por_km2", "lon", "lat"], gdf.columns)
    map_summary = pd.DataFrame(drop_duplicated_columns_keep_first(gdf[summary_cols])).copy()
    return deck, map_summary


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
    variable_search = st.text_input(
        "Buscar variable",
        value="pob",
        help="Busca sobre las variables presentes en census-data.parquet. También incluye POB_TOT_P/VIV_TOT_P desde radios.parquet.",
    )

    variables_df = get_census_variables(year_selected, variable_search, limit=500)
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
    selected_variable_label = st.selectbox(
        "Variable censal",
        list(variable_label_to_code.keys()),
        index=0,
    )
    variable_selected = variable_label_to_code.get(selected_variable_label, "POB_TOT_P")

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

    st.divider()
    st.subheader("Categoría")
    categorias = ensure_dataframe(get_categorias(year_selected, variable_selected), "categorías")
    cat_options = {"Todas": "__ALL__"}
    if not categorias.empty and "valor_categoria" in categorias.columns:
        for _, row in categorias.iterrows():
            label = f"{row.get('etiqueta_categoria', '')} ({row.get('valor_categoria', '')})"
            cat_options[label] = str(row["valor_categoria"])
    selected_cat_label = st.selectbox("Categoría", list(cat_options.keys()))
    categoria_value = cat_options[selected_cat_label]

    st.divider()
    group_options = list(GROUP_MAP_LONG.keys())
    group_selected = st.selectbox("Agrupar por", group_options, index=1)
    limit_rows = st.slider("Máximo de filas", min_value=10, max_value=5000, value=500, step=10)


col_a, col_b, col_c, col_d = st.columns(4)
col_a.metric("Año", str(year_selected))
col_b.metric("Variable", variable_selected)
col_c.metric("Provincia", "Todas" if provincia_code == "__ALL__" else selected_prov_label.split(" (")[0])
col_d.metric("Agrupación", group_selected)

try:
    df_result, query_source = query_censo_resilient(
        year=year_selected,
        variable=variable_selected,
        provincia_code=provincia_code,
        departamento_code=departamento_code,
        categoria_value=categoria_value,
        group_label=group_selected,
        limit=limit_rows,
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
    label_col = get_label_column(df_enriched)
    summary = summarize_result(df_enriched, label_col)

    col_total, col_source = st.columns([1, 2])
    col_total.metric("Total según consulta", format_number(summary["total"]))
    col_source.info(f"Fuente usada para esta consulta: {query_source}")

    tab_resumen, tab_graficos, tab_mapa, tab_metadata, tab_sql = st.tabs(
        ["Resumen", "Gráficos", "Mapa y capas", "Metadata", "SQL avanzado"]
    )

    with tab_resumen:
        st.subheader("Resumen ejecutivo")
        kpi1, kpi2, kpi3, kpi4 = st.columns(4)
        kpi1.metric("Total", format_number(summary["total"]))
        kpi2.metric("Áreas", format_number(summary["n_areas"]))
        kpi3.metric("Promedio por área", format_number(summary["mean_value"]))
        kpi4.metric("Mediana por área", format_number(summary["median_value"]))

        st.caption(
            f"Área con mayor valor: **{summary['top_label']}** "
            f"con {format_number(summary['top_value'])} "
            f"({summary['top_share']:.1%} del total)."
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
            chart_a, chart_b = st.columns(2)
            with chart_a:
                st.markdown("**Ranking de áreas**")
                st.altair_chart(build_bar_chart(df_enriched, label_col, top_n_chart), use_container_width=True)
            with chart_b:
                st.markdown("**Participación sobre el total**")
                st.altair_chart(build_share_chart(df_enriched, label_col, top_n_chart), use_container_width=True)

            st.markdown("**Pareto: acumulado del top seleccionado**")
            st.altair_chart(build_pareto_chart(df_enriched, label_col, top_n_chart), use_container_width=True)

            st.markdown("**Distribución de valores entre áreas**")
            st.altair_chart(build_distribution_chart(df_enriched), use_container_width=True)

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

        if gpd is None or pdk is None:
            st.warning(
                "Para activar el mapa instalá dependencias geográficas: geopandas, pyogrio, shapely y pydeck."
            )
        elif provincia_code == "__ALL__":
            st.info("Seleccioná una provincia para generar el mapa sin cargar todo el país.")
        elif group_selected == "Categoría" and variable_selected in RADIOS_NUMERIC_VARIABLES:
            st.info("Las variables provenientes de radios.parquet no tienen categorías. Cambiá la agrupación para mapear.")
        else:
            cfg1, cfg2, cfg3 = st.columns(3)
            with cfg1:
                metric_mode = st.selectbox(
                    "Métrica de color",
                    ["Valor total", "Valor por km²", "Log(valor + 1)"],
                    index=0,
                    help="Valor por km² sirve como aproximación de densidad para población/viviendas."
                )
                simplify_tolerance = st.slider(
                    "Simplificación geométrica",
                    min_value=0.0,
                    max_value=0.004,
                    value=0.0004,
                    step=0.0001,
                    help="Valores más altos alivian el mapa, pero reducen detalle de polígonos.",
                )
            with cfg2:
                show_polygons = st.checkbox("Capa polígonos", value=True)
                show_bubbles = st.checkbox("Capa burbujas", value=False)
                show_labels = st.checkbox("Etiquetas top radios", value=False)
            with cfg3:
                extruded = st.checkbox("Vista 3D", value=False)
                label_top_n = st.slider("Cantidad de etiquetas", 5, 100, 20, 5)

            if st.button("Generar mapa", type="primary"):
                try:
                    with st.spinner("Consultando radios y geometrías..."):
                        radio_counts, radio_source = query_radio_counts(
                            year_selected,
                            variable_selected,
                            provincia_code,
                            departamento_code,
                            categoria_value,
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

                    if gdf_plot.empty:
                        st.warning("El join entre radios y datos censales quedó vacío.")
                    else:
                        st.caption(f"Fuente del mapa: {radio_source}")
                        deck, map_summary = build_choropleth_map(
                            gdf_plot,
                            year=year_selected,
                            value_col="conteo",
                            simplify_tolerance=simplify_tolerance,
                            metric_mode=metric_mode,
                            show_polygons=show_polygons,
                            show_bubbles=show_bubbles,
                            show_labels=show_labels,
                            label_top_n=label_top_n,
                            extruded=extruded,
                        )
                        st.pydeck_chart(deck, use_container_width=True)

                        m1, m2, m3 = st.columns(3)
                        m1.metric("Radios mapeados", format_number(len(gdf_plot)))
                        m2.metric("Área aproximada km²", format_number(map_summary["area_km2"].sum()))
                        m3.metric("Valor/km² promedio", format_number(map_summary["valor_por_km2"].mean()))

                        with st.expander("Tabla de radios mapeados"):
                            st.dataframe(
                                map_summary.sort_values("conteo", ascending=False).head(1000),
                                use_container_width=True,
                                hide_index=True,
                            )
                except Exception as exc:
                    st.error("No pude generar el mapa.")
                    st.exception(exc)

    with tab_metadata:
        st.subheader("Variables disponibles")
        st.caption("Esta tabla está basada en los códigos presentes en census-data.parquet y enriquecida con metadata.parquet.")
        st.dataframe(variables_df, use_container_width=True, hide_index=True)

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
