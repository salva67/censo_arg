# -*- coding: utf-8 -*-
"""
Dashboard Streamlit para consultar el Censo Argentino publicado en Source Cooperative
sin usar el plugin de QGIS.

Funcionalidades:
- Consulta remota de Parquet/GeoParquet con DuckDB.
- Filtros por año, variable censal, provincia, departamento y categoría.
- Agregación por provincia, departamento, categoría o radio.
- Tabla descargable en CSV.
- Gráfico de barras de principales resultados.
- Mapa opcional de radios censales usando GeoParquet + GeoPandas + PyDeck.
- Consola SQL avanzada para consultas directas.

Ejecutar:
    streamlit run app_censo_streamlit.py

Instalar dependencias:
    pip install -r requirements_censo_streamlit.txt
"""

from __future__ import annotations

import json
from typing import Iterable

import altair as alt
import duckdb
import pandas as pd
import streamlit as st

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

GROUP_MAP = {
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

DEFAULT_VARIABLE_HINTS = [
    "POB_TOT_P",
    "VIV_TOT_P",
]


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
    con.execute("INSTALL httpfs;")
    con.execute("LOAD httpfs;")
    con.execute("SET threads TO 4;")
    return con


@st.cache_data(show_spinner=False, ttl=60 * 60)
def run_sql(sql: str) -> pd.DataFrame:
    con = get_connection()
    return con.execute(sql).df()


def build_where(
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


def safe_sort_cols(df: pd.DataFrame, preferred: Iterable[str]) -> list[str]:
    return [c for c in preferred if c in df.columns]


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
def get_provincias(year: int) -> pd.DataFrame:
    url = source_url(year, "census")
    sql = f"""
    SELECT DISTINCT
        valor_provincia,
        etiqueta_provincia
    FROM read_parquet({sql_quote(url)})
    WHERE valor_provincia IS NOT NULL
      AND etiqueta_provincia IS NOT NULL
    ORDER BY etiqueta_provincia
    """
    return run_sql(sql)


@st.cache_data(show_spinner=False, ttl=60 * 60)
def get_departamentos(year: int, provincia_code: str | None) -> pd.DataFrame:
    url = source_url(year, "census")
    where = build_where(provincia_code=provincia_code)
    sql = f"""
    SELECT DISTINCT
        valor_departamento,
        etiqueta_departamento
    FROM read_parquet({sql_quote(url)})
    WHERE {where}
      AND valor_departamento IS NOT NULL
      AND etiqueta_departamento IS NOT NULL
    ORDER BY etiqueta_departamento
    """
    return run_sql(sql)


@st.cache_data(show_spinner=False, ttl=60 * 60)
def get_categorias(year: int, variable: str | None) -> pd.DataFrame:
    if not variable:
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
def query_censo(
    year: int,
    variable: str,
    provincia_code: str | None,
    departamento_code: str | None,
    categoria_value: str | None,
    group_label: str,
    limit: int,
) -> pd.DataFrame:
    url = source_url(year, "census")
    group_cols = GROUP_MAP[group_label]
    where = build_where(
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
def query_radio_counts(
    year: int,
    variable: str,
    provincia_code: str | None,
    departamento_code: str | None,
    categoria_value: str | None,
) -> pd.DataFrame:
    url = source_url(year, "census")
    where = build_where(
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
    return run_sql(sql)


# -----------------------------------------------------------------------------
# Mapa
# -----------------------------------------------------------------------------


def find_column_case_insensitive(columns: Iterable[str], wanted: str) -> str | None:
    wanted_lower = wanted.lower()
    for col in columns:
        if col.lower() == wanted_lower:
            return col
    return None


@st.cache_data(show_spinner=True, ttl=60 * 60)
def load_radios_geoparquet(year: int, provincia_code: str | None, departamento_code: str | None):
    if gpd is None:
        raise RuntimeError(
            "Faltan dependencias geográficas. Instalá: pip install geopandas pyogrio shapely pydeck"
        )

    url = source_url(year, "radios")
    join_col = JOIN_COL_BY_YEAR[year]

    # Intentamos leer sólo la provincia/departamento elegido usando filtros parquet.
    # Si el tipo de dato del filtro no coincide, caemos a lectura completa y filtrado local.
    filters = []
    if provincia_code and provincia_code != "__ALL__":
        filters.append(("PROV", "=", str(provincia_code)))
    if departamento_code and departamento_code != "__ALL__":
        filters.append(("DEPTO", "=", str(departamento_code)))

    columns = [join_col, "PROV", "DEPTO", "geometry"]

    try:
        if filters:
            gdf = gpd.read_parquet(url, columns=columns, filters=filters)
        else:
            gdf = gpd.read_parquet(url, columns=columns)
    except Exception:
        gdf = gpd.read_parquet(url)

    join_col_actual = find_column_case_insensitive(gdf.columns, join_col)
    if join_col_actual is None:
        raise RuntimeError(f"No encontré la columna de join {join_col} en radios.parquet")

    if provincia_code and provincia_code != "__ALL__" and "PROV" in gdf.columns:
        gdf = gdf[gdf["PROV"].astype(str) == str(provincia_code)]
    if departamento_code and departamento_code != "__ALL__" and "DEPTO" in gdf.columns:
        gdf = gdf[gdf["DEPTO"].astype(str) == str(departamento_code)]

    if join_col_actual != join_col:
        gdf = gdf.rename(columns={join_col_actual: join_col})

    gdf[join_col] = gdf[join_col].astype(str)
    return gdf


def build_choropleth_map(gdf, year: int, value_col: str = "conteo", simplify_tolerance: float = 0.0002):
    if pdk is None:
        raise RuntimeError("Falta pydeck. Instalá: pip install pydeck")

    if gdf.empty:
        return None

    gdf = gdf.copy()

    if gdf.crs is None:
        gdf = gdf.set_crs(4326, allow_override=True)
    else:
        gdf = gdf.to_crs(4326)

    if simplify_tolerance and simplify_tolerance > 0:
        gdf["geometry"] = gdf.geometry.simplify(simplify_tolerance, preserve_topology=True)

    min_val = float(gdf[value_col].min())
    max_val = float(gdf[value_col].max())
    denom = max(max_val - min_val, 1.0)
    gdf["_norm"] = ((gdf[value_col] - min_val) / denom).clip(0, 1)

    # Escala simple: valores altos más intensos. Se guarda como propiedad GeoJSON.
    gdf["fill_color"] = gdf["_norm"].apply(
        lambda x: [int(60 + 160 * x), int(130 - 70 * x), int(220 - 120 * x), 145]
    )

    # Centro del mapa: centroides en WGS84.
    centroids = gdf.geometry.centroid
    latitude = float(centroids.y.mean())
    longitude = float(centroids.x.mean())

    geojson = json.loads(gdf[[JOIN_COL_BY_YEAR[year], value_col, "fill_color", "geometry"]].to_json())

    layer = pdk.Layer(
        "GeoJsonLayer",
        geojson,
        pickable=True,
        stroked=True,
        filled=True,
        get_fill_color="properties.fill_color",
        get_line_color=[80, 80, 80, 80],
        line_width_min_pixels=0.2,
    )

    view_state = pdk.ViewState(
        latitude=latitude,
        longitude=longitude,
        zoom=7 if len(gdf) < 5000 else 5,
        pitch=0,
    )

    tooltip = {
        "html": "<b>Radio:</b> {" + JOIN_COL_BY_YEAR[year] + "}<br/><b>Conteo:</b> {conteo}",
        "style": {"backgroundColor": "white", "color": "black"},
    }

    return pdk.Deck(layers=[layer], initial_view_state=view_state, tooltip=tooltip)


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
    variable_search = st.text_input("Buscar en metadata", value="poblacion")

    metadata = get_metadata(year_selected)
    metadata_filtered = metadata.copy()
    if variable_search.strip():
        term = variable_search.lower().strip()
        metadata_filtered = metadata_filtered[
            metadata_filtered.astype(str)
            .apply(lambda col: col.str.lower().str.contains(term, na=False, regex=False), axis=0)
            .any(axis=1)
        ].copy()

    code_col = infer_col(
        metadata,
        candidates=["codigo_variable", "cod_variable", "variable", "var", "code"],
        contains_any=["codigo", "variable"],
    )
    label_col = infer_col(
        metadata,
        candidates=["etiqueta_variable", "descripcion", "label", "pregunta"],
        contains_any=["etiqueta", "variable"],
    )

    variable_options = []
    if code_col and not metadata_filtered.empty:
        tmp = metadata_filtered[[c for c in [code_col, label_col] if c]].drop_duplicates().head(500)
        for _, row in tmp.iterrows():
            code = str(row[code_col])
            label = str(row[label_col]) if label_col and pd.notna(row[label_col]) else ""
            variable_options.append((code, f"{code} · {label}" if label else code))

    # Hints útiles si la metadata no permite inferir columnas.
    for hint in DEFAULT_VARIABLE_HINTS:
        if hint not in [v[0] for v in variable_options]:
            variable_options.insert(0, (hint, hint))

    variable_label_to_code = {label: code for code, label in variable_options}
    selected_variable_label = st.selectbox(
        "Variable censal",
        list(variable_label_to_code.keys()),
        index=0 if variable_label_to_code else None,
    )
    variable_selected = variable_label_to_code.get(selected_variable_label, "POB_TOT_P")

    manual_variable = st.text_input(
        "O escribir código exacto",
        value=variable_selected,
        help="Ejemplo frecuente: POB_TOT_P. Podés copiar códigos desde la tabla de metadata.",
    )
    variable_selected = manual_variable.strip() or variable_selected

    st.divider()
    st.subheader("Geografía")

    provincias = get_provincias(year_selected)
    prov_options = {"Todas": "__ALL__"}
    for _, row in provincias.iterrows():
        label = f"{row['etiqueta_provincia']} ({row['valor_provincia']})"
        prov_options[label] = str(row["valor_provincia"])

    selected_prov_label = st.selectbox("Provincia", list(prov_options.keys()))
    provincia_code = prov_options[selected_prov_label]

    dept_options = {"Todos": "__ALL__"}
    if provincia_code != "__ALL__":
        departamentos = get_departamentos(year_selected, provincia_code)
        for _, row in departamentos.iterrows():
            label = f"{row['etiqueta_departamento']} ({row['valor_departamento']})"
            dept_options[label] = str(row["valor_departamento"])

    selected_dept_label = st.selectbox("Departamento", list(dept_options.keys()))
    departamento_code = dept_options[selected_dept_label]

    st.divider()
    st.subheader("Categoría")
    categorias = get_categorias(year_selected, variable_selected)
    cat_options = {"Todas": "__ALL__"}
    if not categorias.empty:
        for _, row in categorias.iterrows():
            label = f"{row.get('etiqueta_categoria', '')} ({row.get('valor_categoria', '')})"
            cat_options[label] = str(row["valor_categoria"])
    selected_cat_label = st.selectbox("Categoría", list(cat_options.keys()))
    categoria_value = cat_options[selected_cat_label]

    st.divider()
    group_selected = st.selectbox("Agrupar por", list(GROUP_MAP.keys()), index=1)
    limit_rows = st.slider("Máximo de filas", min_value=10, max_value=5000, value=500, step=10)


# Panel principal
col_a, col_b, col_c, col_d = st.columns(4)
col_a.metric("Año", str(year_selected))
col_b.metric("Variable", variable_selected)
col_c.metric("Provincia", "Todas" if provincia_code == "__ALL__" else selected_prov_label.split(" (")[0])
col_d.metric("Agrupación", group_selected)

try:
    df_result = query_censo(
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
else:
    total = float(df_result["conteo"].sum())
    st.metric("Total según consulta", f"{total:,.0f}".replace(",", "."))

    tab_resumen, tab_metadata, tab_mapa, tab_sql = st.tabs(
        ["Resumen", "Metadata", "Mapa", "SQL avanzado"]
    )

    with tab_resumen:
        st.subheader("Resultados")
        st.dataframe(df_result, use_container_width=True, height=420)

        csv_bytes = df_result.to_csv(index=False).encode("utf-8-sig")
        st.download_button(
            label="Descargar CSV",
            data=csv_bytes,
            file_name=f"censo_{year_selected}_{variable_selected}_{group_selected.lower().replace(' ', '_')}.csv",
            mime="text/csv",
        )

        # Gráfico de barras para agregaciones con etiqueta legible.
        label_candidates = [
            "etiqueta_departamento",
            "etiqueta_provincia",
            "etiqueta_categoria",
            "id_geo",
            "codigo_variable",
        ]
        label_col_chart = next((c for c in label_candidates if c in df_result.columns), None)
        if label_col_chart:
            top_n = st.slider("Top N para gráfico", 5, min(50, max(5, len(df_result))), 20)
            chart_df = df_result.nlargest(top_n, "conteo").copy()
            chart = (
                alt.Chart(chart_df)
                .mark_bar()
                .encode(
                    x=alt.X("conteo:Q", title="Conteo"),
                    y=alt.Y(f"{label_col_chart}:N", sort="-x", title=""),
                    tooltip=list(chart_df.columns),
                )
                .properties(height=max(280, top_n * 22))
            )
            st.altair_chart(chart, use_container_width=True)

    with tab_metadata:
        st.subheader("Metadata de variables")
        st.caption("Usá esta tabla para buscar códigos de variables y categorías disponibles.")
        st.dataframe(metadata_filtered.head(1000), use_container_width=True, height=500)

        with st.expander("Esquemas de archivos remotos"):
            schema_table = st.selectbox("Archivo", ["census", "metadata", "radios"])
            st.dataframe(get_schema(year_selected, schema_table), use_container_width=True)

    with tab_mapa:
        st.subheader("Mapa de radios censales")
        st.caption(
            "Para evitar tiempos de carga altos, conviene filtrar al menos una provincia y, si es Buenos Aires/CABA, idealmente un departamento/comuna."
        )

        if gpd is None or pdk is None:
            st.warning(
                "Para activar el mapa instalá dependencias geográficas: geopandas, pyogrio, shapely y pydeck."
            )
        elif provincia_code == "__ALL__":
            st.info("Seleccioná una provincia para generar el mapa sin cargar todo el país.")
        else:
            simplify_tolerance = st.slider(
                "Simplificación geométrica",
                min_value=0.0,
                max_value=0.002,
                value=0.0002,
                step=0.0001,
                help="Valores más altos alivian el mapa, pero reducen detalle de polígonos.",
            )
            if st.button("Generar mapa", type="primary"):
                try:
                    with st.spinner("Consultando radios y geometrías..."):
                        radio_counts = query_radio_counts(
                            year_selected,
                            variable_selected,
                            provincia_code,
                            departamento_code,
                            categoria_value,
                        )
                        join_col = JOIN_COL_BY_YEAR[year_selected]
                        gdf = load_radios_geoparquet(year_selected, provincia_code, departamento_code)
                        radio_counts["id_geo"] = radio_counts["id_geo"].astype(str)
                        gdf[join_col] = gdf[join_col].astype(str)
                        gdf = gdf.merge(radio_counts, left_on=join_col, right_on="id_geo", how="inner")

                    if gdf.empty:
                        st.warning("No hubo geometrías para los filtros elegidos.")
                    else:
                        st.write(f"Radios en mapa: {len(gdf):,}".replace(",", "."))
                        if len(gdf) > 30000:
                            st.warning(
                                "El mapa tiene muchos radios. Puede tardar o quedar pesado; filtrá un departamento para trabajar mejor."
                            )
                        deck = build_choropleth_map(gdf, year_selected, simplify_tolerance=simplify_tolerance)
                        if deck:
                            st.pydeck_chart(deck, use_container_width=True)
                except Exception as exc:
                    st.error("No pude generar el mapa.")
                    st.exception(exc)

    with tab_sql:
        st.subheader("SQL avanzado")
        st.caption("Podés consultar directamente los Parquet remotos. Usá las URLs de abajo como referencia.")
        st.code(
            f"""
-- Datos censales
SELECT *
FROM read_parquet('{source_url(year_selected, 'census')}')
LIMIT 10;

-- Metadata
SELECT *
FROM read_parquet('{source_url(year_selected, 'metadata')}')
LIMIT 10;

-- Radios censales / geometrías
SELECT *
FROM read_parquet('{source_url(year_selected, 'radios')}')
LIMIT 10;
""".strip(),
            language="sql",
        )

        default_sql = f"""
SELECT
    etiqueta_departamento,
    SUM(conteo) AS total
FROM read_parquet('{source_url(year_selected, 'census')}')
WHERE codigo_variable = '{variable_selected}'
  AND valor_provincia = '{provincia_code}'
GROUP BY 1
ORDER BY total DESC
LIMIT 50
""".strip()
        user_sql = st.text_area("Consulta SQL", value=default_sql, height=220)
        if st.button("Ejecutar SQL"):
            try:
                custom_df = run_sql(user_sql)
                st.dataframe(custom_df, use_container_width=True, height=420)
                st.download_button(
                    "Descargar resultado SQL",
                    custom_df.to_csv(index=False).encode("utf-8-sig"),
                    file_name="consulta_sql_censo.csv",
                    mime="text/csv",
                )
            except Exception as exc:
                st.error("Error ejecutando SQL.")
                st.exception(exc)
