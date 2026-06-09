#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Consulta del Censo Argentino desde Source Cooperative sin QGIS.

Permite:
- Leer metadata, datos censales y radios censales remotos en Parquet/GeoParquet.
- Consultar variables censales con DuckDB.
- Filtrar por provincia/departamento/categoría.
- Agregar por provincia, departamento, radio, categoría o variable.
- Exportar resultados tabulares a CSV/Parquet/XLSX.
- Exportar una capa geográfica uniendo radios censales + resultado censal.

Instalación mínima:
    pip install duckdb pandas pyarrow openpyxl

Para exportar capas geográficas:
    pip install geopandas pyogrio shapely

Ejemplos:
    python consulta_censo_argentino.py variables --year 2022 --search poblacion

    python consulta_censo_argentino.py query \
        --year 2022 \
        --variable POB_TOT_P \
        --provincia "Buenos Aires" \
        --group departamento \
        --out outputs/poblacion_ba_departamento.csv

    python consulta_censo_argentino.py geo \
        --year 2022 \
        --variable POB_TOT_P \
        --provincia "Buenos Aires" \
        --out outputs/radios_ba_poblacion.gpkg
"""

from __future__ import annotations

import argparse
import hashlib
import os
import sys
from pathlib import Path
from typing import Iterable

import duckdb
import pandas as pd
import requests


VALID_YEARS = {1991, 2001, 2010, 2022}
BASE_URL = "https://data.source.coop/nlebovits/censo-argentino/{year}/{filename}"
CACHE_DIR = Path(os.getenv("CENSO_CACHE_DIR", "/tmp/censo_argentino_cache"))

FILES = {
    "census": "census-data.parquet",
    "metadata": "metadata.parquet",
    "radios": "radios.parquet",
}

# Según la documentación del dataset, el join es:
# radios.parquet COD_YYYY <-> census-data.parquet id_geo
JOIN_COL_BY_YEAR = {
    1991: "COD_1991",
    2001: "COD_2001",
    2010: "COD_2010",
    2022: "COD_2022",
}

# Columnas esperadas del archivo census-data.parquet en formato largo.
# Si el dataset cambia, el comando `schema` permite inspeccionar nombres reales.
EXPECTED_CENSUS_COLS = {
    "id_geo",
    "valor_provincia",
    "etiqueta_provincia",
    "valor_departamento",
    "etiqueta_departamento",
    "codigo_variable",
    "valor_categoria",
    "etiqueta_categoria",
    "conteo",
}

GROUP_MAP = {
    "provincia": ["valor_provincia", "etiqueta_provincia"],
    "departamento": [
        "valor_provincia",
        "etiqueta_provincia",
        "valor_departamento",
        "etiqueta_departamento",
    ],
    "radio": ["id_geo"],
    "categoria": ["codigo_variable", "valor_categoria", "etiqueta_categoria"],
    "variable": ["codigo_variable"],
}


def die(message: str, exit_code: int = 1) -> None:
    print(f"ERROR: {message}", file=sys.stderr)
    raise SystemExit(exit_code)


def validate_year(year: int) -> None:
    if year not in VALID_YEARS:
        die(f"Año inválido: {year}. Usá uno de: {sorted(VALID_YEARS)}")


def source_url(year: int, file_key: str) -> str:
    validate_year(year)
    if file_key not in FILES:
        die(f"Archivo inválido: {file_key}. Usá uno de: {list(FILES)}")
    return BASE_URL.format(year=year, filename=FILES[file_key])


def q(value: str) -> str:
    """Escapa strings para incluirlos en SQL."""
    return "'" + value.replace("'", "''") + "'"


def connect_duckdb(load_spatial: bool = False) -> duckdb.DuckDBPyConnection:
    con = duckdb.connect(database=":memory:")

    # httpfs permite leer Parquet remoto vía HTTPS/S3.
    con.execute("INSTALL httpfs;")
    con.execute("LOAD httpfs;")

    # Mejora rendimiento local.
    con.execute("SET threads TO 4;")

    # Spatial es opcional. Para la mayoría de las consultas tabulares no hace falta.
    if load_spatial:
        con.execute("INSTALL spatial;")
        con.execute("LOAD spatial;")

    return con


def read_schema(con: duckdb.DuckDBPyConnection, year: int, table: str) -> pd.DataFrame:
    url = source_url(year, table)
    sql = f"DESCRIBE SELECT * FROM read_parquet({q(url)})"
    return con.execute(sql).df()


def read_metadata(
    con: duckdb.DuckDBPyConnection,
    year: int,
    search: str | None = None,
    limit: int = 200,
) -> pd.DataFrame:
    url = source_url(year, "metadata")
    df = con.execute(f"SELECT * FROM read_parquet({q(url)})").df()

    if search:
        term = search.lower().strip()
        mask = df.astype(str).apply(
            lambda col: col.str.lower().str.contains(term, na=False, regex=False), axis=0
        ).any(axis=1)
        df = df.loc[mask].copy()

    return df.head(limit)


def get_census_columns(con: duckdb.DuckDBPyConnection, year: int) -> list[str]:
    schema = read_schema(con, year, "census")
    return schema["column_name"].tolist()


def assert_census_columns(con: duckdb.DuckDBPyConnection, year: int) -> None:
    cols = set(get_census_columns(con, year))
    missing = sorted(EXPECTED_CENSUS_COLS - cols)
    if missing:
        print(
            "AVISO: el archivo census-data.parquet no tiene algunas columnas esperadas: "
            + ", ".join(missing),
            file=sys.stderr,
        )
        print(
            "Ejecutá `python consulta_censo_argentino.py schema --year "
            f"{year} --table census` para revisar el esquema real.",
            file=sys.stderr,
        )


def build_where(
    variable: str | None = None,
    provincia: str | None = None,
    departamento: str | None = None,
    categoria: str | None = None,
    id_geo: str | None = None,
) -> str:
    filters: list[str] = []

    if variable:
        filters.append(f"codigo_variable = {q(variable)}")

    if provincia:
        # Permite pasar nombre o código de provincia.
        filters.append(
            "(lower(etiqueta_provincia) = lower({p}) OR valor_provincia = {p})".format(
                p=q(provincia)
            )
        )

    if departamento:
        # Permite pasar nombre o código de departamento.
        filters.append(
            "(lower(etiqueta_departamento) = lower({d}) OR valor_departamento = {d})".format(
                d=q(departamento)
            )
        )

    if categoria:
        # Permite pasar etiqueta o valor de categoría.
        filters.append(
            "(lower(etiqueta_categoria) = lower({c}) OR valor_categoria = {c})".format(
                c=q(categoria)
            )
        )

    if id_geo:
        filters.append(f"id_geo = {q(id_geo)}")

    return " AND ".join(filters) if filters else "1 = 1"


def list_geographies(
    con: duckdb.DuckDBPyConnection,
    year: int,
    provincia: str | None = None,
    limit: int = 500,
) -> pd.DataFrame:
    census_url = source_url(year, "census")
    where = build_where(provincia=provincia)

    sql = f"""
    SELECT DISTINCT
        valor_provincia,
        etiqueta_provincia,
        valor_departamento,
        etiqueta_departamento
    FROM read_parquet({q(census_url)})
    WHERE {where}
      AND valor_provincia IS NOT NULL
      AND etiqueta_provincia IS NOT NULL
      AND valor_departamento IS NOT NULL
      AND etiqueta_departamento IS NOT NULL
    ORDER BY etiqueta_provincia, etiqueta_departamento
    LIMIT {int(limit)}
    """

    return con.execute(sql).df()


def query_census(
    con: duckdb.DuckDBPyConnection,
    year: int,
    variable: str | None = None,
    provincia: str | None = None,
    departamento: str | None = None,
    categoria: str | None = None,
    id_geo: str | None = None,
    group: str = "departamento",
    limit: int | None = None,
) -> pd.DataFrame:
    assert_census_columns(con, year)

    if group not in GROUP_MAP:
        die(f"Grupo inválido: {group}. Usá uno de: {list(GROUP_MAP)}")

    census_url = source_url(year, "census")
    group_cols = GROUP_MAP[group]
    select_cols = ",\n        ".join(group_cols)
    group_by = ", ".join(str(i + 1) for i in range(len(group_cols)))
    where = build_where(
        variable=variable,
        provincia=provincia,
        departamento=departamento,
        categoria=categoria,
        id_geo=id_geo,
    )

    limit_clause = f"LIMIT {int(limit)}" if limit else ""

    sql = f"""
    SELECT
        {select_cols},
        SUM(conteo) AS conteo
    FROM read_parquet({q(census_url)})
    WHERE {where}
    GROUP BY {group_by}
    ORDER BY conteo DESC
    {limit_clause}
    """

    return con.execute(sql).df()


def save_table(df: pd.DataFrame, out: str | Path) -> None:
    out_path = Path(out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    suffix = out_path.suffix.lower()

    if suffix == ".csv":
        df.to_csv(out_path, index=False, encoding="utf-8-sig")
    elif suffix == ".parquet":
        df.to_parquet(out_path, index=False)
    elif suffix in {".xlsx", ".xls"}:
        df.to_excel(out_path, index=False)
    else:
        die("Formato de salida no soportado. Usá .csv, .parquet o .xlsx")

    print(f"Archivo guardado: {out_path}")


def export_geo_layer(
    con: duckdb.DuckDBPyConnection,
    year: int,
    variable: str,
    out: str | Path,
    provincia: str | None = None,
    departamento: str | None = None,
    categoria: str | None = None,
) -> None:
    try:
        import geopandas as gpd
    except ImportError as exc:
        raise SystemExit(
            "Para exportar geometrías instalá dependencias geográficas:\n"
            "    pip install geopandas pyogrio shapely"
        ) from exc

    join_col = JOIN_COL_BY_YEAR[year]
    radios_url = source_url(year, "radios")

    # Agregamos a nivel radio para poder unir contra radios.parquet.
    df = query_census(
        con=con,
        year=year,
        variable=variable,
        provincia=provincia,
        departamento=departamento,
        categoria=categoria,
        group="radio",
    )

    df = df.rename(columns={"id_geo": join_col})
    df[join_col] = df[join_col].astype(str)

    print("Descargando/leyendo radios censales GeoParquet...")
    radios_local = download_remote_parquet(radios_url)
    gdf = gpd.read_parquet(radios_local)

    if join_col not in gdf.columns:
        die(f"No encontré la columna {join_col} en radios.parquet")

    gdf[join_col] = gdf[join_col].astype(str)
    out_gdf = gdf.merge(df, on=join_col, how="left")

    out_path = Path(out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    suffix = out_path.suffix.lower()

    if suffix == ".gpkg":
        out_gdf.to_file(out_path, driver="GPKG")
    elif suffix == ".geojson":
        out_gdf.to_file(out_path, driver="GeoJSON")
    elif suffix == ".parquet":
        out_gdf.to_parquet(out_path, index=False)
    else:
        die("Formato geográfico no soportado. Usá .gpkg, .geojson o .parquet")

    print(f"Capa geográfica guardada: {out_path}")
    print(f"Filas: {len(out_gdf):,}")


def print_df(df: pd.DataFrame, max_rows: int = 30) -> None:
    if df.empty:
        print("Sin resultados.")
        return
    with pd.option_context(
        "display.max_rows",
        max_rows,
        "display.max_columns",
        50,
        "display.width",
        180,
        "display.max_colwidth",
        80,
    ):
        print(df.head(max_rows).to_string(index=False))


def parse_args(argv: Iterable[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Consulta el Censo Argentino en Source Cooperative sin QGIS usando DuckDB."
    )
    sub = parser.add_subparsers(dest="command", required=True)

    # schema
    p_schema = sub.add_parser("schema", help="Ver esquema de una tabla remota")
    p_schema.add_argument("--year", type=int, default=2022, choices=sorted(VALID_YEARS))
    p_schema.add_argument(
        "--table", choices=["census", "metadata", "radios"], default="census"
    )

    # variables
    p_vars = sub.add_parser("variables", help="Buscar variables en metadata.parquet")
    p_vars.add_argument("--year", type=int, default=2022, choices=sorted(VALID_YEARS))
    p_vars.add_argument("--search", type=str, default=None, help="Texto a buscar")
    p_vars.add_argument("--limit", type=int, default=100)
    p_vars.add_argument("--out", type=str, default=None)

    # geographies
    p_geo_list = sub.add_parser("geografias", help="Listar provincias/departamentos")
    p_geo_list.add_argument("--year", type=int, default=2022, choices=sorted(VALID_YEARS))
    p_geo_list.add_argument("--provincia", type=str, default=None)
    p_geo_list.add_argument("--limit", type=int, default=500)
    p_geo_list.add_argument("--out", type=str, default=None)

    # query
    p_query = sub.add_parser("query", help="Consultar y agregar datos censales")
    p_query.add_argument("--year", type=int, default=2022, choices=sorted(VALID_YEARS))
    p_query.add_argument("--variable", type=str, default=None, help="Ej: POB_TOT_P")
    p_query.add_argument("--provincia", type=str, default=None, help="Nombre o código")
    p_query.add_argument("--departamento", type=str, default=None, help="Nombre o código")
    p_query.add_argument("--categoria", type=str, default=None, help="Etiqueta o valor")
    p_query.add_argument("--id-geo", type=str, default=None)
    p_query.add_argument(
        "--group",
        choices=["provincia", "departamento", "radio", "categoria", "variable"],
        default="departamento",
    )
    p_query.add_argument("--limit", type=int, default=None)
    p_query.add_argument("--out", type=str, default=None)

    # geo export
    p_geo = sub.add_parser("geo", help="Exportar radios censales + variable como capa GIS")
    p_geo.add_argument("--year", type=int, default=2022, choices=sorted(VALID_YEARS))
    p_geo.add_argument("--variable", type=str, required=True, help="Ej: POB_TOT_P")
    p_geo.add_argument("--provincia", type=str, default=None, help="Nombre o código")
    p_geo.add_argument("--departamento", type=str, default=None, help="Nombre o código")
    p_geo.add_argument("--categoria", type=str, default=None, help="Etiqueta o valor")
    p_geo.add_argument(
        "--out",
        type=str,
        required=True,
        help="Archivo .gpkg, .geojson o .parquet",
    )

    return parser.parse_args(argv)


def main(argv: Iterable[str] | None = None) -> None:
    args = parse_args(argv)
    con = connect_duckdb(load_spatial=False)

    if args.command == "schema":
        df = read_schema(con, args.year, args.table)
        print_df(df, max_rows=200)
        return

    if args.command == "variables":
        df = read_metadata(con, args.year, args.search, args.limit)
        print_df(df, max_rows=args.limit)
        if args.out:
            save_table(df, args.out)
        return

    if args.command == "geografias":
        df = list_geographies(con, args.year, args.provincia, args.limit)
        print_df(df, max_rows=args.limit)
        if args.out:
            save_table(df, args.out)
        return

    if args.command == "query":
        df = query_census(
            con=con,
            year=args.year,
            variable=args.variable,
            provincia=args.provincia,
            departamento=args.departamento,
            categoria=args.categoria,
            id_geo=args.id_geo,
            group=args.group,
            limit=args.limit,
        )
        print_df(df)
        if args.out:
            save_table(df, args.out)
        return

    if args.command == "geo":
        export_geo_layer(
            con=con,
            year=args.year,
            variable=args.variable,
            provincia=args.provincia,
            departamento=args.departamento,
            categoria=args.categoria,
            out=args.out,
        )
        return

    die(f"Comando no reconocido: {args.command}")


if __name__ == "__main__":
    main()
