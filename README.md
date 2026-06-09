# Dashboard Censo Argentino · Streamlit + DuckDB

Dashboard para consultar datos del **Censo Argentino** publicados en Source Cooperative, sin usar QGIS. La app lee archivos Parquet/GeoParquet remotos con DuckDB y permite explorar variables censales, filtrar por geografía, descargar resultados y visualizar radios censales en un mapa.

## Qué incluye

- Consulta directa a Parquet remoto con DuckDB.
- Selector de año censal: 1991, 2001, 2010 y 2022.
- Buscador de variables censales desde `metadata.parquet`.
- Filtros por provincia, departamento y categoría.
- Agregación por provincia, departamento, categoría o radio censal.
- Descarga de resultados a CSV.
- Mapa de radios censales con GeoParquet + GeoPandas + PyDeck.
- Consola SQL avanzada dentro de Streamlit.
- Script CLI opcional para consultas y exportaciones.

## Estructura del repo

```text
.
├── app_censo_streamlit.py          # Dashboard Streamlit
├── consulta_censo_argentino.py     # CLI para consultas/exportaciones
├── requirements.txt                # Dependencias
├── .gitignore
├── .streamlit/
│   └── config.toml
└── outputs/
    └── .gitkeep
```

## Instalación local

```bash
python -m venv .venv

# Windows
.venv\Scripts\activate

# macOS / Linux
source .venv/bin/activate

pip install --upgrade pip
pip install -r requirements.txt
```

## Ejecutar dashboard

```bash
streamlit run app_censo_streamlit.py
```

Ejemplo sugerido para validar rápido:

- Año: `2022`
- Buscar metadata: `poblacion`
- Variable: `POB_TOT_P`
- Provincia: `Buenos Aires`
- Agrupar por: `Departamento`

Para el mapa conviene filtrar al menos una provincia y, si el resultado es pesado, un departamento.

## Uso del script CLI

Ver esquema de datos censales:

```bash
python consulta_censo_argentino.py schema --year 2022 --table census
```

Buscar variables:

```bash
python consulta_censo_argentino.py variables --year 2022 --search poblacion
```

Consultar población por departamento:

```bash
python consulta_censo_argentino.py query \
  --year 2022 \
  --variable POB_TOT_P \
  --provincia "Buenos Aires" \
  --group departamento \
  --out outputs/poblacion_ba_departamento.csv
```

Exportar capa geográfica:

```bash
python consulta_censo_argentino.py geo \
  --year 2022 \
  --variable POB_TOT_P \
  --provincia "Buenos Aires" \
  --out outputs/radios_ba_poblacion.gpkg
```

## Deploy en Streamlit Community Cloud

1. Crear un repositorio en GitHub.
2. Subir estos archivos al repo.
3. Entrar a Streamlit Community Cloud.
4. Crear una app nueva apuntando a:
   - Repository: tu repo.
   - Branch: `main`.
   - Main file path: `app_censo_streamlit.py`.
5. Deploy.

No requiere secrets ni credenciales porque consulta datos públicos remotos.

## Notas técnicas

- La app usa `httpfs` de DuckDB para leer Parquet por HTTPS.
- Las geometrías se leen desde `radios.parquet`.
- El join esperado es `radios.COD_YYYY = census.id_geo`, donde `YYYY` es el año censal.
- Los archivos grandes pueden tardar al cargar mapas completos. Para mejor rendimiento, filtrar por provincia/departamento.

## Licencia

Definir antes de publicar si el repositorio será público. Una opción habitual para proyectos demostrativos es MIT, pero conviene confirmarlo según el uso previsto.


## Nota sobre variables POB_TOT_P / VIV_TOT_P

La app consulta primero la tabla larga `census-data.parquet`. Para variables de totales muy usadas, como `POB_TOT_P` y `VIV_TOT_P`, la versión actual también puede consultar directamente `radios.parquet`, donde esas columnas están disponibles como atributos de los radios censales. Esto evita consultas vacías cuando esos totales no aparecen como filas en la tabla larga o cuando el filtro elegido no corresponde a una variable categórica.

Si una consulta devuelve vacío, la app muestra un diagnóstico con el conteo exacto en `census-data.parquet` y variables parecidas disponibles.
