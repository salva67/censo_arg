# Dashboard Censo Argentino · Streamlit + DuckDB

Dashboard para consultar datos del **Censo Argentino** publicados en Source Cooperative, sin usar QGIS. La app lee archivos Parquet/GeoParquet remotos con DuckDB y permite explorar variables censales, filtrar por geografía, descargar resultados, generar gráficos informativos y visualizar radios censales en mapas con capas.

## Qué incluye

- Consulta directa a Parquet remoto con DuckDB.
- Selector de año censal: 1991, 2001, 2010 y 2022.
- Buscador de variables censales desde `census-data.parquet`, enriquecido con `metadata.parquet`.
- Filtros por provincia, departamento y categoría.
- Agregación por provincia, departamento, categoría o radio censal.
- Descarga de resultados a CSV enriquecido.
- KPIs automáticos: total, cantidad de áreas, promedio, mediana y área con mayor valor.
- Gráficos informativos:
  - ranking top N,
  - participación sobre el total,
  - Pareto acumulado,
  - distribución de valores.
- Mapa de radios censales con capas:
  - polígonos coropléticos,
  - burbujas por centroides,
  - etiquetas para top radios,
  - vista 3D,
  - métrica de color por valor total, valor/km² o log(valor + 1).
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
- Buscar variable: `pob`
- Variable: `POB_TOT_P`
- Provincia: `Buenos Aires`
- Departamento: `Luján`
- Agrupar por: `Departamento`

Para el mapa conviene filtrar al menos una provincia y, si el resultado es pesado, un departamento.

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
- Para evitar el error de PyArrow al leer HTTPS desde GeoPandas en Streamlit Cloud, el GeoParquet se descarga primero a `/tmp/censo_argentino_cache` y después se lee localmente.
- Los nombres de provincias y departamentos se intentan resolver con GeoRef. Si no responde, la app mantiene códigos como fallback.
- Para `POB_TOT_P` y `VIV_TOT_P`, la app puede consultar directamente `radios.parquet`, porque esas variables están publicadas como columnas geográficas.

## Licencia

Definir antes de publicar si el repositorio será público. Una opción habitual para proyectos demostrativos es MIT, pero conviene confirmarlo según el uso previsto.


## Cambios v6

- Corrige el error `GeoDataFrame cannot contain duplicated column names` al generar el GeoJSON del mapa.
- Deduplica columnas luego de merges y antes de construir capas PyDeck.
- Evita columnas repetidas cuando la métrica elegida es el mismo campo que el valor (`conteo`).
- Mantiene el mapa funcionando con coroplético, burbujas, etiquetas y vista 3D.
