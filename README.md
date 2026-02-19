# SafeRoute_AI
Plataforma de analisis de riesgo y planeacion de rutas para ciclistas en la CDMX. El proyecto integra datos de red ciclista, accidentes, infraestructura, clima y afluencia para generar visualizaciones, rankings de calles peligrosas y una planeacion de ruta con base en riesgo.

## Planteamiento de la solucion
Objetivo: ayudar a ciclistas a elegir rutas mas seguras con base en informacion historica y variables de contexto.

En `main.py` se implementa un dashboard en Streamlit que:
1. Carga y normaliza datos de accidentes e infraestructura.
2. Construye un grafo dirigido de la red ciclista (OSM).
3. Estima afluencia por alcaldia y hora.
4. Muestra mapas de calor y metricas de riesgo.
5. Permite planeacion de ruta basada en el grafo y el riesgo.

## Estructura del proyecto
- `main.py`: aplicacion Streamlit (dashboard).
- `main_pre.py`: version previa del dashboard.
- `Bases_datos_proy5/`: datos de trabajo y resultados.
- `IPYNB_proyecto5/`: notebooks y scripts de preparacion de datos.
- `pyproject.toml` y `uv.lock`: dependencias.

## Flujo de actualizacion de datos (resumen)
Los notebooks/scripts en `IPYNB_proyecto5/` generan los archivos usados por `main.py`:
1. `Base_principal.ipynb` o `Base_principal.py`: red ciclista (OSM) -> `Bases_datos_proy5/Resultados/`.
2. `Afluencia.ipynb` o `Afluencia.py`: modelo de afluencia -> `Bases_datos_proy5/Resultados/DB_afluencia.parquet`.
3. `Incidencia_1er_modelo.ipynb` o `Incidencia_1er_modelo.py`: riesgo de infraestructura y accidentes ->
   `Bases_datos_proy5/Resultados/DB_Infraestructura.parquet` y `Bases_datos_proy5/Resultados/DB_Accidentes.parquet`.
4. `Clima.ipynb` o `Clima.py`: clima -> `Bases_datos_proy5/Resultados/DB_Clima.parquet`.
5. `main.py`: consume los parquet y levanta la aplicacion.

## Requisitos
- Python 3.12 (ver `.python-version`).
- Dependencias Python del proyecto.

## Instalacion
El proyecto usa `pyproject.toml`. Puedes instalar con `uv` o `pip`.

### Opcion A: uv (recomendado)
```bash
uv venv
uv pip install -r pyproject.toml
```

### Opcion B: venv + pip
```bash
python -m venv .venv
```

Windows (PowerShell):
```powershell
.venv\Scripts\Activate.ps1
pip install -r pyproject.toml
```

macOS/Linux (bash/zsh):
```bash
source .venv/bin/activate
pip install -r pyproject.toml
```

Nota: si tu instalador no acepta `-r pyproject.toml`, usa `pip install -e .` desde la raiz del proyecto.

## Ejecucion
Una vez instaladas las dependencias y con los parquet disponibles en `Bases_datos_proy5/Resultados/`:

```bash
streamlit run main.py
```

## Datos requeridos
`main.py` espera los siguientes archivos:
- `Bases_datos_proy5/ABT_raiz_red_ciclista_completa_cdmx.parquet`
- `Bases_datos_proy5/Resultados/DB_Accidentes.parquet`
- `Bases_datos_proy5/Resultados/DB_Infraestructura.parquet`
- `Bases_datos_proy5/Resultados/DB_Clima.parquet`
- `Bases_datos_proy5/Resultados/DB_afluencia.parquet`

Si faltan, ejecuta los notebooks/scripts descritos en la seccion "Flujo de actualizacion de datos".

## Soporte por sistema operativo
- Windows 10/11: probado con PowerShell. Si ves errores por politicas de ejecucion, usa una consola que permita activar el entorno virtual.
- macOS: usa `python3` y `source .venv/bin/activate`.
- Linux: mismo flujo que macOS.

## Problemas comunes
- Archivos faltantes: verifica la carpeta `Bases_datos_proy5/Resultados/`.
- Errores de dependencias: reinstala el entorno virtual y ejecuta `pip install -e .`.

## Licencia
Ver `LICENSE`.
