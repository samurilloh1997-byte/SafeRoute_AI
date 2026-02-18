import streamlit as st
import folium
import pandas as pd
import networkx as nx
from shapely import wkb
from streamlit_folium import st_folium
import networkx as nx
from shapely import wkb
import pandas as pd
import numpy as np
import datetime
from folium.plugins import HeatMap
from sklearn.preprocessing import QuantileTransformer

########### EJEMPLO SIMPLE DE STREAMLIT ###########
# def main():
#     print("Hello from saferoute-ai!")
# if __name__ == "__main__":
#     main()

#################################################################################################
# Forma de actualización de la información de la base de datos y del proyecto en general
#################################################################################################

# 1.	Se ejecuta y actualiza la información de IPYNB_proyecto5/Base_principal.ipynb 
# o Base_principal.py, esto actualizara la información de la base principal de OSMx, 
# que es la fuente principal que nos da las rutas legales en bicicleta dentro de la CDMX. 
# La base de datos queda almacenada en Bases_datos_proy5/Resultados.
# 2.	Posteriormente se ejecuta el código de IPYNB_proyecto5/Afluencia.ipynb o Afluencia.py, 
# esta información actualiza el modelo y nos da la tabla de resultados de los pronósticos de 
# la afluencia dentro de la CDMX y los resultados se guardan en la carpeta de 
# Bases_datos_proy5/Resultados como DB_afluencia.parquet.
# 3.	Posteriormente se actualiza la parte de IPYNB_proyecto5/Incidencia_1er_modelo.ipynb 
# o Incidencia_1er_modelo.py cuya información se almacena en Bases_datos_proy5/Resultados 
# como DB_Infraestructura.parquet y DB_Accidentes.parquet.
# 4.	Se compila el código de IPYNB_proyecto5/Clima.ipynb o Clima.py, cuya información de 
# resultado se almacena en Bases_datos_proy5/Resultados como DB_Clima.parquet.
# 5.	Finalmente se compila la información de main.py, que nos devuelve la información de la 
# aplicación de Streamlit lista para desplegar.
#################################################################################################


########### CREACIÓN DE TABLERO CON STREAMLIT ###########
st.set_page_config(page_title="SafeRoute AI - Dashboard", page_icon=":bike:", layout="wide")

st.title("SafeRoute AI")
st.write("This is a demo of the SafeRoute AI model.")

# --- CARGA Y PREPROCESAMIENTO ---
@st.cache_data
def load_and_preprocess():
    # Carga de archivos (Usando 'r' para evitar SyntaxWarning)
    df_acc = pd.read_parquet(r"bases_datos_proy5\Resultados\DB_Accidentes.parquet")
    df_inf = pd.read_parquet(r"Bases_datos_proy5\Resultados\DB_Infraestructura.parquet")
    df_cli = pd.read_parquet(r"Bases_datos_proy5\Resultados\DB_Clima.parquet")
    df_aflu = pd.read_parquet(r"Bases_datos_proy5\Resultados\DB_afluencia.parquet")
    
    # Asegurar datetime
    df_acc['timestamp_fijo'] = pd.to_datetime(df_acc['timestamp_fijo'])
    df_inf['timestamp_fijo'] = pd.to_datetime(df_inf['timestamp_fijo'])
    df_cli['time'] = pd.to_datetime(df_cli['time'])
    df_aflu['fecha'] = pd.to_datetime(df_aflu['fecha'])

    # --- NORMALIZACIÓN DE CUANTILES (0.05 a 0.95) ---
    qt = QuantileTransformer(output_distribution='uniform', n_quantiles=1000)
    
    for df in [df_acc, df_inf]:
        # Ajustamos y transformamos para expandir varianza
        vals = df[['Score_riesgo_cluster']].values
        norm_vals = qt.fit_transform(vals)
        # Mapeo a [0.05, 0.95]
        df['Score_Final'] = 0.05 + (norm_vals * 0.90)
        
    return df_acc, df_inf, df_cli, df_aflu

df_acc, df_inf, df_cli, df_aflu = load_and_preprocess()

# print(df_ciclistas.head())

# --- SIDEBAR: FILTROS GLOBALES ---
st.sidebar.header("Filtros de Control")

# 1. Filtro de Fecha (Rango)
# Tomamos el mínimo y máximo de la tabla de accidentes como referencia
# min_fecha = df_accidentes['timestamp_fijo'].min().date()
# max_fecha = df_accidentes['timestamp_fijo'].max().date()
# min_fecha = datetime.date(2024, 1, 1)
# max_fecha = datetime.date(2026, 2, 21)

# fecha_rango = st.sidebar.date_input(
#     "Selecciona Rango de Fechas",
#     value=(min_fecha, max_fecha),
#     min_value=min_fecha,
#     max_value=max_fecha
# )

# # 2. Filtro de Hora (Slider de Rango)
# hora_rango = st.sidebar.slider(
#     "Selecciona Rango de Horas",
#     0, 23, (0, 23)
# )

# --- SIDEBAR (FILTROS) ---
with st.sidebar:
    st.header("SafeRoute AI: Filtros")
    
    # Filtro Fecha
    f_inicio = datetime.date(2024, 1, 1)
    f_final = datetime.date(2026, 2, 21)
    fecha_sel = st.date_input("Fecha de Análisis", value=f_inicio, min_value=f_inicio, max_value=f_final)
    
    # Filtro Hora
    hora_sel = st.slider("Hora del día", 0, 23, 12)
    
    # Filtro Lugar (Simulado para Demo)
    lugar_a = st.text_input("Punto de Origen (A)", "Pantitlán")
    lugar_b = st.text_input("Punto de Destino (B)", "Polanco")

# # 3. Filtro de Alcaldía (Multiselect)
# lista_alcaldias = sorted(df_ciclistas['alcaldia_name'].unique())
# alcaldias_sel = st.sidebar.multiselect(
#     "Selecciona Alcaldía(s)",
#     options=lista_alcaldias,
#     default=lista_alcaldias[:3] # Seleccionamos las primeras 3 por defecto
# )
# --- LÓGICA DE FILTRADO TEMPORAL ---
# Para Accidentes e Infraestructura (Solo hasta 2025): Filtramos por MES y HORA del año seleccionado (o el anterior si es 2026)
mes_objetivo = fecha_sel.month
df_acc_f = df_acc[(df_acc['timestamp_fijo'].dt.month == mes_objetivo) & (df_acc['timestamp_fijo'].dt.hour == hora_sel)]
df_inf_f = df_inf[(df_inf['timestamp_fijo'].dt.month == mes_objetivo) & (df_inf['timestamp_fijo'].dt.hour == hora_sel)]

# Clima: Filtro directo por fecha exacta
df_cli_f = df_cli[(df_cli['time'].dt.date == fecha_sel) & (df_cli['time'].dt.hour == hora_sel)]

# --- LÓGICA DE FILTRADO GLOBAL ---

# # Función auxiliar para aplicar filtros de fecha y hora
# def filtrar_tabla(df, col_fecha, col_alcaldia=None):
#     # Filtrar por Rango de Fecha (Streamlit devuelve una tupla con 2 fechas)
#     if len(fecha_rango) == 2:
#         mask = (df[col_fecha].dt.date >= fecha_rango[0]) & (df[col_fecha].dt.date <= fecha_rango[1])
#         df = df[mask]
    
#     # Filtrar por Rango de Hora
#     df = df[(df[col_fecha].dt.hour >= hora_rango[0]) & (df[col_fecha].dt.hour <= hora_rango[1])]
    
#     # Filtrar por Alcaldía si la tabla tiene esa columna
#     if col_alcaldia and alcaldias_sel:
#         df = df[df[col_alcaldia].isin(alcaldias_sel)]
        
#     return df

# --- MODELO DE DISTRIBUCIÓN DE AFLUENCIA (BIMODAL) ---
def aplicar_distribucion_afluencia(df, hora):
    # Definimos picos: 8 AM (mu=8) y 6 PM (mu=18)
    pico_1 = np.exp(-((hora - 8)**2) / (2 * 2**2)) 
    pico_2 = np.exp(-((hora - 18)**2) / (2 * 3**2))
    factor_hora = (pico_1 + pico_2) / 1.5 # Factor de ajuste bimodal
    
    df_res = df.groupby('alcaldia')['afluencia'].mean().reset_index()
    df_res['Afluencia_Estimada'] = (df_res['afluencia'] * factor_hora).astype(int)
    # Normalización 0-1 para visualización
    df_res['Afluencia_Norm'] = df_res['Afluencia_Estimada'] / df_res['Afluencia_Estimada'].max()
    return df_res

df_aflu_resumen = aplicar_distribucion_afluencia(df_aflu, hora_sel)

# # Aplicar filtros a cada tabla (cada una con su nombre de columna respectivo)
# df_acc_f = filtrar_tabla(df_accidentes, 'timestamp_fijo') # Alcaldía depende de tu join
# df_inf_f = filtrar_tabla(df_infra, 'timestamp_fijo')
# df_cli_f = filtrar_tabla(df_clima, 'time', 'alcaldia')
# df_aflu_f = filtrar_tabla(df_afluencia, 'fecha', 'alcaldia')

# # Para la red ciclista (que suele ser estática), filtramos solo por Alcaldía
# df_ciclistas_f = df_ciclistas[df_ciclistas['alcaldia_name'].isin(alcaldias_sel)]

# # --- VISTA DE RESULTADOS (Demo) ---
# col1, col2, col3 = st.columns(3)
# col1.metric("Accidentes en Rango", len(df_acc_f))
# col2.metric("Reportes Infraestructura", len(df_inf_f))
# col3.metric("Segmentos de Red", len(df_ciclistas_f))

# st.write("### Vista previa de Accidentes Filtrados")
# st.dataframe(df_acc_f.head())


###########################################
# --- LAYOUT PRINCIPAL ---
###########################################
col_mapa, col_stats = st.columns([2, 1])

with col_mapa:
    st.subheader("Mapa de Riesgo e Incidentes")
    m = folium.Map(location=[19.4326, -99.1332], zoom_start=11, tiles='cartodbdark_matter')
    
    # Heatmap de incidentes ciclistas
    if not df_acc_f.empty:
        heat_data = df_acc_f[['latitud', 'longitud', 'inc_Ciclista']].values.tolist()
        HeatMap(heat_data, radius=10, blur=15, gradient={0.4: 'blue', 0.65: 'lime', 1: 'red'}).add_to(m)
    
    st_folium(m, width=800, height=500)

with col_stats:
    st.subheader("Probabilidad de Lluvia")
    if not df_cli_f.empty:
        st.bar_chart(df_cli_f.set_index('alcaldia')['precipitation_probability'])
    else:
        st.info("Sin datos climáticos para esta fecha.")

    st.subheader("Afluencia Proyectada por Alcaldía")
    st.table(df_aflu_resumen[['alcaldia', 'Afluencia_Estimada', 'Afluencia_Norm']].sort_values('Afluencia_Estimada', ascending=False))

# --- INDICADORES DE RIESGO ---
st.write("---")
c1, c2 = st.columns(2)
c1.metric("Riesgo Promedio Accidentes", f"{df_acc_f['Score_Final'].mean():.2f}")
c2.metric("Riesgo Promedio Infraestructura", f"{df_inf_f['Score_Final'].mean():.2f}")

