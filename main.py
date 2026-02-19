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


import streamlit as st
import folium
import pandas as pd
import numpy as np
import datetime
from streamlit_folium import st_folium
from folium.plugins import HeatMap
from sklearn.preprocessing import QuantileTransformer
from itertools import islice

########### CONFIGURACIÓN Y ESTILO ###########
st.set_page_config(page_title="SafeRoute AI - Dashboard", page_icon=":bike:", layout="wide")

st.title("🚴 SafeRoute AI")
st.write("Plataforma de análisis de riesgo y navegación inteligente para ciclistas.")


import os

# --- BLOQUE DE EMERGENCIA PARA VER RUTAS ---
st.sidebar.write("### 📂 Servidor de Archivos")
raiz = os.getcwd()
st.sidebar.write(f"Raíz: `{raiz}`")

# Intentamos listar lo que hay en la carpeta de datos
try:
    # Cambia esto al nombre exacto que veas en GitHub (ej. "Bases_datos_proy5")
    folder = "bases_datos_proy5" 
    st.sidebar.write(f"Contenido de {folder}:", os.listdir(folder))
    
    res_folder = os.path.join(folder, "Resultados")
    if os.path.exists(res_folder):
        st.sidebar.write("✅ Carpeta Resultados encontrada.")
    else:
        st.sidebar.error("❌ No existe 'Resultados'. ¿Quizá es 'resultados'?")
except Exception as e:
    st.sidebar.error(f"Error explorando: {e}")
# -------------------------------------------



##########################################################################
# --- CARGA Y PREPROCESAMIENTO ---
##########################################################################
from pathlib import Path
import os

# Get the absolute path of the directory containing the main.py script
script_dir = Path(__file__).parent
# Construct the full absolute path to the data file
file_path_red = script_dir / "bases_datos_proy5" / "ABT_raiz_red_ciclista_completa_cdmx.parquet"
file_path_acc = script_dir / "bases_datos_proy5" / "Resultados" / "DB_Accidentes.parquet"
file_path_inf = script_dir / "bases_datos_proy5" / "Resultados" / "DB_Infraestructura.parquet"
file_path_cli = script_dir / "bases_datos_proy5" / "Resultados" / "DB_Clima.parquet"
file_path_aflu = script_dir / "bases_datos_proy5" / "Resultados" / "DB_afluencia.parquet"


@st.cache_data
def load_and_preprocess():
    # Carga de archivos
    # df_red = pd.read_parquet("./Bases_datos_proy5/ABT_raiz_red_ciclista_completa_cdmx.parquet")
    # df_acc = pd.read_parquet("./bases_datos_proy5/Resultados/DB_Accidentes.parquet")
    # df_inf = pd.read_parquet("./Bases_datos_proy5/Resultados/DB_Infraestructura.parquet")
    # df_cli = pd.read_parquet("./Bases_datos_proy5/Resultados/DB_Clima.parquet")
    # df_aflu = pd.read_parquet("./Bases_datos_proy5/Resultados/DB_afluencia.parquet")
    df_red = pd.read_parquet(file_path_red)
    df_acc = pd.read_parquet(file_path_acc)
    df_inf = pd.read_parquet(file_path_inf)
    df_cli = pd.read_parquet(file_path_cli)
    df_aflu = pd.read_parquet(file_path_aflu)
    
    # Asegurar datetime
    df_acc['timestamp_fijo'] = pd.to_datetime(df_acc['timestamp_fijo'])
    df_inf['timestamp_fijo'] = pd.to_datetime(df_inf['timestamp_fijo'])
    df_cli['time'] = pd.to_datetime(df_cli['time'])
    df_aflu['fecha'] = pd.to_datetime(df_aflu['fecha'])

    # --- NORMALIZACIÓN DE CUANTILES (0.05 a 0.95) ---
    qt = QuantileTransformer(output_distribution='uniform', n_quantiles=1000)
    
    for df in [df_acc, df_inf]:
        vals = df[['Score_riesgo_cluster']].values
        norm_vals = qt.fit_transform(vals)
        df['Score_Final'] = 0.05 + (norm_vals * 0.90)
        
    return df_acc, df_inf, df_cli, df_aflu, df_red

df_acc, df_inf, df_cli, df_aflu, df_red = load_and_preprocess()

# --- FUNCIONES DE APOYO ---
def aplicar_distribucion_afluencia(df, hora):
    pico_1 = np.exp(-((hora - 8)**2) / (2 * 2**2)) 
    pico_2 = np.exp(-((hora - 18)**2) / (2 * 3**2))
    factor_hora = (pico_1 + pico_2) / 1.5
    df_res = df.groupby('alcaldia')['afluencia'].mean().reset_index()
    df_res['Afluencia_Estimada'] = (df_res['afluencia'] * factor_hora).astype(int)
    df_res['Afluencia_Norm'] = df_res['Afluencia_Estimada'] / df_res['Afluencia_Estimada'].max()
    return df_res

# --- 1. CONSTRUCCIÓN DEL GRAFO (Optimizado) ---
@st.cache_resource # Usamos cache_resource para objetos pesados como el Grafo
def construir_grafo(df):
    G = nx.DiGraph()
    for _, row in df.iterrows():
        geom = row['geometry']
        if isinstance(geom, bytes):
            geom = wkb.loads(geom)
        
        u = (row['lat_start'], row['lon_start'])
        v = (row['lat_end'], row['lon_end'])
        
        # Guardamos longitud y segment_id para cálculos posteriores
        G.add_edge(u, v, weight=row['length'], geometry=geom, id=row['segment_id'], name=row['street_name'])
        if not row['oneway']:
            G.add_edge(v, u, weight=row['length'], geometry=geom, id=row['segment_id'], name=row['street_name'])
    return G

G = construir_grafo(df_red)
# --- 2. PREPARACIÓN DE BUSCADORES ---
# Creamos una lista de opciones: "Nombre de Calle (Lat, Lon)"
opciones_nodos = df_red[['street_name', 'lat_start', 'lon_start']].drop_duplicates()
opciones_nodos['label'] = opciones_nodos['street_name'].fillna("S/N") + " (" + opciones_nodos['lat_start'].astype(str) + ", " + opciones_nodos['lon_start'].astype(str) + ")"
dict_nodos = opciones_nodos.set_index('label')[['lat_start', 'lon_start']].to_dict('index')

###########################################
# --- CREACIÓN DE PESTAÑAS ---
###########################################
tab1, tab2 = st.tabs(["📊 Análisis de Riesgo", "🗺️ Planeación de Ruta"])

# ==========================================
# PESTAÑA 1: ANÁLISIS GENERAL
# ==========================================
with tab1:
    st.header("Estadísticas y Puntos Críticos")
    
    # Filtros dentro de la pestaña para limpieza visual
    c_f1, c_f2 = st.columns(2)
    with c_f1:
        fecha_sel = st.date_input("Fecha de Análisis", 
                                 value=datetime.date(2024, 1, 1),
                                 min_value=datetime.date(2024, 1, 1), 
                                 max_value=datetime.date(2026, 2, 21),
                                 key="fecha_t1")
    with c_f2:
        hora_sel = st.slider("Hora del día", 0, 23, 12, key="hora_t1")

    # Lógica de Filtrado
    mes_objetivo = fecha_sel.month
    df_acc_f = df_acc[(df_acc['timestamp_fijo'].dt.month == mes_objetivo) & (df_acc['timestamp_fijo'].dt.hour == hora_sel)]
    df_inf_f = df_inf[(df_inf['timestamp_fijo'].dt.month == mes_objetivo) & (df_inf['timestamp_fijo'].dt.hour == hora_sel)]
    df_cli_f = df_cli[(df_cli['time'].dt.date == fecha_sel) & (df_cli['time'].dt.hour == hora_sel)]
    df_aflu_resumen = aplicar_distribucion_afluencia(df_aflu, hora_sel)

    # Métricas Globales
    avg_acc = df_acc_f['Score_Final'].mean() if not df_acc_f.empty else 0.05
    avg_inf = df_inf_f['Score_Final'].mean() if not df_inf_f.empty else 0.05
    avg_aflu = df_aflu_resumen['Afluencia_Estimada'].mean() if not df_aflu_resumen.empty else 0
    avg_lluvia = df_cli_f['precipitation_probability'].mean() if not df_cli_f.empty else 0

    #=============================================
    # --- CÁLCULO DE TOP 10 CALLES PELIGROSAS ---
    #=============================================

    # 1. Agrupamos el riesgo filtrado por segmento
    resumen_acc = df_acc_f.groupby('segment_id')['Score_Final'].mean().reset_index()
    resumen_inf = df_inf_f.groupby('segment_id')['Score_Final'].mean().reset_index()

    # 2. Unimos ambos riesgos (Score Maestro)
    df_riesgo_vial = pd.merge(resumen_acc, resumen_inf, on='segment_id', how='outer', suffixes=('_acc', '_inf')).fillna(0.05)

    # Ponderación: 60% accidentes, 40% infraestructura (puedes ajustar esto)
    df_riesgo_vial['Score_Maestro'] = (df_riesgo_vial['Score_Final_acc'] * 0.6) + (df_riesgo_vial['Score_Final_inf'] * 0.4)

    # 3. Cruzamos con los nombres de las calles
    top_calles = pd.merge(df_riesgo_vial, df_red[['segment_id', 'street_name', 'alcaldia_name']], on='segment_id')

    # Limpieza: eliminamos segmentos sin nombre de calle y ordenamos
    top_calles = top_calles[top_calles['street_name'].notna()]
    top_10_ranking = top_calles.sort_values(by='Score_Maestro', ascending=False).head(10)



    st.write("---")
    m1, m2, m3, m4 = st.columns(4)
    m1.metric("🚨 Riesgo Accidentes", f"{avg_acc:.2f}", delta=f"{((avg_acc-0.5)/0.5)*100:.1f}% vs Prom", delta_color="inverse")
    m2.metric("🛠️ Riesgo Infra.", f"{avg_inf:.2f}", delta=f"{((avg_inf-0.5)/0.5)*100:.1f}% vs Prom", delta_color="inverse")
    m3.metric("👥 Afluencia Prom.", f"{int(avg_aflu):,}")
    m4.metric("🌧️ Prob. Lluvia", f"{avg_lluvia:.1f}%")
    st.write("---")

    # 2. Mapas de Score (Fila 1)
    st.markdown("### Distribución de Riesgo (Scores)")
    col_m1, col_m2 = st.columns(2)
    
    with col_m1:
        st.caption("🚨 Riesgo por Siniestralidad Vial")
        m1 = folium.Map(location=[19.4326, -99.1332], zoom_start=11, tiles='cartodbdark_matter')
        if not df_acc_f.empty:
            # Heatmap pesado por el Score_Final
            HeatMap(df_acc_f[['latitud', 'longitud', 'Score_Final']].values.tolist(), radius=8, blur=10).add_to(m1)
        st_folium(m1, width=None, height=400, key="map_acc")

    with col_m2:
        st.caption("🛠️ Riesgo por Infraestructura Urbana")
        m2 = folium.Map(location=[19.4326, -99.1332], zoom_start=11, tiles='cartodbdark_matter')
        if not df_inf_f.empty:
            HeatMap(df_inf_f[['latitud', 'longitud', 'Score_Final']].values.tolist(), radius=8, blur=10, gradient={0.2:'blue', 0.5:'yellow', 1:'orange'}).add_to(m2)
        st_folium(m2, width=None, height=400, key="map_inf")

    # 3. Mapa de Calor Incidentes Ciclistas (Fila 2 - Ancho Completo)
    st.write("---")
    st.markdown("### Mapa de Concentración: Incidentes Ciclistas")
    m3 = folium.Map(location=[19.4326, -99.1332], zoom_start=11, tiles='cartodbdark_matter')
    if not df_acc_f.empty:
        # Aquí solo mapeamos la densidad de puntos donde hubo incidentes ciclistas
        HeatMap(df_acc_f[df_acc_f['inc_Ciclista']>0][['latitud', 'longitud']].values.tolist(), radius=15, blur=20, gradient={0.4:'blue', 0.6:'cyan', 0.7:'lime', 1:'red'}).add_to(m3)
    st_folium(m3, width=1500, height=500, key="map_heat_full")

    # 4. Gráficos Comparativos (Fila 3)
    st.write("---")
    col_g1, col_g2 = st.columns(2)
    with col_g1:
        st.markdown("#### Probabilidad de Lluvia (%)")
        st.bar_chart(df_cli_f.set_index('alcaldia')['precipitation_probability'], height=300)
    with col_g2:
        st.markdown("#### Afluencia Estimada (Pasajeros)")
        st.bar_chart(df_aflu_resumen.set_index('alcaldia')['Afluencia_Estimada'], height=300)

    # 5. Ranking de Peligrosidad (Fila Final)
    st.write("---")
    st.markdown("### 🏆 Top 10: Segmentos con Mayor Riesgo Combinado")
    st.info("Este ranking pondera la siniestralidad histórica (60%) y las fallas de infraestructura reportadas (40%).")
    
    # Formateo para mostrar a los usuarios
    ranking_display = top_10_ranking[['street_name', 'alcaldia_name', 'Score_Maestro', 'Score_Final_acc', 'Score_Final_inf']].copy()
    ranking_display.columns = ['Calle', 'Alcaldía', 'Riesgo Total', 'Riesgo Accidente', 'Riesgo Infra.']
    
    st.dataframe(ranking_display.style.background_gradient(cmap='YlOrRd', subset=['Riesgo Total']), use_container_width=True)

#############################################################################################
# 3er intento real de la segunda pestaña
#############################################################################################

# --- Función Auxiliar para Categorizar Seguridad (10 Niveles) ---
def clasificar_seguridad(score):
    if score <= 0.10: return "🌟 Máxima"
    if score <= 0.20: return "✅ Muy Alta"
    if score <= 0.30: return "🟢 Alta"
    if score <= 0.40: return "🟡 Moderada-Alta"
    if score <= 0.50: return "🟠 Moderada"
    if score <= 0.60: return "🟠 Moderada-Baja"
    if score <= 0.70: return "🔴 Baja"
    if score <= 0.80: return "🚫 Muy Baja"
    if score <= 0.90: return "⚠️ Crítica"
    return "💀 Extrema"

# --- 1. ESTADO DE SESIÓN (Fuera de las pestañas) ---
if 'resumen_rutas' not in st.session_state:
    st.session_state.resumen_rutas = None
if 'nodos_rutas' not in st.session_state:
    st.session_state.nodos_rutas = None

with tab2:
    st.header("📍 Planeación de Rutas Seguras")
    
    # 1. Filtros Temporales Específicos para la Ruta
    c1, c2 = st.columns(2)
    with c1:
        f_ruta = st.date_input("Día del viaje", value=datetime.date(2024, 1, 1), key="f_t2")
    with c2:
        h_ruta = st.slider("Hora de salida", 0, 23, 12, key="h_t2")

    # 2. Origen y Destino
    col_a, col_b = st.columns(2)
    with col_a:
        origen_label = st.selectbox("Punto A", options=list(dict_nodos.keys()), key="o_t2")
        p_a = (dict_nodos[origen_label]['lat_start'], dict_nodos[origen_label]['lon_start'])
    with col_b:
        destino_label = st.selectbox("Punto B", options=list(dict_nodos.keys()), key="d_t2")
        # p_b = (dict_nodos[destino_label]['lat_end'], dict_nodos[destino_label]['lon_end'])
        p_b = (dict_nodos[destino_label]['lat_start'], dict_nodos[destino_label]['lon_start'])

    if st.button("🚀 Calcular 5 Rutas con Riesgo Dinámico"):
        with st.spinner("Analizando micro-riesgos y velocidad de vías..."):
            try:
                # 1. Factores Dinámicos
                df_cli_ruta = df_cli[(df_cli['time'].dt.date == f_ruta) & (df_cli['time'].dt.hour == h_ruta)]
                df_aflu_ruta = aplicar_distribucion_afluencia(df_aflu, h_ruta)
                mapa_lluvia = df_cli_ruta.set_index('alcaldia')['precipitation_probability'].to_dict()
                mapa_aflu = df_aflu_ruta.set_index('alcaldia')['Afluencia_Norm'].to_dict()

                # 2. Actualización de Pesos con Velocidad
                for u, v, d in G.edges(data=True):
                    alc = d.get('alcaldia_name', 'CUAUHTEMOC')
                    v_max = d.get('maxspeed', 40)
                    
                    # Penalización por velocidad (normalizada a 80km/h)
                    s_vel = (v_max / 80)**2
                    s_lluvia = (mapa_lluvia.get(alc, 0) / 100)
                    s_aflu = mapa_aflu.get(alc, 0)
                    
                    # Fórmula Maestra Ajustada
                    score_g = (0.3 * avg_acc) + (0.2 * avg_inf) + (0.25 * s_vel) + (0.15 * s_lluvia) + (0.1 * s_aflu)
                    
                    d['costo_seguro'] = d['weight'] * (1 + score_g)
                    d['score_seg_segmento'] = score_g
                    d['v_segmento'] = v_max

                # 3. Calcular las 5 rutas
                k_rutas = list(islice(nx.shortest_simple_paths(G, p_a, p_b, weight='costo_seguro'), 5))
                st.session_state.nodos_rutas = k_rutas
                
                resumen_final = []
                for idx, ruta in enumerate(k_rutas):
                    d_total, r_total, v_total, count = 0, 0, 0, 0
                    for i in range(len(ruta)-1):
                        edge_data = G.get_edge_data(ruta[i], ruta[i+1])
                        d_total += edge_data['weight']
                        r_total += edge_data['score_seg_segmento']
                        v_total += edge_data['v_segmento']
                        count += 1
                    
                    avg_risk = (r_total / count) if count > 0 else 0
                    avg_speed = (v_total / count) if count > 0 else 0
                    
                    resumen_final.append({
                        "Ruta": f"Ruta {idx+1}",
                        "Distancia (m)": round(d_total, 1),
                        "Vel. Promedio": f"{round(avg_speed, 1)} km/h",
                        "Riesgo General": round(avg_risk, 4),
                        "Seguridad": clasificar_seguridad(avg_risk)
                    })
                
                st.session_state.resumen_rutas = pd.DataFrame(resumen_final)

            except nx.NetworkXNoPath:
                st.error("❌ No existe una ruta ciclista conectada.")

    # --- Renderizado ---
    if st.session_state.nodos_rutas:
        st.write("---")
        c_map, c_tab = st.columns([2, 1])
        with c_map:
            
            m_final = folium.Map(location=[p_a[0], p_a[1]], zoom_start=14, tiles='cartodbpositron')
            colores = ['#007bff', '#28a745', '#ffc107', '#fd7e14', '#dc3545']
            for idx, ruta in enumerate(st.session_state.nodos_rutas):
                for i in range(len(ruta)-1):
                    data = G.get_edge_data(ruta[i], ruta[i+1])
                    folium.PolyLine([(lat, lon) for lon, lat in data['geometry'].coords], 
                                    color=colores[idx], weight=5, opacity=0.7).add_to(m_final)
            folium.Marker(p_a, icon=folium.Icon(color='green', icon='play')).add_to(m_final)
            folium.Marker(p_b, icon=folium.Icon(color='red', icon='stop')).add_to(m_final)
            st_folium(m_final, width='stretch', height=550, key="mapa_final_t2", returned_objects=[])

        with c_tab:
            st.write("### 📋 Tabla de Seguridad")
            st.dataframe(
                st.session_state.resumen_rutas, 
                width='stretch', 
                hide_index=True,
                column_config={
                    "Riesgo General": st.column_config.ProgressColumn(min_value=0, max_value=1, format="%.4f"),
                    "Seguridad": st.column_config.TextColumn("Clasificación")
                }
            )