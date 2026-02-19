from pathlib import Path
import pandas as pd
import networkx as nx
from shapely import wkb
import pickle
import gzip
import os

script_dir = Path(__file__).parent
file_path_red = script_dir / "Bases_datos_proy5" / "ABT_raiz_red_ciclista_completa_cdmx.parquet"
ruta_salida_grafo = script_dir / "Bases_datos_proy5" / "grafo_cdmx.pkl.gz"

# 1. Configuración de rutas locales
# Asegúrate de que estas rutas coincidan con tu estructura de carpetas en tu PC
# ruta_red_parquet = "Bases_datos_proy5/ABT_raiz_red_ciclista_completa_cdmx.parquet"
# ruta_salida_grafo = "Bases_datos_proy5/grafo_cdmx.pkl.gz"

def generar_y_guardar_grafo(ruta_red_parquet):
    print("🚀 Iniciando construcción del grafo...")
    
    # Cargar el dataframe de la red
    df = pd.read_parquet(ruta_red_parquet)
    
    # Crear Grafo Dirigido
    G = nx.DiGraph()
    
    for _, row in df.iterrows():
        # Procesar geometría (WKB a Shapely)
        geom = row['geometry']
        if isinstance(geom, bytes):
            geom = wkb.loads(geom)
        
        # Definir nodos por coordenadas
        u = (row['lat_start'], row['lon_start'])
        v = (row['lat_end'], row['lon_end'])
        
        # Extraer atributos críticos
        # Nota: Guardamos maxspeed aquí para que ya esté disponible en la nube
        atributos = {
            'weight': row['length'],
            'geometry': geom,
            'id': row['segment_id'],
            'name': row.get('street_name', 'S/N'),
            'maxspeed': row.get('maxspeed', 40),
            'alcaldia_name': row.get('alcaldia_name', 'CUAUHTEMOC')
        }
        
        # Agregar arista primaria
        G.add_edge(u, v, **atributos)
        
        # Si no es sentido único, agregar el sentido inverso
        if not row['oneway']:
            G.add_edge(v, u, **atributos)
            
    print(f" Grafo construido con {G.number_of_nodes()} nodos y {G.number_of_edges()} aristas.")

    # 2. Guardar con compresión máxima
    print(f" Guardando archivo en {ruta_salida_grafo}...")
    with gzip.open(ruta_salida_grafo, "wb") as f:
        pickle.dump(G, f, protocol=pickle.HIGHEST_PROTOCOL)
    
    print("✨ ¡Proceso completado! Sube este nuevo archivo (.pkl.gz) a GitHub.")

if __name__ == "__main__":
    generar_y_guardar_grafo(file_path_red)
