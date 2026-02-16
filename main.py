# def main():
#     print("Hello from saferoute-ai!")


# if __name__ == "__main__":
#     main()


import streamlit as st
import folium
import pandas as pd
import networkx as nx
from shapely import wkb
from streamlit_folium import st_folium


# Datos de las 16 alcaldías (Centroides aproximados)
data = {
    'alcaldia': [
        'Álvaro Obregón', 'Azcapotzalco', 'Benito Juárez', 'Coyoacán', 
        'Cuajimalpa de Morelos', 'Cuauhtémoc', 'Gustavo A. Madero', 'Iztacalco', 
        'Iztapalapa', 'La Magdalena Contreras', 'Miguel Hidalgo', 'Milpa Alta', 
        'Tláhuac', 'Tlalpan', 'Venustiano Carranza', 'Xochimilco'
    ],
    'lat': [
        19.3585, 19.4844, 19.3806, 19.3502, 
        19.3581, 19.4451, 19.4925, 19.3953, 
        19.3551, 19.3032, 19.4316, 19.1917, 
        19.2736, 19.2212, 19.4323, 19.2541
    ],
    'lon': [
        -99.2033, -99.1859, -99.1611, -99.1615, 
        -99.2871, -99.1462, -99.1171, -99.0974, 
        -99.0621, -99.2372, -99.1915, -99.0232, 
        -99.0041, -99.1687, -99.0886, -99.1032
    ]
}

df_alcaldias = pd.DataFrame(data)


st.title("SafeRoute AI")
st.write("This is a demo of the SafeRoute AI model.")
st.write("Enter your text below to see the model's response.")

# text = st.text_input("Enter your text here")
# st.write(text)

m = folium.Map(location=[19.4326, -99.1332], zoom_start=12)

# for idx, row in df_alcaldias.iterrows():
#     folium.Marker(
#         location=[row['lat'], row['lon']],
#         popup=row['alcaldia'],
#         icon=folium.Icon(color='blue')
#     ).add_to(m)

st_folium(m, width=1200, height=600)