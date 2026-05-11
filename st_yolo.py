from camera_input_live import camera_input_live
from PIL import Image
from ultralytics import YOLO

import cv2
import numpy as np
import streamlit as st
import torch


def reset_session_state():
    st.session_state.modo_entrada = "Archivo"
    st.session_state.imagen_actual = None
    st.session_state.file_uploader = None

"""
    Import the model and send it to the GPU, if available.
"""
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
vehicle_model = YOLO('./models/v_det/best.pt').to(device)

st.set_page_config(layout="wide", page_title="Detección de Vehículos", page_icon="🚗")
st.title("Vehicle Detection interface")

if 'modo_entrada' not in st.session_state:
    st.session_state.modo_entrada = "Archivo"
if 'imagen_actual' not in st.session_state:
    st.session_state.imagen_actual = None

with st.sidebar:
    st.header("Selecciona una opción:")

    entrada = st.radio("Fuente de imagen", 
                       ("Archivo", "Camara", "Video"),
                       index=0 if st.session_state.modo_entrada == "Archivo" else 1 if st.session_state.modo_entrada == "Camara" else 2,
                       on_change=reset_session_state,
                       horizontal=True)

    st.session_state.modo_entrada = "Archivo" if entrada == "Archivo" else "Camara" if entrada == "Camara" else "Video"

    st.divider()

    conf_threshold = st.slider(
        "Umbral de confianza:",
        min_value=0.0,
        max_value=1.0,
        value=0.7,
        step=0.05
    )

with st.container():
    if st.session_state.modo_entrada == "Archivo":
        uploaded_file = st.file_uploader("Sube una imagen", 
                                         type=["jpg", "jpeg", "png"],
                                         accept_multiple_files=False,
                                         key="file_uploader")
        
        if uploaded_file is not None:
            img = Image.open(uploaded_file).convert("RGB")
            img = cv2.cvtColor(np.array(img), cv2.COLOR_RGB2BGR)

            with st.spinner('Detectando objetos...'):
                det_results = vehicle_model.predict(img, device=device, conf=conf_threshold)
                img_detected = det_results[0].plot()
                st.session_state.imagen_actual = img_detected
    else:
        st.info("Mostrando la cámara en vivo. Asegúrate de que tu cámara esté conectada y funcionando.")
        with st.spinner('Accediendo a la cámara...'):
            img = camera_input_live()
            if img is not None:
                with st.spinner('Detectando objetos en la cámara...'):
                    img = Image.open(img).convert("RGB")
                    img = cv2.cvtColor(np.array(img), cv2.COLOR_RGB2BGR)
                    det_results = vehicle_model.predict(img, device=device, conf=conf_threshold)
                    img_detected = det_results[0].plot()
                    st.session_state.imagen_actual = img_detected
            else:
                st.error("No se pudo acceder a la cámara. Por favor, verifica tu conexión y permisos.")


st.subheader("Resultado de la detección:")
if st.session_state.imagen_actual is not None:
    st.image(st.session_state.imagen_actual, caption="Imagen con Detecciones", channels="BGR", width='stretch')