"""
    Package imports:

    camera_input_live: A streamlit component to capture live video feed from the camera.
    cv2: Modifies the image to draw the bounding boxes and the license plate text.
    numpy: Used for image manipulation and conversion between PIL and OpenCV formats.
    PIL: Used for image manipulation and conversion between OpenCV and PIL formats.
    pytesseract: Extracts the text from the detected license plate regions.
    re: Regex library for checking if the license plates introduced are correct.
    streamlit: Used for creating the web interface for uploading images and displaying results.
    tempfile: Used to store the video in the video processing option.
    torch: Enables GPU usage to accelerate model inference.
    ultralytics: Provides the loading of the YOLO model, which is used for vehicle and license plate detection.
"""
from camera_input_live import camera_input_live
import cv2
import numpy as np
from PIL import Image
import pytesseract
import re
import streamlit as st
from streamlit_webrtc import webrtc_streamer
import tempfile
import torch
from ultralytics import YOLO


def reset_session_state():
    """Reset the session state variables when the input mode is changed."""
    st.session_state.modo_entrada = "Archivo"
    st.session_state.imagen_actual = None


def getTesseractText(cropped_img):
    """Extract text from the cropped license plate image using the engine Tesseract OCR."""
    text = pytesseract.image_to_string(cropped_img, config='--psm 8')
    text = [char for char in text if char.isdigit() or char.isspace() or char.isupper()]
    return ''.join(text)


@st.cache_resource
def load_models(device):
    """Load the vehicle and license plate detection models."""
    vehicle_model = YOLO('./models/v_det/best.pt').to(device)
    lp_model = YOLO('./models/lp_det/best.pt').to(device)

    return vehicle_model, lp_model


@st.dialog("Matrícula detectada")
def lp_detected(lp):
    """Shows a dialog when a license plate that is on the list of plates to detect is found in the image."""
    st.write(f":red[La matrícula {lp} ha sido detectada]")


def combineDetections(v_results, lp_results, img):
    """Joins both vehicle detection and license plate detection with the recognition on the license plate on a single image: with the bounding boxes and the license plate characters visible on the image.
    
    Args:
        v_results (Results): Results object obtained from the vehicle detection on the image/frame.
        lp_results (Results): Results object obtained from the license plate detection on the image/frame.
        img (MatLike): Image object from the cv2 module.
    """
    img_copy = img

    for box in v_results[0].boxes:
        x1, y1, x2, y2 = map(int, box.xyxy[0])
        conf = box.conf[0].item()
        cls = int(box.cls[0].item())
        label = vehicle_model.names[cls]
        cv2.rectangle(img_copy, (x1, y1), (x2, y2), (0, 255, 0), 2)
        cv2.putText(img_copy, f"{label} {conf:.2f}", (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

    for box in lp_results[0].boxes:
        x1, y1, x2, y2 = map(int, box.xyxy[0])
        cropped_img = img[y1:y2, x1:x2]

        text = getTesseractText(cropped_img)
        text = text.replace(" ", "")

        conf = box.conf[0].item()
        cls = int(box.cls[0].item())
        label = lp_model.names[cls]
        cv2.rectangle(img_copy, (x1, y1), (x2, y2), (255, 0, 0), 2)
        cv2.putText(img_copy, text, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 0), 4)

        file = open("./web_app/lp.txt", "r")
        for line in file:
            if line != "" and text == line:
                lp_detected(text)
        file.close()
    
    return img_copy


@st.cache_resource
def load_lp():
    """Load the license plates from the file."""
    lps = []

    file = open("./web_app/lp.txt", "r")
    for line in file:
        if line != "":
            lps.append(line)
    file.close()

    return lps


def valid_lp(lp):
    """Check if the license plate has the correct structure"""

    # Matrícula normal de vehículos (también se aplica a taxis y VCTs)
    if re.fullmatch(r'[0-9]{4}[A-Z]{3}', lp):
        return True 
    # Matrícula antigua provincial numérica
    if re.fullmatch(r'[A-Z][0-9]{6}', lp):
        return True 
    # Matrícula antigua provincial alfanumérica
    if re.fullmatch(r'[A-Z][0-9]{4}[A-Z]{2}', lp):
        return True 
    # Matrícula histórica
    elif re.fullmatch(r'H[0-9]{4}[A-Z]{3}', lp):
        return True
    # Matrícula de fuerzas del estado
    elif re.fullmatch(r'(PGC|CNP|ET|EA|FN|E[0-9]|CGPC)[0-9]{4}B', lp):
        return True
    # Matrícula de remolques y semiremolques
    elif re.fullmatch(r'[A-Z][A-Z]{3}', lp):
        return True
    # Matrícula de ciclomotores
    elif re.fullmatch(r'C[0-9]{4}[A-Z]{3}', lp):
        return True
    # Matrícula temporal
    elif re.fullmatch(r'P[0-9]{4}[A-Z]{3}', lp):
        return True
    # Matrícula diplomáticas
    elif re.fullmatch(r'((CD|CC)[0-9]{2}|(TA|OI)[0-9]{3})[0-9]{3}', lp):
        return True
    return False


@st.dialog("Lista de matrículas")
def lp_list():
    """Opens the list of license plates to be detected and allows the user to add or remove plates from the list. The list is stored in a text file and is loaded when the app is opened."""

    st.write("Añade matrículas para lanzar una advertencia cuando se detecte. Pulsa el botón guardar para poder almacenar las matrículas en el archivo.")
    col1, col2 = st.columns(2)
    with col1:
        with st.popover("Añadir matrícula"):
            lp = st.text_input("Introduce la matrícula. Sólo se permiten formatos de matrículas españolas y sin espacios.")
            if st.button("Añadir matrícula"):
                if valid_lp(lp):
                    st.session_state.lps.append(lp) 
                else:
                    st.write(":red[La matrícula introducida no tiene el formato correcto.]")
                   
    with col2:
        save_lp = st.button("Guardar")
    
    if save_lp:
        file = open("./web_app/lp.txt", "w")
        [file.write(lp.replace("\n","") + "\n") for lp in st.session_state.lps]
        file.close()
        st.rerun()

    with st.container():

        for idx, lp in enumerate(st.session_state.lps):
            cont_col1, cont_col2 = st.columns([3,1])

            with cont_col1:
                st.write(lp) 
            with cont_col2:
                if st.button("Remove", key=str(idx)+lp):
                    st.session_state.lps.pop(idx)
        

# Import the models and the license plate list and load Tesseract OCR to be used on the detections.
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
vehicle_model, lp_model = load_models(device)
st.session_state.lps = load_lp()
pytesseract.pytesseract.tesseract_cmd = r'C:\Program Files\Tesseract-OCR\tesseract.exe'  


st.set_page_config(layout="wide", page_title="Detección de Vehículos", page_icon="🚗")
st.title("Detección de vehículos y matrículas")

if 'modo_entrada' not in st.session_state:
    st.session_state.modo_entrada = "Archivo"
if 'imagen_actual' not in st.session_state:
    st.session_state.imagen_actual = None
if 'video_actual' not in st.session_state:
    st.session_state.video_actual = None 
if 'video_processing' not in st.session_state:
    st.session_state.video_processing = False
if 'current_frame' not in st.session_state:
    st.session_state.current_frame = None
if 'stop_video' not in st.session_state:
    st.session_state.stop_video = False


with st.sidebar:
    st.header("Selecciona una opción:")

    entrada = st.radio("Fuente de imagen", 
                       ("Archivo", "Camara", "Video", "Multi-cam"),
                       index=0 if st.session_state.modo_entrada == "Archivo" else 1 if st.session_state.modo_entrada == "Camara" else 2 if st.session_state.modo_entrada == "Multi-cam" else 3,
                       on_change=reset_session_state,
                       horizontal=True)

    st.session_state.modo_entrada = "Archivo" if entrada == "Archivo" else "Camara" if entrada == "Camara" else "Video"\
          if entrada == "Video" else "Multi-cam" 

    st.divider()

    if "lp_list" not in st.session_state:
        st.write("Lista de matrículas")
        if st.button("Cambiar"):
            lp_list()

    st.divider()

    v_conf_threshold = st.slider(
        "Umbral de confianza para vehículos:",
        min_value=0.0,
        max_value=1.0,
        value=0.7,
        step=0.05
    )

    lp_conf_threshold = st.slider(
        "Umbral de confianza para matrículas:",
        min_value=0.0,
        max_value=1.0,
        value=0.7,
        step=0.05
    )


with st.container():
    # Option for single image detection
    if st.session_state.modo_entrada == "Archivo":
        uploaded_file = st.file_uploader("Sube una imagen", 
                                         type=["jpg", "jpeg", "png"],
                                         accept_multiple_files=False,
                                         key="file_uploader")
        
        if uploaded_file is not None:
            img = Image.open(uploaded_file).convert("RGB")
            img = cv2.cvtColor(np.array(img), cv2.COLOR_RGB2BGR)

            with st.spinner('Detectando objetos...'):
                v_results = vehicle_model.predict(img, device=device, conf=v_conf_threshold)
                lp_results = lp_model.predict(img, device=device, conf=lp_conf_threshold)

                img_copy = combineDetections(v_results, lp_results, img)
                st.session_state.imagen_actual = img_copy
    # Option for live camera feed
    elif st.session_state.modo_entrada == "Camara":
        st.info("Mostrando la cámara en vivo. Asegúrate de que tu cámara esté conectada y funcionando.")
        with st.spinner('Accediendo a la cámara...'):
            img = camera_input_live()
            if img is not None:
                with st.spinner('Detectando objetos en la cámara...'):
                    img = Image.open(img).convert("RGB")
                    img = cv2.cvtColor(np.array(img), cv2.COLOR_RGB2BGR)

                    v_results = vehicle_model.predict(img, device=device, conf=v_conf_threshold)
                    lp_results = lp_model.predict(img, device=device, conf=lp_conf_threshold)
                    
                    img_detected = combineDetections(v_results, lp_results, img)
                    st.session_state.imagen_actual = img_detected
            else:
                st.error("No se pudo acceder a la cámara. Por favor, verifica tu conexión y permisos.")
    # Option for single video files
    elif st.session_state.modo_entrada == "Video":
        uploaded_file = st.file_uploader("Sube una imagen", 
                                         type=['mp4', 'avi', 'mov', 'mkv', 'webm'],
                                         accept_multiple_files=False,
                                         key="file_uploader")
        
        col1, col2 = st.columns(2)
        with col1:
            start_button = st.button("▶️ Iniciar", type="primary", key="start_button")
        with col2:
            stop_button = st.button("⏹️ Detener", key="stop_button")

        # Loads the video into the program and resets the state variables for the processing.
        if uploaded_file is not None:
            tfile = tempfile.NamedTemporaryFile(delete=False, suffix='.mp4')
            tfile.write(uploaded_file.read())
            st.session_state.video_file_path = tfile.name
            st.session_state.last_uploaded_name = uploaded_file.name
            st.session_state.video_processing = False
            st.session_state.current_frame = None

        if start_button:
            st.session_state.video_processing = True
            st.session_state.stop_video = False
        
        if stop_button:
            st.session_state.video_processing = False
            st.session_state.stop_video = True
 
        # Open the video capture on the file, read each frame and do detection on each one
        if st.session_state.video_processing:
            cap = cv2.VideoCapture(st.session_state.video_file_path)

            if cap.isOpened():
                video_placeholder = st.empty()

                while cap.isOpened():
                    ret, frame = cap.read()
                    if not ret:
                        cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
                        continue

                    img = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    
                    v_results = vehicle_model.predict(img, device=device, conf=v_conf_threshold)
                    lp_results = lp_model.predict(img, device=device, conf=lp_conf_threshold)
                    
                    detected_frame = combineDetections(v_results, lp_results, img)
                    video_placeholder.image(detected_frame, channels="BGR", width='stretch')
                    
                    # Check if stop was requested
                    if st.session_state.get('stop_video', False):
                        st.session_state.video_processing = False
                        break
                cap.release()
    else:
        st.info("Mostrando la función multi-cámara. Asegúrate de que las cámaras estén conectadas y funcionando.")
        with st.spinner("Accediendo a las cámaras"):
            col1, col2 = st.columns(2)

            with col1:
                placeholder1 = st.empty()
                bt1 = st.button("Informe cámara 1")
                placeholder3 = st.empty()
                bt2 = st.button("Informe cámara 3")

            with col2:
                placeholder2 = st.empty()
                bt2 = st.button("Informe cámara 2")
                placeholder4 = st.empty()
                bt4 = st.button("Informe cámara 4")

            # Inicializar capturas con diferentes índices
            cap1 = cv2.VideoCapture(0)
            cap2 = cv2.VideoCapture(1)
            cap3 = cv2.VideoCapture(2)
            cap4 = cv2.VideoCapture(3)

            if not cap1.isOpened():
                st.error("No se pudo abrir la cámara 1")
            if not cap2.isOpened():
                st.warning("No se pudo abrir la cámara 2")
            if not cap3.isOpened():
                st.warning("No se pudo abrir la cámara 3")
            if not cap4.isOpened():
                st.warning("No se pudo abrir la cámara 4")

            while cap1.isOpened() or cap2.isOpened() or cap3.isOpened() or cap4.isOpened():
                # Leer frames
                ret1, frame1 = cap1.read() if cap1.isOpened() else (False, None)
                ret2, frame2 = cap2.read() if cap2.isOpened() else (False, None)
                ret3, frame3 = cap3.read() if cap3.isOpened() else (False, None)
                ret4, frame4 = cap4.read() if cap4.isOpened() else (False, None)
                
                if ret1 and frame1 is not None:
                    frame1 = cv2.cvtColor(frame1, cv2.COLOR_BGR2RGB)

                    v_results = vehicle_model.predict(frame1, device=device, conf=v_conf_threshold)
                    lp_results = lp_model.predict(frame1, device=device, conf=lp_conf_threshold)
                    img_detected = combineDetections(v_results, lp_results, frame1)

                    placeholder1.image(frame1, caption="Cámara 1", use_container_width=True)
                
                if ret2 and frame2 is not None:
                    frame2 = cv2.cvtColor(frame2, cv2.COLOR_BGR2RGB)

                    v_results = vehicle_model.predict(frame2, device=device, conf=v_conf_threshold)
                    lp_results = lp_model.predict(frame2, device=device, conf=lp_conf_threshold)
                    img_detected = combineDetections(v_results, lp_results, frame2)

                    placeholder2.image(img_detected, caption="Cámara 2", use_container_width=True)

                if ret3 and frame3 is not None:
                    frame3 = cv2.cvtColor(frame3, cv2.COLOR_BGR2RGB)

                    v_results = vehicle_model.predict(frame3, device=device, conf=v_conf_threshold)
                    lp_results = lp_model.predict(frame3, device=device, conf=lp_conf_threshold)
                    img_detected = combineDetections(v_results, lp_results, frame3)

                    placeholder2.image(img_detected, caption="Cámara 3", use_container_width=True)

                if ret4 and frame4 is not None:
                    frame4 = cv2.cvtColor(frame4, cv2.COLOR_BGR2RGB)

                    v_results = vehicle_model.predict(frame4, device=device, conf=v_conf_threshold)
                    lp_results = lp_model.predict(frame4, device=device, conf=lp_conf_threshold)
                    img_detected = combineDetections(v_results, lp_results, frame4)

                    placeholder2.image(img_detected, caption="Cámara 2", use_container_width=True)


# Shows the detections only on the single image option
if st.session_state.imagen_actual is not None:
    st.subheader("Resultado de la detección:")
    st.image(st.session_state.imagen_actual, channels="BGR", width='stretch')
