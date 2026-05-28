"""
    Package imports:

    camera_input_live: A streamlit component to capture live video feed from the camera.
    cv2: Used for modifying the image to draw the bounding boxes and the license plate text.
    numpy: Used for image manipulation and conversion between PIL and OpenCV formats.
    PIL: Used for image manipulation and conversion between OpenCV and PIL formats.
    pytesseract: Used for extracting text from the detected license plate regions.
    streamlit: Used for creating the web interface for uploading images and displaying results.
    torch: Enabling GPU usage to accelerate model training and inference.
    ultralytics: Provides the implementation, training and validation of the YOLO model, which is used for vehicle detection.
"""
from camera_input_live import camera_input_live
from PIL import Image
from ultralytics import YOLO

import cv2
import numpy as np
import pytesseract
import streamlit as st
import tempfile
import torch


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

        conf = box.conf[0].item()
        cls = int(box.cls[0].item())
        label = lp_model.names[cls]
        cv2.rectangle(img_copy, (x1, y1), (x2, y2), (255, 0, 0), 2)
        cv2.putText(img_copy, text, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 0), 4)
    
    return img_copy


# Import the models and load Tesseract OCR to be used on the detections.
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
vehicle_model, lp_model = load_models(device)
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
                       ("Archivo", "Camara", "Video"),
                       index=0 if st.session_state.modo_entrada == "Archivo" else 1 if st.session_state.modo_entrada == "Camara" else 2,
                       on_change=reset_session_state,
                       horizontal=True)

    st.session_state.modo_entrada = "Archivo" if entrada == "Archivo" else "Camara" if entrada == "Camara" else "Video"

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
    else:
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
                frame_count = 0

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

# Shows the detections only on the single image option
if st.session_state.imagen_actual is not None:
    st.subheader("Resultado de la detección:")
    st.image(st.session_state.imagen_actual, channels="BGR", width='stretch')
