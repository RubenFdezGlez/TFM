"""
    Package imports:

    camera_input_live: A streamlit component to capture live video feed from the camera.
    cv2: Modifies the image to draw the bounding boxes and the license plate text.
    datatime: Gets the current time for the reports.
    numpy: Used for image manipulation and conversion between PIL and OpenCV formats.
    os: Pre-creates the folder for storing the reports.
    PIL: Used for image manipulation and conversion between OpenCV and PIL formats.
    pytesseract: Extracts the text from the detected license plate regions.
    re: Regex library for checking if the license plates introduced are correct.
    streamlit: Used for creating the web interface for uploading images and displaying results.
    tempfile: Used to store the video in the video processing option.
    torch: Enables GPU usage to accelerate model inference.
    ultralytics: Provides the loading of the YOLO model, which is used for vehicle and license plate detection.
    xlsxwriter: Module for creating Excel files and write information like date, time and location into them.
    zoneInfo: Gets the time information for the local zone (Madrid).
"""
from camera_input_live import camera_input_live
import cv2
from datetime import datetime
import numpy as np
import os
from PIL import Image
import pytesseract
import re
import streamlit as st
from streamlit_webrtc import webrtc_streamer
import tempfile
import torch
from threading import Thread
from ultralytics import YOLO
import xlsxwriter
from zoneinfo import ZoneInfo


def reset_session_state():
    """Reset the session state variables when the input mode is changed."""
    st.session_state.modo_entrada = "Archivo"
    st.session_state.imagen_actual = None


def getTesseractText(cropped_img):
    """Extract text from the cropped license plate image using the engine Tesseract OCR. Removes any non alphanumeric characters (such as '&' or '%'), including the space.

    Args:
        cropped_img (MatLike): Image object from the cv2 module containing only the license plate region.
    
    Returns:
        text (str): The text extracted from the license plate region, containing only digits, uppercase letters and spaces.
    """
    text = pytesseract.image_to_string(cropped_img, config='--psm 8')
    text = [char for char in text if char.isdigit() or char.isspace() or char.isupper()]

    return ''.join(text)


@st.cache_resource
def load_models(device):
    """Load the vehicle and license plate detection models.

    Args:
        device (torch.device): The device to load the models on, either CPU or GPU.

    Returns:
        vehicle_model (YOLO): The loaded YOLO model for vehicle detection.
        lp_model (YOLO): The loaded YOLO model for license plate detection.
    """
    vehicle_model = YOLO('./models/v_det/best.pt').to(device)
    lp_model = YOLO('./models/lp_det/best.pt').to(device)

    return vehicle_model, lp_model


@st.dialog("Matrícula detectada")
def lp_detected(lp):
    """Shows a dialog when a license plate that is on the list of plates to detect is found in the image.

    Args:
        lp (str): The license plate that has been detected and is on the list of plates to detect.
    """
    st.write(f":red[La matrícula {lp} ha sido detectada]")


def valid_lp(lp):
    """Check if the license plate has the correct structure for Spanish license plates.
    
    Args:
        lp (str): The license plate to be checked.

    Returns:
        valid (bool): True if the license plate has a valid structure, False otherwise.
    """
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


def combineDetections(v_results, lp_results, img):
    """Joins both vehicle detection and license plate detection with the recognition on the license plate on a single image: with the bounding boxes and the license plate characters visible on the image.
    
    Args:
        v_results (Results): Results object obtained from the vehicle detection on the image/frame.
        lp_results (Results): Results object obtained from the license plate detection on the image/frame.
        img (MatLike): Image object from the cv2 module.
    
    Returns:
        img_copy (MatLike): Image object from the cv2 module with the bounding boxes and license plate text drawn on it.
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
        # cropped_img = cv2.resize(cropped_img, None, fx=5, fy=5, interpolation=cv2.INTER_CUBIC)
        # cropped_img = cv2.cvtColor(cropped_img, cv2.COLOR_BGR2RGB)
        # cv2.imshow('Imagen', cropped_img)
        # cv2.waitKey(0)
        # cv2.destroyAllWindows()

        # gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        # gray = cv2.bilateralFilter(gray, 11, 17, 17)
        # _, thresh = cv2.threshold(
        #     gray, 0, 255,
        #     cv2.THRESH_BINARY + cv2.THRESH_OTSU
        # )
        # gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

        text = pytesseract.image_to_string(
            cropped_img,
            config='--oem 3 --psm 6 -c tessedit_char_whitelist=ABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789 load_system_dawg=0 load_freq_dawg=0'
        )
        if text != "":  text = text.replace(text[-1],"")


        conf = box.conf[0].item()
        cls = int(box.cls[0].item())
        label = lp_model.names[cls]
        cv2.rectangle(img_copy, (x1, y1), (x2, y2), (255, 0, 0), 2)
        # if valid_lp(text):
        print(text)
        cv2.putText(img_copy, text, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 0, 0), 2)

        if os.path.exists("./web/lp.txt"):
            file = open("./web/lp.txt", "r")
            for line in file:
                if line != "" and text == line:
                    lp_detected(text)
            file.close()
    
    return img_copy


@st.cache_resource
def load_lp():
    """Load the license plates from the file.
    
    Returns:
        lps (list): A list of license plates to be detected, obtained from the file.
    """
    lps = []
    
    if os.path.exists("./web/lp.txt"):
        file = open("./web/lp.txt", "r")
        for line in file:
            if line != "":
                lps.append(line)
        file.close()

    return lps


@st.dialog("Lista de matrículas")
def lp_list():
    """Opens the list of license plates to be detected and allows the user to add or remove plates from the list. The list is stored in a text file and is loaded when the app is opened."""

    st.write("Añade matrículas para lanzar una advertencia cuando se detecte. Pulsa el botón guardar para poder almacenar las matrículas en el archivo.")
    col1, col2 = st.columns(2)
    with col1:
        with st.popover("Añadir matrícula"):
            lp = st.text_input("Introduce la matrícula. Sólo se permiten formatos de matrículas españolas y sin espacios.")
            if st.button("Añadir matrícula", key="addLPToList"):
                if valid_lp(lp):
                    st.session_state.lps.append(lp) 
                else:
                    st.write(":red[La matrícula introducida no tiene el formato correcto.]")
                   
    with col2:
        save_lp = st.button("Guardar", key="saveLPList")
    
    if save_lp:
        file = open("./web/lp.txt", "w")
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


def generate_report(cam_number, location):
    """Generates an Excel report with the date, time and location of the detections for each camera.
    Args:
        cam_number (str): The number of the camera for which the report is being generated.
        location (str): The location of the camera, to be included in the report.
    """
    os.makedirs("./reports", exist_ok=True)
    report = xlsxwriter.Workbook(f'./reports/report_camera_{cam_number}.xlsx')
    rep_sheet = report.add_worksheet()

    dt = datetime.now().astimezone(ZoneInfo('Europe/Madrid'))

    labels = ["Date", "Time", "Location"]
    data = [dt.strftime("%d/%m/%Y"), dt.strftime("%X"), location]

    for column, label in enumerate(labels):
        rep_sheet.write(0, column, label)

    for column, dat in enumerate(data):
        rep_sheet.write(1, column, dat)

    report.close()


def changePredictionState(index):
    """Changes the state of the prediction for the camera with the given index. If the state is "Activar predicción", it changes to "Parar predicción" and vice versa. This state is used to determine whether to perform the detection on the camera feed or not.
    Args:
        index (int): The index of the camera for which the prediction state is being changed.
    """
    st.session_state.predictions_state[index] = "Activar predicción" if st.session_state.predictions_state[index] == "Parar predicción" else "Parar predicción"   


# Import the models and the license plate list and load Tesseract OCR to be used on the detections.
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
vehicle_model, lp_model = load_models(device)
st.session_state.lps = load_lp()
pytesseract.pytesseract.tesseract_cmd = r'C:\Program Files\Tesseract-OCR\tesseract.exe'  


st.set_page_config(layout="wide", page_title="Detección de Vehículos", page_icon="🚗")
st.title("Detección de vehículos y matrículas")

# State variable for application option between: "Archivo", "Camara", "Video" and "Multi-cam".
if 'modo_entrada' not in st.session_state:
    st.session_state.modo_entrada = "Archivo"
# State variable for options "Archivo" and "Camara" to show the last image/frame on the screen.
if 'imagen_actual' not in st.session_state:
    st.session_state.imagen_actual = None
# State variable for option "Video" to show the last video on the screen.
if 'video_actual' not in st.session_state:
    st.session_state.video_actual = None 
# State variable for option "Video" to start or pause the video with the buttons.
if 'video_processing' not in st.session_state:
    st.session_state.video_processing = False
# State variables for option "Multi-cam" for changing the buttons values to start or stop doing predictions.
if 'predictions_state' not in st.session_state:
    st.session_state.predictions_state = ["Activar predicción","Activar predicción","Activar predicción","Activar predicción"]


with st.sidebar:
    st.header("Selecciona una opción:")

    entrada = st.radio("Fuente de imagen", 
                       ("Archivo", "Camara", "Video", "Multi-cam"),
                       index=0 if st.session_state.modo_entrada == "Archivo" else 1 if st.session_state.modo_entrada == "Camara" else 2 if st.session_state.modo_entrada == "Video" else 3,
                       on_change=reset_session_state,
                       horizontal=True)

    st.session_state.modo_entrada = "Archivo" if entrada == "Archivo" else "Camara" if entrada == "Camara" else "Video" if entrada == "Video" else "Multi-cam" 
    

    st.divider()

    if "lp_list" not in st.session_state:
        st.write("Lista de matrículas")
        if st.button("Cambiar", key="changeLpList"):
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
                                         key="img_uploader")
        
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
        uploaded_file = st.file_uploader("Sube un vídeo", 
                                         type=['mp4', 'avi', 'mov', 'mkv', 'webm'],
                                         accept_multiple_files=False,
                                         key="video_uploader")
        
        col1, col2 = st.columns(2)
        with col1:
            start_button = st.button("▶️ Iniciar", key="start_button")
        with col2:
            stop_button = st.button("⏹️ Detener", key="stop_button")

        # Loads the video into the program and resets the state variables for the processing.
        if uploaded_file is not None:
            tfile = tempfile.NamedTemporaryFile(delete=False, suffix='.mp4')
            tfile.write(uploaded_file.read())
            st.session_state.video_file_path = tfile.name
            st.session_state.last_uploaded_name = uploaded_file.name
            st.session_state.video_processing = False

        if start_button and uploaded_file is not None:
            st.session_state.video_processing = True
        
        if stop_button:
            st.session_state.video_processing = False
 
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
                    
                cap.release()

    # Option for multiple inputs or cameras
    elif st.session_state.modo_entrada == "Multi-cam":
        st.info("Mostrando la función multi-cámara. Asegúrate de que las cámaras estén conectadas y funcionando.")
        coords = ["42.588778, -5.576272", "42.596464, -5.577549", "42.588778, -5.576272", "42.596464, -5.577549"]

        col1_cam, col2_cam = st.columns(2)
        caps = []
        placeholders = []
        for i in range(4):
            if i % 2 == 0: 
                placeholder = col1_cam.empty()
                col1_rep, col2_pred = col1_cam.columns(2)
            else: 
                placeholder = col2_cam.empty()
                col1_rep, col2_pred = col2_cam.columns(2)
            placeholders.append(placeholder)
            
            with col1_rep:
                report = st.button(f"Informe cámara {i}", key=f"rep{i}")
            with col2_pred:
                pred = st.button(st.session_state.predictions_state[i], key=f"inf{i}")

            if report:
                generate_report(f"{i}", coords[i])
            if pred:
                changePredictionState(i)

            cap = cv2.VideoCapture(i)
            caps.append(cap)
            
        while(caps[0].isOpened()):
            for i, (cap, placeholder) in enumerate(zip(caps, placeholders)):
                if not cap.isOpened(): st.warning(f"No se pudo abrir la cámara {i}") 
                else:
                    ret, frame = cap.read() if cap.isOpened() else (False, None)
            
                    if ret and frame is not None:

                        if st.session_state.predictions_state[i] == "Parar predicción":
                            frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

                            v_results = vehicle_model.predict(frame, device=device, conf=v_conf_threshold)
                            lp_results = lp_model.predict(frame, device=device, conf=lp_conf_threshold)
                            img_detected = combineDetections(v_results, lp_results, frame)

                            placeholder.image(img_detected, caption=f"Cámara {i}", width="stretch")
                        else:
                            placeholder.image(frame, caption=f"Cámara {i}", width="stretch")



# Shows the detections only on the single image option
if st.session_state.imagen_actual is not None:
    st.subheader("Resultado de la detección:")
    st.image(st.session_state.imagen_actual, channels="BGR", width='stretch')