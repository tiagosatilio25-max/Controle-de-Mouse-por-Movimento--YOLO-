import cv2
import pyautogui
import streamlit as st
from ultralytics import YOLO

# Configurações de segurança e performance do PyAutoGUI
pyautogui.FAILSAFE = True
pyautogui.PAUSE = 0.0

st.set_page_config(
    page_title="YOLO Mouse Control", page_icon="🖱️", layout="wide"
)

st.title("Controle de Mouse por Movimento (YOLO)")
st.write("O cursor seguirá o centro da sua detecção na câmera.")

# Controles na barra lateral
st.sidebar.header("Configurações")
smoothing = st.sidebar.slider("Suavização (Lag vs Tremor)", 1, 10, 5)

run = st.checkbox("Ativar Câmera e Controle")
col1, col2 = st.columns([2, 1])

with col1:
    frame_window = st.image([])

with col2:
    st.info(
        "**Instruções:**\n\n1. Posicione-se em frente à câmera.\n2. O mouse"
        " seguirá o centro da sua detecção.\n3. Para abortar/travar, jogue o"
        " mouse rapidamente para o canto superior esquerdo da tela (Failsafe)."
    )

# Carregamento otimizado do modelo YOLO com cache
@st.cache_resource
def load_yolo_model():
    return YOLO("yolov8n.pt")


model = load_yolo_model()
camera = cv2.VideoCapture(0)

# Resolução da tela para o mapeamento proporcional
screen_w, screen_h = pyautogui.size()
prev_x, prev_y = 0, 0

try:
    while run:
        ret, frame = camera.read()
        if not ret:
            st.error("Não foi possível acessar a webcam.")
            break

        # Espelhar para controle intuitivo estilo espelho
        frame = cv2.flip(frame, 1)
        h, w, _ = frame.shape

        # Inferência YOLO focando na classe 'person' (classe 0 do COCO)
        results = model(frame, verbose=False, classes=[0])
        annotated_frame = results[0].plot()

        boxes = results[0].boxes
        if len(boxes) > 0:
            # Pega a primeira pessoa detectada na cena
            box = boxes[0].xyxy[0].cpu().numpy()
            x1, y1, x2, y2 = map(int, box)

            # Calcula o centroide da bounding box
            cx = int((x1 + x2) / 2)
            cy = int((y1 + y2) / 2)

            # Destaca o ponto central na imagem visualizada
            cv2.circle(annotated_frame, (cx, cy), 10, (0, 255, 255), -1)

            # Mapeamento proporcional para a resolução da tela
            target_x = int((cx / w) * screen_w)
            target_y = int((cy / h) * screen_h)

            # Suavização (Low-pass filter) para evitar tremores bruscos
            smoothed_x = int(prev_x + (target_x - prev_x) / smoothing)
            smoothed_y = int(prev_y + (target_y - prev_y) / smoothing)

            # Move o cursor do mouse
            pyautogui.moveTo(smoothed_x, smoothed_y)
            prev_x, prev_y = smoothed_x, smoothed_y

        # Conversão de cores e renderização eficiente no Streamlit
        annotated_frame = cv2.cvtColor(annotated_frame, cv2.COLOR_BGR2RGB)
        frame_window.image(annotated_frame, channels="RGB")

except:
    camera.release()
    st.info("Câmera desligada.")