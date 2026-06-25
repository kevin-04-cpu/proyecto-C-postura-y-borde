import streamlit as st
import cv2
import numpy as np
import os
import time

# Configuración de dimensiones de la arquitectura de la red
INPUT_SIZE = 4096  # 64x64
HIDDEN_SIZE = 128
OUTPUT_SIZE = 1

st.set_page_config(page_title="Detector de Postura - HPC", layout="centered")

# =========================================================================
# INYECCIÓN DE CSS  
# =========================================================================
st.markdown("""
    <style>
    /* Ocultar UI nativa de Streamlit */
    #MainMenu {visibility: hidden;}
    header {visibility: hidden;}
    footer {visibility: hidden;}
    
    /* Fondo principal y reducción de márgenes */
    .block-container { padding-top: 1rem !important; max-width: 850px; }

    /* Tipografía Premium y Gradientes para el Título */
    .title-container { text-align: center; margin-bottom: 2rem; padding-bottom: 1.5rem; border-bottom: 1px solid rgba(255, 255, 255, 0.1); }
    .main-title { font-size: 2.4rem; font-weight: 800; background: linear-gradient(135deg, #00F2FE 0%, #4FACFE 100%); -webkit-background-clip: text; -webkit-text-fill-color: transparent; margin-bottom: 0.5rem; letter-spacing: -1px; }
    .sub-title { color: #e6edf3; font-size: 1.1rem; font-weight: 600; margin-bottom: 10px; }
    .description-text { color: #8b949e; font-size: 0.95rem; line-height: 1.5; max-width: 650px; margin: 0 auto; }

    /* Indicadores de Estado (Pills) */
    .status-pill { display: inline-block; padding: 6px 16px; border-radius: 50px; font-size: 0.85rem; font-weight: 600; letter-spacing: 0.5px; text-transform: uppercase; margin-bottom: 1.5rem; }
    .status-ok { background-color: rgba(46, 204, 113, 0.15); color: #2ecc71; border: 1px solid rgba(46, 204, 113, 0.3); }
    .status-err { background-color: rgba(231, 76, 60, 0.15); color: #e74c3c; border: 1px solid rgba(231, 76, 60, 0.3); }

    /* Alinear el selector de radio al centro */
    div.row-widget.stRadio > div { display: flex; justify-content: center; margin-bottom: 15px; }

    /* Tarjetas de Imágenes con Hover Effect y Sombras */
    [data-testid="stImage"] img { border-radius: 16px; box-shadow: 0 10px 30px rgba(0, 0, 0, 0.5); border: 1px solid rgba(255, 255, 255, 0.08); transition: transform 0.3s ease, box-shadow 0.3s ease; }
    [data-testid="stImage"] img:hover { transform: translateY(-5px); box-shadow: 0 15px 40px rgba(0, 242, 254, 0.2); }
    .image-caption { text-align: center; font-size: 0.9rem; color: #a1aab3; margin-top: 12px; font-weight: 500; background: rgba(255, 255, 255, 0.05); padding: 6px 0; border-radius: 8px; }

    /* Tarjeta de Resultado Final (Glassmorphism) */
    .result-card { padding: 24px; border-radius: 20px; text-align: center; backdrop-filter: blur(10px); margin-top: 2rem; animation: fadeIn 0.6s ease-out; }
    .result-good { background: linear-gradient(145deg, rgba(46, 204, 113, 0.1), rgba(39, 174, 96, 0.05)); border: 1px solid rgba(46, 204, 113, 0.3); box-shadow: 0 0 30px rgba(46, 204, 113, 0.15); }
    .result-bad { background: linear-gradient(145deg, rgba(231, 76, 60, 0.1), rgba(192, 57, 43, 0.05)); border: 1px solid rgba(231, 76, 60, 0.3); box-shadow: 0 0 30px rgba(231, 76, 60, 0.15); }
    .result-title { font-size: 1.5rem; font-weight: 800; margin-bottom: 8px; }
    .result-subtitle { font-size: 0.9rem; opacity: 0.9; margin-top: 10px;}
    .text-good { color: #2ecc71; }
    .text-bad { color: #e74c3c; }

    @keyframes fadeIn { from { opacity: 0; transform: translateY(20px); } to { opacity: 1; transform: translateY(0); } }
    </style>
""", unsafe_allow_html=True)

# =========================================================================
# ENCABEZADO DE LA APLICACIÓN 
# =========================================================================
st.markdown("""
    <div class="title-container">
        <div class="main-title">Clasificador de Postura Corporal desde Cero</div>
        <div class="sub-title">Cómputo Paralelo y Distribuido - Universidad de Pamplona</div>
        <div class="description-text">Esta aplicación ejecuta inferencia cargando los pesos entrenados en CUDA (GPU) y aplicando el pipeline matemático de Sobel.</div>
    </div>
""", unsafe_allow_html=True)

# =========================================================================
# FUNCIONES MATEMÁTICAS: REPLICANDO EL PIPELINE DE C
# =========================================================================

def apply_sobel_python(img_gray):
    """Réplica exacta de la convolución Sobel implementada en C."""
    # Kernels de Sobel
    gx = np.array([[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]], dtype=np.float32)
    gy = np.array([[-1, -2, -1], [0, 0, 0], [1, 2, 1]], dtype=np.float32)
    
    # Aplicar convolución usando OpenCV (solo para operaciones de filtrado base)
    grad_x = cv2.filter2D(img_gray, cv2.CV_32F, gx)
    grad_y = cv2.filter2D(img_gray, cv2.CV_32F, gy)
    
    # Calcular magnitud
    magnitude = np.sqrt(grad_x**2 + grad_y**2)
    
    # Inclusión de saturación [0, 255] tal como se hizo en preprocess.c
    magnitude = np.clip(magnitude, 0, 255).astype(np.uint8)
    return magnitude

def hay_persona(img_color):
    """Detecta si hay una persona en la imagen, de frente o de perfil.
    Devuelve True si encuentra al menos un rostro, False si no."""
    # Cargar los clasificadores incluidos en OpenCV
    frontal_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + "haarcascade_frontalface_default.xml")
    profile_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + "haarcascade_profileface.xml")

    # Escala de grises + ecualización de histograma (mejora el contraste para detectar)
    gray = cv2.cvtColor(img_color, cv2.COLOR_BGR2GRAY)
    gray = cv2.equalizeHist(gray)

    # Parámetros más sensibles para reconocer caras inclinadas o parcialmente tapadas
    sf, mn, ms = 1.05, 2, (30, 30)

    # 1. Rostros de frente
    if len(frontal_cascade.detectMultiScale(gray, scaleFactor=sf, minNeighbors=mn, minSize=ms)) > 0: return True
    # 2. Rostros de perfil (un lado)
    if len(profile_cascade.detectMultiScale(gray, scaleFactor=sf, minNeighbors=mn, minSize=ms)) > 0: return True
    # 3. Perfil del otro lado (imagen volteada)
    gray_volteado = cv2.flip(gray, 1)
    if len(profile_cascade.detectMultiScale(gray_volteado, scaleFactor=sf, minNeighbors=mn, minSize=ms)) > 0: return True

    return False

def preprocess_image(image_bytes):
    """Transforma la imagen capturada al formato del vector de entrada de 4096."""
    # Convertir bytes a arreglo OpenCV
    file_bytes = np.asarray(bytearray(image_bytes), dtype=np.uint8)
    img = cv2.imdecode(file_bytes, cv2.IMREAD_COLOR)
    
    # 1. Escala de grises (Luminosity formula equivalente a STB)
    img_gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    
    # 2. Redimensionar a 64x64 (Linear interpolación)
    img_resized = cv2.resize(img_gray, (64, 64), interpolation=cv2.INTER_LINEAR)
    
    # 3. Aplicar Filtro Sobel
    img_sobel = apply_sobel_python(img_resized)
    
    # 4. Aplanar y Normalizar [0, 1]
    flattened = img_sobel.flatten().astype(np.float32) / 255.0
    
    return img, img_sobel, flattened

@st.cache_resource
def load_trained_weights():
    """Carga los archivos binarios exportados por el programa de CUDA."""
    path = "../weights/"
    try:
        w1 = np.fromfile(os.path.join(path, "W1.bin"), dtype=np.float32).reshape(HIDDEN_SIZE, INPUT_SIZE)
        b1 = np.fromfile(os.path.join(path, "b1.bin"), dtype=np.float32).reshape(HIDDEN_SIZE)
        w2 = np.fromfile(os.path.join(path, "W2.bin"), dtype=np.float32).reshape(OUTPUT_SIZE, HIDDEN_SIZE)
        b2 = np.fromfile(os.path.join(path, "b2.bin"), dtype=np.float32).reshape(OUTPUT_SIZE)
        return w1, b1, w2, b2, True
    except Exception as e:
        return None, None, None, None, False

# =========================================================================
# FLUJO PRINCIPAL DE LA APLICACIÓN
# =========================================================================

w1, b1, w2, b2, success = load_trained_weights()

# Reemplazo de st.success por el pill de estado centrado
estado_html = "<div style='text-align: center;'>"
if success:
    estado_html += "<span class='status-pill status-ok'>● Pesos del modelo neuronal (W1, b1, W2, b2) cargados con éxito</span>"
else:
    estado_html += "<span class='status-pill status-err'>● Error al cargar los pesos binarios</span>"
estado_html += "</div>"
st.markdown(estado_html, unsafe_allow_html=True)

if success:
    st.markdown("<p style='text-align: center; color: #8b949e; font-weight: 500;'>Selecciona el método de entrada de imagen:</p>", unsafe_allow_html=True)
    
    # Selector de entrada de imagen (Horizontal para ahorrar espacio)
    option = st.radio("Método", ["📁 Cargar Archivo", "📷 Cámara Web en Vivo"], horizontal=True, label_visibility="collapsed")
    
    img_file = None
    if option == "📷 Cámara Web en Vivo":
        img_file = st.camera_input("Captura tu postura frente a la pantalla", label_visibility="collapsed")
    else:
        img_file = st.file_uploader("Elige una foto...", type=["jpg", "jpeg", "png"], label_visibility="collapsed")
        
    if img_file is not None:
        # Spinner visual añadido
        with st.spinner('Procesando imagen...'):
            time.sleep(0.5)
            bytes_data = img_file.read()
            
            # Ejecutar preprocesamiento idéntico al de C
            img_original, img_sobel, x_vector = preprocess_image(bytes_data)
        
        # Verificar que haya una persona antes de predecir la postura
        if not hay_persona(img_original):
            st.error("⚠️ No se detecta ninguna persona en la imagen. Por favor, asegúrate de estar visible frente a la cámara e inténtalo de nuevo.")
            st.stop()
        
        st.markdown("<br>", unsafe_allow_html=True)
        
        # Mostrar imágenes de diagnóstico en la App (Centradas)
        col_espacio1, col1, col2, col_espacio2 = st.columns([0.5, 4, 4, 0.5])
        with col1:
            st.image(img_original, channels="BGR", use_container_width=True)
            st.markdown("<div class='image-caption'>Imagen Original Capturada</div>", unsafe_allow_html=True)
        with col2:
            st.image(img_sobel, use_container_width=True)
            st.markdown("<div class='image-caption'>Pipeline de Bordes Sobel (64x64)</div>", unsafe_allow_html=True)
            
        # ---------------------------------------------------------------------
        # FORWARD PASS EN PYTHON (Puro NumPy, Cero Frameworks)
        # ---------------------------------------------------------------------
        # Capa Oculta: Z1 = X * W1^T + b1 -> A1 = ReLU(Z1)
        z1 = np.dot(x_vector, w1.T) + b1
        a1 = np.maximum(0, z1)
        
        # Capa de Salida: Z2 = A1 * W2^T + b2 -> A2 = Sigmoid(Z2)
        z2 = np.dot(a1, w2.T) + b2
        a2 = 1.0 / (1.0 + np.exp(-z2))
        
        probabilidad = a2[0]
        
        # Determinar clase basada en el umbral estándar (0.5)
        if probabilidad >= 0.5:
            st.markdown(f"""
                <div class="result-card result-bad">
                    <div class="result-title text-bad">POSTURA ENCORVADA DETECTADA</div>
                    <div class="result-subtitle">Consejos de Ergonomía: Alinea tus hombros con tus orejas, mantén la espalda apoyada en el respaldo y levanta tu monitor.</div>
                    <div style="margin-top: 15px; font-weight: bold;">Probabilidad: {probabilidad*100:.2f}%</div>
                </div>
            """, unsafe_allow_html=True)
        else:
            st.markdown(f"""
                <div class="result-card result-good">
                    <div class="result-title text-good">POSTURA RECTA CORRECTA</div>
                    <div class="result-subtitle">¡Excellent! Mantienes una higiene postural idónea para largas jornadas de programación.</div>
                    <div style="margin-top: 15px; font-weight: bold;">Probabilidad de encorvamiento: {probabilidad*100:.2f}%</div>
                </div>
            """, unsafe_allow_html=True)