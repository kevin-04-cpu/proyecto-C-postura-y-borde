import streamlit as st
import cv2
import numpy as np
import os

# Configuración de dimensiones de la arquitectura de la red
INPUT_SIZE = 4096  # 64x64
HIDDEN_SIZE = 128
OUTPUT_SIZE = 1

st.set_page_config(page_title="Detector de Postura - HPC", layout="centered")

st.title("Clasificador de Postura Corporal desde Cero")
st.subheader("Cómputo Paralelo y Distribuido - Universidad de Pamplona")
st.write("Esta aplicación ejecuta inferencia cargando los pesos entrenados en CUDA (GPU) y aplicando el pipeline matemático de Sobel.")

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
        st.error(f"Error al cargar los pesos binarios: {e}")
        return None, None, None, None, False

# =========================================================================
# FLUJO PRINCIPAL DE LA APLICACIÓN
# =========================================================================

w1, b1, w2, b2, success = load_trained_weights()

if success:
    st.success("Pesos del modelo neuronal (`W1, b1, W2, b2`) cargados con éxito desde la carpeta local.")
    
    # Selector de entrada de imagen
    option = st.radio("Selecciona el método de entrada de imagen:", ("Cámara Web en Vivo", "Subir un archivo (.jpg/.png)"))
    
    img_file = None
    if option == "Cámara Web en Vivo":
        img_file = st.camera_input("Captura tu postura frente a la pantalla")
    else:
        img_file = st.file_uploader("Elige una foto...", type=["jpg", "jpeg", "png"])
        
    if img_file is not None:
        bytes_data = img_file.read()
        
        # Ejecutar preprocesamiento idéntico al de C
        img_original, img_sobel, x_vector = preprocess_image(bytes_data)
        
        # Mostrar imágenes de diagnóstico en la App
        col1, col2 = st.columns(2)
        with col1:
            st.image(img_original, channels="BGR", caption="Imagen Original Capturada")
        with col2:
            st.image(img_sobel, width=240, caption="Pipeline de Bordes Sobel (64x64)")
            
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
        st.markdown("---")
        st.subheader("Predicción del Modelo Neuronal:")
        
        if probabilidad >= 0.5:
            st.error(f"POSTURA ENCORVADA DETECTADA (Probabilidad: {probabilidad*100:.2f}%)")
            st.warning("Consejos de Ergonomía: Alinea tus hombros con tus orejas, mantén la espalda apoyada en el respaldo y levanta tu monitor.")
        else:
            st.success(f"POSTURA RECTA CORRECTA (Probabilidad de encorvamiento: {probabilidad*100:.2f}%)")
            st.info("Excellent! Mantienes una higiene postural idónea para largas jornadas de programación.")
            
else:
    st.warning("Asegúrate de haber ejecutado el programa `./train` de CUDA para generar los archivos de pesos binarios en la ruta correcta.")