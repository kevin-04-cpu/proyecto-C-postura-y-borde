import os
import shutil
from pathlib import Path
from sklearn.model_selection import train_test_split

# ==========================================
# CONFIGURACIÓN DE RUTAS (Ajustada a tus carpetas reales)
# ==========================================
RUTA_ORIGEN = Path("images")             # Tu carpeta con las fotos
RUTA_DESTINO = Path("dataset/processed") # Aquí se creará la división limpia

# Nombre EXACTO de tus carpetas en el disco duro y su etiqueta ID
CLASES = {
    "postura_recta": 0, 
    "postura_encorvada": 1
}

# ==========================================
# 1. RECOLECCIÓN DE RUTAS Y ETIQUETAS
# ==========================================
imagenes = []
etiquetas = []

for nombre_clase, etiqueta_id in CLASES.items():
    carpeta_clase = RUTA_ORIGEN / nombre_clase
    if not carpeta_clase.exists():
        print(f"Error: No se encontró la carpeta {carpeta_clase.resolve()}")
        exit()
        
    for archivo in carpeta_clase.glob("*"):
        if archivo.suffix.lower() in ['.jpg', '.jpeg', '.png']:
            imagenes.append(str(archivo))
            etiquetas.append(etiqueta_id)

print(f"Total de imágenes encontradas: {len(imagenes)}")

if len(imagenes) == 0:
    print("No se encontraron imágenes. Verifica el formato (.jpg, .jpeg, .png).")
    exit()

# ==========================================
# 2. DIVISIÓN ESTRATIFICADA (70% Train, 15% Val, 15% Test)
# ==========================================
X_train, X_resto, y_train, y_resto = train_test_split(
    imagenes, etiquetas, test_size=0.30, random_state=42, stratify=etiquetas
)

X_val, X_test, y_val, y_test = train_test_split(
    X_resto, y_resto, test_size=0.50, random_state=42, stratify=y_resto
)

divisiones = {
    "train": (X_train, y_train),
    "val": (X_val, y_val),
    "test": (X_test, y_test)
}

# ==========================================
# 3. CREACIÓN DE CARPETAS Y COPIA DE ARCHIVOS
# ==========================================
# Asegurar que exista el directorio base de destino
RUTA_DESTINO.mkdir(parents=True, exist_ok=True)

for nombre_div, (rutas_fotos, labels) in divisiones.items():
    # Crear archivo .txt para OpenMP
    txt_pipeline = open(RUTA_DESTINO / f"{nombre_div}.txt", "w")
    
    print(f"\nProcesando división: {nombre_div.upper()} ({len(rutas_fotos)} imágenes)...")
    
    for ruta_foto, label in zip(rutas_fotos, labels):
        # Usamos los mismos nombres de carpeta originales para el destino
        nombre_subcarpeta = "postura_recta" if label == 0 else "postura_encorvada"
        carpeta_final = RUTA_DESTINO / nombre_div / nombre_subcarpeta
        
        carpeta_final.mkdir(parents=True, exist_ok=True)
        
        nombre_archivo = Path(ruta_foto).name
        ruta_destino_foto = carpeta_final / nombre_archivo
        
        # Copiar el archivo físicamente
        shutil.copy(ruta_foto, ruta_destino_foto)
        
        # Guardar ruta relativa simplificada para el lector en C
        ruta_relativa_c = f"{nombre_div}/{nombre_subcarpeta}/{nombre_archivo}"
        txt_pipeline.write(f"{ruta_relativa_c} {label}\n")
        
    txt_pipeline.close()

print("\n¡Proceso completado con éxito!")
print(f"Estructura creada en: {RUTA_DESTINO.resolve()}")
