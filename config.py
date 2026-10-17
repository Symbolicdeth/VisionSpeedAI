# ==================================
# SYMBOLICDETH VISIONSPEED AI
# Configuration
# ==================================

# Modelo YOLO guardado en la carpeta principal del proyecto.
MODEL_NAME = "yolo26n.pt"

# Umbral de confianza de las detecciones.
CONFIDENCE = 0.35

# Resolución de entrada para la detección.
IMAGE_SIZE = 640

# Clases COCO del modelo YOLO. El modo warehouse usa objetos genéricos;
# para reconocer productos propios se necesitará un modelo entrenado para ellos.
MODES = {
    "vehicles": {
        "name": "Vehículos",
        "classes": [2, 3, 5, 7],
    },

    "people": {
        "name": "Personas",
        "classes": [0],
    },

    "warehouse": {
        "name": "Almacén (objetos genéricos)",
        "classes": [24, 26, 28, 39, 41, 56, 57, 60, 63, 67, 73],
    },
}
