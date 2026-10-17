# Symbolicdeth VisionSpeed AI

Aplicación de escritorio para elegir un video, detectar personas, vehículos u objetos genéricos y guardar una copia anotada.

## Requisitos

- Windows 10/11
- Python 3.10 o superior
- `yolo26n.pt` en la carpeta principal del proyecto

## Preparar el entorno

Abre PowerShell en esta carpeta y ejecuta:

```powershell
py -m venv .venv-desktop
.\.venv-desktop\Scripts\Activate.ps1
python -m pip install --upgrade pip
pip install -r requirements.txt
```

El proyecto copiado puede incluir un entorno `.venv` enlazado a una ruta antigua. Crea uno nuevo en `.venv-desktop` y deja el anterior sin usar. Si PowerShell bloquea la activación, ejecuta los comandos directamente con:

```powershell
.\.venv-desktop\Scripts\python.exe -m pip install --upgrade pip
.\.venv-desktop\Scripts\python.exe -m pip install -r requirements.txt
```

## Iniciar

```powershell
python main.py
```

Sin activar el entorno, inicia con `\.venv-desktop\Scripts\python.exe main.py`.

Elige un video, selecciona un modo y pulsa **Iniciar análisis**. En **Calidad y velocidad**, 480 prioriza velocidad, 640 ofrece un equilibrio y 800 busca más detalle. Pulsa `Q` o `Esc` en la ventana del video para detenerlo. Si activas la opción para guardar, el resultado anotado se guarda en `outputs/`; guardarlo puede reducir la velocidad del análisis.

Cada análisis genera automáticamente un CSV de resumen en `outputs/`, con una fila por clase: número de detecciones acumuladas en los fotogramas, objetos únicos seguidos, fotogramas analizados, resolución y si el análisis terminó o se detuvo manualmente. El conteo acumulado cuenta la misma detección en varios fotogramas; `objetos_unicos_seguidos` es la estimación del tracker.

El análisis ya no fuerza el uso de CPU: Ultralytics selecciona el dispositivo disponible. En un equipo con PyTorch y CUDA correctamente instalados podrá usar una GPU NVIDIA; si no, usará CPU.

## Modos disponibles

- **Vehículos:** autos, motocicletas, autobuses y camiones.
- **Personas:** personas detectadas por el modelo.
- **Almacén (objetos genéricos):** clases genéricas del modelo COCO, como mochilas, botellas, sillas y teléfonos. No reconoce productos propios del almacén; eso requerirá un conjunto de imágenes y un modelo entrenado para esos productos.

El seguimiento usa ByteTrack. El conteo "en cuadro" corresponde al fotograma actual; los IDs únicos son los objetos que el tracker pudo seguir durante el análisis y no garantizan un conteo perfecto de personas u objetos reales.
