from pathlib import Path
from datetime import datetime
import csv
import tkinter as tk
from tkinter import filedialog, messagebox, ttk

import cv2
from ultralytics import YOLO

from config import CONFIDENCE, MODES, MODEL_NAME


PROJECT_DIR = Path(__file__).resolve().parent
MODEL_PATH = PROJECT_DIR / MODEL_NAME
OUTPUT_DIR = PROJECT_DIR / "outputs"
WINDOW_NAME = "Symbolicdeth VisionSpeed AI"
LABEL_TRANSLATIONS = {
    "car": "Auto",
    "motorcycle": "Motocicleta",
    "bus": "Autobús",
    "truck": "Camión",
    "person": "Persona",
    "backpack": "Mochila",
    "handbag": "Bolso",
    "suitcase": "Maleta",
    "bottle": "Botella",
    "cup": "Taza",
    "chair": "Silla",
    "couch": "Sofá",
    "dining table": "Mesa",
    "laptop": "Portátil",
    "cell phone": "Teléfono",
    "book": "Libro",
}


class VisionSpeedApp:
    def __init__(self, root):
        self.root = root
        self.root.title("Symbolicdeth VisionSpeed AI")
        self.root.geometry("620x410")
        self.root.minsize(540, 380)

        self.video_path = tk.StringVar()
        self.mode_names = {mode["name"]: key for key, mode in MODES.items()}
        self.resolution_options = {
            "Más rápido (480)": 480,
            "Equilibrado (640)": 640,
            "Más detalle (800)": 800,
        }
        self.resolution_var = tk.StringVar(value="Equilibrado (640)")
        self.save_video = tk.BooleanVar(value=False)
        self.status = tk.StringVar(value="Selecciona un video para comenzar.")

        self._build_ui()

    def _build_ui(self):
        container = ttk.Frame(self.root, padding=24)
        container.pack(fill="both", expand=True)

        ttk.Label(
            container,
            text="Symbolicdeth VisionSpeed AI",
            font=("Segoe UI", 18, "bold"),
        ).pack(anchor="w")
        ttk.Label(
            container,
            text="Analiza un video con detección y seguimiento de objetos.",
        ).pack(anchor="w", pady=(4, 18))

        video_row = ttk.Frame(container)
        video_row.pack(fill="x")
        ttk.Entry(video_row, textvariable=self.video_path, state="readonly").pack(
            side="left", fill="x", expand=True
        )
        ttk.Button(video_row, text="Elegir video…", command=self.choose_video).pack(
            side="left", padx=(10, 0)
        )

        ttk.Label(container, text="Modo de detección").pack(anchor="w", pady=(18, 5))
        self.mode_combo = ttk.Combobox(
            container,
            state="readonly",
            values=list(self.mode_names.keys()),
        )
        self.mode_combo.pack(fill="x")
        self.mode_combo.bind("<<ComboboxSelected>>", self._update_mode_note)
        self.mode_combo.set(MODES["vehicles"]["name"])

        self.mode_note = ttk.Label(container, text="", wraplength=560)
        self.mode_note.pack(anchor="w", pady=(6, 0))
        self._update_mode_note()

        ttk.Label(container, text="Calidad y velocidad").pack(anchor="w", pady=(14, 5))
        self.resolution_combo = ttk.Combobox(
            container,
            state="readonly",
            textvariable=self.resolution_var,
            values=list(self.resolution_options.keys()),
        )
        self.resolution_combo.pack(fill="x")

        ttk.Checkbutton(
            container,
            text="Guardar una copia anotada en outputs/ (puede tardar más)",
            variable=self.save_video,
        ).pack(anchor="w", pady=(16, 0))
        ttk.Label(
            container,
            text="El reporte CSV de detecciones se guarda automáticamente en outputs/.",
            foreground="#555555",
        ).pack(anchor="w", pady=(7, 0))

        controls = ttk.Frame(container)
        controls.pack(fill="x", pady=(20, 0))
        self.start_button = ttk.Button(
            controls, text="Iniciar análisis", command=self.start_analysis
        )
        self.start_button.pack(side="left")
        ttk.Label(controls, textvariable=self.status).pack(side="left", padx=12)

        ttk.Label(
            container,
            text="Durante el análisis, pulsa Q o Esc en la ventana del video para detenerlo.",
            foreground="#555555",
        ).pack(anchor="w", pady=(14, 0))

    def _update_mode_note(self, _event=None):
        mode_key = self.mode_names.get(self.mode_combo.get(), "vehicles")
        if mode_key == "warehouse":
            self.mode_note.configure(
                text=(
                    "Este modo detecta objetos genéricos del modelo (por ejemplo, "
                    "mochilas, botellas y sillas); todavía no reconoce productos "
                    "específicos del almacén."
                )
            )
        else:
            self.mode_note.configure(text="")

    def choose_video(self):
        path = filedialog.askopenfilename(
            title="Selecciona un video",
            initialdir=str(PROJECT_DIR / "videos"),
            filetypes=[
                ("Videos", "*.mp4 *.avi *.mov *.mkv *.wmv"),
                ("Todos los archivos", "*.*"),
            ],
        )
        if path:
            self.video_path.set(path)
            self.status.set(Path(path).name)

    def start_analysis(self):
        video_path = Path(self.video_path.get())
        if not video_path.is_file():
            messagebox.showwarning("Falta el video", "Elige un archivo de video válido.")
            return
        if not MODEL_PATH.is_file():
            messagebox.showerror(
                "No se encuentra el modelo",
                f"No encontré el modelo aquí:\n{MODEL_PATH}\n\n"
                "Coloca yolo26n.pt en la carpeta principal del proyecto.",
            )
            return

        self.start_button.configure(state="disabled")
        self.status.set("Cargando modelo…")
        self.root.update_idletasks()
        try:
            mode_key = self.mode_names.get(self.mode_combo.get(), "vehicles")
            image_size = self.resolution_options[self.resolution_var.get()]
            self._process_video(video_path, mode_key, image_size)
        except Exception as exc:
            messagebox.showerror("Error durante el análisis", str(exc))
        finally:
            self.start_button.configure(state="normal")
            self.status.set("Listo para otro video.")

    def _process_video(self, video_path, mode_key, image_size):
        mode = MODES[mode_key]
        OUTPUT_DIR.mkdir(exist_ok=True)
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        report_path = OUTPUT_DIR / f"{video_path.stem}_{mode_key}_{timestamp}_report.csv"
        self.status.set("Abriendo video…")
        self.root.update_idletasks()

        capture = cv2.VideoCapture(str(video_path))
        if not capture.isOpened():
            capture.release()
            raise RuntimeError("No se pudo abrir el video. Comprueba que el archivo sea válido.")

        fps = capture.get(cv2.CAP_PROP_FPS)
        width = int(capture.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(capture.get(cv2.CAP_PROP_FRAME_HEIGHT))
        if not fps or fps <= 0:
            fps = 30.0

        output_path = None
        writer = None
        if self.save_video.get():
            output_path = OUTPUT_DIR / (
                f"{video_path.stem}_{mode_key}_{timestamp}_annotated.mp4"
            )
            writer = cv2.VideoWriter(
                str(output_path),
                cv2.VideoWriter_fourcc(*"mp4v"),
                fps,
                (width, height),
            )
            if not writer.isOpened():
                capture.release()
                writer.release()
                raise RuntimeError(f"No se pudo crear el video de salida:\n{output_path}")

        window_created = False
        unique_ids = set()
        class_observations = {}
        class_track_ids = {}
        frame_number = 0
        end_reason = "video_end"
        model = None
        self.status.set("Cargando modelo…")
        self.root.update_idletasks()

        try:
            model = YOLO(str(MODEL_PATH))
            for class_id in mode["classes"]:
                class_name = self._get_class_name(model.names, class_id)
                class_observations[class_name] = 0
                class_track_ids[class_name] = set()

            while True:
                success, frame = capture.read()
                if not success:
                    break

                results = model.track(
                    frame,
                    persist=True,
                    tracker="bytetrack.yaml",
                    classes=mode["classes"],
                    conf=CONFIDENCE,
                    imgsz=image_size,
                    verbose=False,
                )
                result = results[0]
                annotated = result.plot()
                counts, current_ids, ids_by_class = self._count_detections(
                    result, model.names
                )
                unique_ids.update(current_ids)
                for class_name, count in counts.items():
                    class_observations[class_name] += count
                for class_name, track_ids in ids_by_class.items():
                    class_track_ids[class_name].update(track_ids)
                frame_number += 1

                total = sum(counts.values())
                cv2.putText(
                    annotated,
                    f"{mode['name']}: {total} en cuadro | {len(unique_ids)} IDs unicos",
                    (18, 32),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.7,
                    (0, 255, 0),
                    2,
                )
                summary = " | ".join(
                    f"{name}: {count}" for name, count in counts.items() if count
                ) or "Sin detecciones"
                cv2.putText(
                    annotated,
                    summary[:110],
                    (18, 62),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.55,
                    (0, 255, 0),
                    2,
                )

                if writer is not None:
                    writer.write(annotated)

                if not window_created:
                    cv2.namedWindow(WINDOW_NAME, cv2.WINDOW_NORMAL)
                    window_created = True
                display = self._fit_frame(annotated, 1400, 900)
                cv2.imshow(WINDOW_NAME, display)
                if frame_number % 15 == 0:
                    self.status.set(f"Procesando: fotograma {frame_number}")
                    self.root.update_idletasks()

                key = cv2.waitKey(1) & 0xFF
                if key in (ord("q"), 27):
                    end_reason = "stopped_by_user"
                    break
        finally:
            capture.release()
            if writer is not None:
                writer.release()
            if window_created:
                cv2.destroyWindow(WINDOW_NAME)

        if frame_number == 0:
            raise RuntimeError("No se pudieron leer fotogramas del video.")

        self._write_report(
            report_path,
            video_path,
            mode["name"],
            image_size,
            frame_number,
            class_observations,
            class_track_ids,
            end_reason,
        )

        message = (
            f"Análisis terminado. IDs seguidos: {len(unique_ids)}.\n"
            f"Reporte CSV guardado en:\n{report_path}"
        )
        if output_path is not None:
            message += f"\nVideo guardado en:\n{output_path}"
        messagebox.showinfo("Análisis terminado", message)

    @staticmethod
    def _get_class_name(names, class_id):
        name = names.get(class_id, str(class_id)) if isinstance(names, dict) else names[class_id]
        return LABEL_TRANSLATIONS.get(name, name)

    @staticmethod
    def _write_report(
        report_path,
        video_path,
        mode_name,
        image_size,
        frame_number,
        class_observations,
        class_track_ids,
        end_reason,
    ):
        with report_path.open("w", newline="", encoding="utf-8-sig") as report_file:
            writer = csv.DictWriter(
                report_file,
                fieldnames=[
                    "video",
                    "modo",
                    "clase",
                    "fotogramas_analizados",
                    "resolucion_modelo",
                    "detecciones_en_fotogramas",
                    "objetos_unicos_seguidos",
                    "estado_analisis",
                ],
            )
            writer.writeheader()
            for class_name, observations in class_observations.items():
                writer.writerow(
                    {
                        "video": video_path.name,
                        "modo": mode_name,
                        "clase": class_name,
                        "fotogramas_analizados": frame_number,
                        "resolucion_modelo": image_size,
                        "detecciones_en_fotogramas": observations,
                        "objetos_unicos_seguidos": len(class_track_ids[class_name]),
                        "estado_analisis": end_reason,
                    }
                )

    @staticmethod
    def _count_detections(result, names):
        counts = {}
        tracked_ids = set()
        ids_by_class = {}
        boxes = result.boxes
        if boxes is None or boxes.cls is None:
            return counts, tracked_ids, ids_by_class

        class_ids = boxes.cls.int().tolist()
        ids = boxes.id.int().tolist() if boxes.id is not None else [None] * len(class_ids)
        for class_id, track_id in zip(class_ids, ids):
            label = VisionSpeedApp._get_class_name(names, class_id)
            counts[label] = counts.get(label, 0) + 1
            ids_by_class.setdefault(label, set())
            if track_id is not None:
                tracked_ids.add(int(track_id))
                ids_by_class[label].add(int(track_id))
        return counts, tracked_ids, ids_by_class

    @staticmethod
    def _fit_frame(frame, max_width, max_height):
        height, width = frame.shape[:2]
        scale = min(max_width / width, max_height / height, 1.0)
        if scale == 1.0:
            return frame
        return cv2.resize(
            frame,
            (int(width * scale), int(height * scale)),
            interpolation=cv2.INTER_AREA,
        )


def main():
    root = tk.Tk()
    VisionSpeedApp(root)
    root.mainloop()


if __name__ == "__main__":
    main()
