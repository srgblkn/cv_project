# pages/facescanner.py
from __future__ import annotations

import io
import json
import zipfile
from dataclasses import dataclass
from typing import List, Tuple, Optional

import numpy as np
import streamlit as st
from PIL import Image, ImageFilter, ImageDraw

# Убедитесь, что ultralytics установлен в requirements.txt
try:
    from ultralytics import YOLO
except Exception:
    YOLO = None


# -----------------------------
# Конфигурация страницы
# -----------------------------
st.set_page_config(
    page_title="FaceScanner (YOLO) — маскировка лиц",
    page_icon="🕵️",
    layout="wide",
)

st.title("FaceScanner (YOLO) — детекция и маскировка лиц")
st.caption(
    "Загрузка нескольких файлов, детекция YOLO и маскировка области лица. "
    "Веса можно заменить без изменения кода."
)


# -----------------------------
# Вспомогательные структуры
# -----------------------------
@dataclass
class MaskConfig:
    mode: str  # "Blur" | "Pixelate" | "Solid"
    blur_radius: int = 12
    pixel_size: int = 12
    solid_color: Tuple[int, int, int] = (0, 0, 0)
    padding: float = 0.10  # расширение бокса (10%)


# -----------------------------
# Проверки окружения
# -----------------------------
def show_runtime_info():
    with st.sidebar.expander("Среда выполнения", expanded=True):
        st.write("**YOLO/Ultralytics**:", "OK" if YOLO is not None else "Не найден (проверь requirements)")
        try:
            import torch

            st.write("**PyTorch**:", torch.__version__)
            st.write("**CUDA доступна**:", bool(torch.cuda.is_available()))
            if torch.cuda.is_available():
                st.write("**GPU**:", torch.cuda.get_device_name(0))
        except Exception as e:
            st.write("**PyTorch**: недоступен")
            st.caption(f"Детали: {e}")


show_runtime_info()


# -----------------------------
# Загрузка модели (кэш)
# -----------------------------
@st.cache_resource(show_spinner=False)
def load_yolo_model(weights_path: str):
    if YOLO is None:
        raise RuntimeError(
            "Пакет ultralytics не доступен. Проверь requirements.txt и установку зависимостей."
        )
    return YOLO(weights_path)


# -----------------------------
# Обработка боксов и маскировка
# -----------------------------
def expand_box_xyxy(x1, y1, x2, y2, w, h, pad: float):
    bw = x2 - x1
    bh = y2 - y1
    x1n = max(0, int(round(x1 - bw * pad)))
    y1n = max(0, int(round(y1 - bh * pad)))
    x2n = min(w - 1, int(round(x2 + bw * pad)))
    y2n = min(h - 1, int(round(y2 + bh * pad)))
    if x2n <= x1n or y2n <= y1n:
        return None
    return x1n, y1n, x2n, y2n


def apply_mask_pil(img: Image.Image, boxes_xyxy: List[Tuple[int, int, int, int]], cfg: MaskConfig) -> Image.Image:
    """
    Маскируем области по списку боксов xyxy.
    """
    out = img.copy()
    w, h = out.size

    for (x1, y1, x2, y2) in boxes_xyxy:
        expanded = expand_box_xyxy(x1, y1, x2, y2, w, h, cfg.padding)
        if expanded is None:
            continue
        x1e, y1e, x2e, y2e = expanded

        roi = out.crop((x1e, y1e, x2e, y2e))

        if cfg.mode == "Blur":
            roi_masked = roi.filter(ImageFilter.GaussianBlur(radius=int(cfg.blur_radius)))
        elif cfg.mode == "Pixelate":
            ps = max(2, int(cfg.pixel_size))
            small = roi.resize((max(1, roi.size[0] // ps), max(1, roi.size[1] // ps)), resample=Image.NEAREST)
            roi_masked = small.resize(roi.size, resample=Image.NEAREST)
        else:  # Solid
            roi_masked = Image.new("RGB", roi.size, cfg.solid_color)

        out.paste(roi_masked, (x1e, y1e))
    return out


def draw_boxes_pil(img: Image.Image, boxes_xyxy: List[Tuple[int, int, int, int]]) -> Image.Image:
    out = img.copy()
    d = ImageDraw.Draw(out)
    for (x1, y1, x2, y2) in boxes_xyxy:
        d.rectangle([x1, y1, x2, y2], width=3, outline=(255, 0, 0))
    return out


def yolo_predict_boxes(
    model,
    img_rgb: np.ndarray,
    conf: float,
    iou: float,
    max_det: int,
) -> List[Tuple[int, int, int, int]]:
    """
    Возвращает боксы xyxy (int) для всех детекций.
    (Для face-модели обычно класс один, так что фильтр классов не нужен.)
    """
    # ultralytics ожидает np.uint8 HWC RGB
    results = model.predict(img_rgb, conf=conf, iou=iou, max_det=max_det, verbose=False)
    if not results:
        return []

    r0 = results[0]
    if r0.boxes is None or len(r0.boxes) == 0:
        return []

    xyxy = r0.boxes.xyxy.detach().cpu().numpy()
    boxes = []
    for x1, y1, x2, y2 in xyxy:
        boxes.append((int(round(x1)), int(round(y1)), int(round(x2)), int(round(y2))))
    return boxes


# -----------------------------
# Sidebar: настройки
# -----------------------------
st.sidebar.header("Настройки FaceScanner")

weights_path = st.sidebar.text_input(
    "Путь к весам YOLO (локально в репо)",
    value="models/face_yolo.pt",
    help="Рекомендуется хранить веса локально (не в git). Можно заменить файл — страница продолжит работать.",
)

conf_th = st.sidebar.slider("Confidence threshold", 0.05, 0.95, 0.25, 0.05)
iou_th = st.sidebar.slider("IoU threshold", 0.10, 0.90, 0.50, 0.05)
max_det = st.sidebar.number_input("Max detections per image", min_value=1, max_value=500, value=50, step=1)

st.sidebar.divider()
st.sidebar.subheader("Маскировка")

mask_mode = st.sidebar.selectbox("Режим маскировки", ["Blur", "Pixelate", "Solid"], index=0)
padding = st.sidebar.slider("Padding бокса (расширение)", 0.0, 0.5, 0.10, 0.02)

blur_radius = 12
pixel_size = 12
solid_color = (0, 0, 0)

if mask_mode == "Blur":
    blur_radius = st.sidebar.slider("Blur radius", 1, 40, 12, 1)
elif mask_mode == "Pixelate":
    pixel_size = st.sidebar.slider("Pixel size", 2, 40, 12, 1)
else:
    # базовые цвета без лишней сложности
    color_name = st.sidebar.selectbox("Цвет заливки", ["Black", "White", "Gray"], index=0)
    solid_color = {"Black": (0, 0, 0), "White": (255, 255, 255), "Gray": (120, 120, 120)}[color_name]

mask_cfg = MaskConfig(
    mode=mask_mode,
    blur_radius=blur_radius,
    pixel_size=pixel_size,
    solid_color=solid_color,
    padding=padding,
)


# -----------------------------
# Основной UI: загрузка файлов
# -----------------------------
left, right = st.columns([1.2, 1.0], gap="large")

with left:
    st.subheader("1) Загрузка изображений")
    uploads = st.file_uploader(
        "Загрузите изображения (можно несколько)",
        type=["png", "jpg", "jpeg", "bmp", "tif", "tiff"],
        accept_multiple_files=True,
    )

    run_btn = st.button("Запустить детекцию и маскировку", type="primary", use_container_width=True)


with right:
    st.subheader("2) Информация о модели (шаблон + опциональный отчёт)")
    st.caption("Здесь держим данные об обучении, качестве и рекомендациях (обязательное требование проекта).")

    report_file = st.file_uploader(
        "Если есть отчёт в JSON — загрузите сюда (необязательно)",
        type=["json"],
        accept_multiple_files=False,
        help="Например: epochs, train_size, val_size, mAP, PR-curve (ссылки/файлы), confusion matrix и т.п.",
        key="facescanner_report",
    )

    report = None
    if report_file is not None:
        try:
            report = json.loads(report_file.read().decode("utf-8"))
        except Exception as e:
            st.error(f"Не удалось прочитать JSON: {e}")

    with st.expander("Информация о модели / обучении / метриках", expanded=True):
        if report:
            st.json(report)
        else:
            c1, c2 = st.columns(2)
            with c1:
                st.markdown("**Обучение**")
                st.write("- Эпохи: TBD")
                st.write("- Размер train: TBD")
                st.write("- Размер val: TBD")
            with c2:
                st.markdown("**Метрики**")
                st.write("- mAP@0.5: TBD")
                st.write("- PR curve: TBD")
                st.write("- Confusion matrix: TBD")

            st.markdown("**Рекомендации**")
            st.write("- Подобрать conf/iou под задачу и данные.")
            st.write("- Для прототипа сохранять веса каждые 2–5 эпох.")
            st.write("- При ложных срабатываниях — дообучение на собственных примерах + баланс данных.")


# -----------------------------
# Исполнение: инференс по загрузкам
# -----------------------------
if run_btn:
    if not uploads:
        st.warning("Сначала загрузите хотя бы один файл.")
        st.stop()

    if YOLO is None:
        st.error("Ultralytics не установлен. Проверьте requirements.txt и установку зависимостей.")
        st.stop()

    # Загружаем модель
    with st.spinner("Загружаю YOLO модель..."):
        try:
            model = load_yolo_model(weights_path)
        except Exception as e:
            st.error(
                "Не удалось загрузить веса YOLO.\n\n"
                f"Путь: `{weights_path}`\n\n"
                f"Ошибка: {e}\n\n"
                "Проверьте, что файл весов существует (например, положите его в папку `models/`)."
            )
            st.stop()

    st.success("Модель загружена. Обрабатываю изображения...")

    results_for_zip = []  # (filename, bytes)
    preview_rows = []

    prog = st.progress(0)
    for idx, up in enumerate(uploads, start=1):
        try:
            img = Image.open(up).convert("RGB")
            img_np = np.array(img)

            boxes = yolo_predict_boxes(
                model=model,
                img_rgb=img_np,
                conf=float(conf_th),
                iou=float(iou_th),
                max_det=int(max_det),
            )

            masked = apply_mask_pil(img, boxes, mask_cfg)
            boxed = draw_boxes_pil(img, boxes)  # исходник с боксами

            # сохраняем в память
            buf_masked = io.BytesIO()
            masked.save(buf_masked, format="PNG")
            buf_masked.seek(0)

            out_name = f"{up.name.rsplit('.', 1)[0]}_masked.png"
            results_for_zip.append((out_name, buf_masked.getvalue()))

            # для превью
            preview_rows.append((up.name, img, boxed, masked, len(boxes)))

        except Exception as e:
            st.error(f"Ошибка обработки файла {up.name}: {e}")

        prog.progress(int(idx / len(uploads) * 100))

    prog.empty()

    # Показываем превью
    st.divider()
    st.subheader("Результаты")

    for name, orig, boxed, masked, n_boxes in preview_rows:
        with st.expander(f"{name} — детекций: {n_boxes}", expanded=False):
            c1, c2, c3 = st.columns(3)
            with c1:
                st.markdown("**Оригинал**")
                st.image(orig, use_container_width=True)
            with c2:
                st.markdown("**Детекции (боксы)**")
                st.image(boxed, use_container_width=True)
            with c3:
                st.markdown("**Маскировано**")
                st.image(masked, use_container_width=True)

    # ZIP для скачивания
    st.divider()
    st.subheader("Скачать результаты")
    zip_buf = io.BytesIO()
    with zipfile.ZipFile(zip_buf, "w", compression=zipfile.ZIP_DEFLATED) as zf:
        for fname, fbytes in results_for_zip:
            zf.writestr(fname, fbytes)
    zip_buf.seek(0)

    st.download_button(
        label="Скачать ZIP с маскированными изображениями",
        data=zip_buf,
        file_name="facescanner_results.zip",
        mime="application/zip",
        use_container_width=True,
    )


# -----------------------------
# Подсказка по весам
# -----------------------------
with st.expander("Где взять веса и как их подключить", expanded=False):
    st.markdown(
        """
**Практика для проекта:**
- Храните веса локально в `models/` (обычно не коммитим в git).
- Для быстрого прототипа: обучили 1–3 эпохи, сохранили `models/face_yolo.pt`, проверили страницу.
- Потом просто заменяете файл весов на более качественный — приложение продолжит работать.

**Важно:**
- Если вы используете кастомные веса (не COCO), путь в сайдбаре должен указывать на существующий файл.
        """
    )
