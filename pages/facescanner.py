# pages/facescanner.py
from __future__ import annotations

import base64
import io
import json
import zipfile
from dataclasses import dataclass
from pathlib import Path
from typing import List, Tuple

import numpy as np
import pandas as pd
import streamlit as st
from PIL import Image, ImageFilter, ImageDraw

# ultralytics YOLO
try:
    from ultralytics import YOLO
except Exception:
    YOLO = None

# YAML (опционально)
try:
    import yaml  # pyyaml
except Exception:
    yaml = None


# =============================
# Paths (relative to this file)
# =============================
THIS_DIR = Path(__file__).resolve().parent
FB_DIR = THIS_DIR / "facebook"

DEFAULT_WEIGHTS = FB_DIR / "best-13.pt"
DEFAULT_ARGS_YAML = FB_DIR / "args.yaml"
DEFAULT_RESULTS_CSV = FB_DIR / "results.csv"
DEFAULT_BG_JPG = FB_DIR / "background.jpg"


# =============================
# Page config
# =============================
st.set_page_config(
    page_title="FaceScanner — маскировка лиц",
    page_icon="🕵️",
    layout="wide",
)


# =============================
# Styling (background + opaque cards)
# =============================
def _apply_background_and_theme(bg_path: Path):
    """
    Ставит фон и повышает контраст/читабельность:
    - фон: jpg
    - сайдбар и ключевые контейнеры: непрозрачные
    """
    if bg_path.exists():
        b64 = base64.b64encode(bg_path.read_bytes()).decode("utf-8")
        bg_css = f"""
        .stApp {{
            background-image: url("data:image/jpeg;base64,{b64}");
            background-size: cover;
            background-position: center;
            background-attachment: fixed;
        }}
        """
    else:
        bg_css = ""

    css = f"""
    <style>
    {bg_css}

    /* Общий контраст текста */
    .stApp, .stMarkdown, .stText, .stCaption, .stWrite {{
        color: #F8FAFC;
    }}

    /* Header прозрачный/минимальный */
    header[data-testid="stHeader"] {{
        background: rgba(0,0,0,0);
    }}

    /* Sidebar: тёмный непрозрачный */
    section[data-testid="stSidebar"] {{
        background: #0B1220;
        border-right: 1px solid rgba(255,255,255,0.08);
    }}
    section[data-testid="stSidebar"] * {{
        color: #F8FAFC !important;
    }}

    /* "Карточки" (мы будем использовать HTML-блоки) */
    .opaque-card {{
        background: #0B1220;
        border: 1px solid rgba(255,255,255,0.10);
        border-radius: 18px;
        padding: 16px 16px 14px 16px;
        box-shadow: 0 10px 24px rgba(0,0,0,0.35);
        margin-bottom: 14px;
    }}
    .opaque-card h3, .opaque-card h4, .opaque-card p {{
        color: #F8FAFC;
        margin: 0;
    }}
    .opaque-card .muted {{
        color: rgba(248,250,252,0.80);
        margin-top: 6px;
    }}

    /* Expander: непрозрачный */
    div[data-testid="stExpander"] > details {{
        background: #0B1220;
        border: 1px solid rgba(255,255,255,0.10);
        border-radius: 18px;
        padding: 10px 12px;
        box-shadow: 0 10px 24px rgba(0,0,0,0.25);
    }}
    div[data-testid="stExpander"] summary {{
        color: #F8FAFC !important;
        font-weight: 600;
    }}

    /* File uploader and inputs: непрозрачные */
    div[data-testid="stFileUploader"] section {{
        background: #0B1220;
        border: 1px solid rgba(255,255,255,0.10);
        border-radius: 18px;
        padding: 10px;
    }}
    div[data-testid="stNumberInput"] div,
    div[data-testid="stSlider"] div {{
        background: transparent;
    }}

    /* Buttons */
    .stButton > button {{
        border-radius: 14px;
        border: 1px solid rgba(255,255,255,0.12);
    }}

    /* Links (если будут) */
    a {{
        color: #93C5FD !important;
    }}
    </style>
    """
    st.markdown(css, unsafe_allow_html=True)


_apply_background_and_theme(DEFAULT_BG_JPG)


# =============================
# Data structures
# =============================
@dataclass
class MaskConfig:
    mode: str  # "Blur" | "Pixelate" | "Solid"
    blur_radius: int = 12
    pixel_size: int = 12
    solid_color: Tuple[int, int, int] = (0, 0, 0)
    padding: float = 0.10  # расширение бокса (10%)


# =============================
# Helpers
# =============================
@st.cache_resource(show_spinner=False)
def load_yolo_model(weights_path: str):
    if YOLO is None:
        raise RuntimeError("ultralytics не установлен. Проверьте requirements.txt.")
    return YOLO(weights_path)


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
            small = roi.resize(
                (max(1, roi.size[0] // ps), max(1, roi.size[1] // ps)),
                resample=Image.NEAREST,
            )
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


def yolo_predict_boxes(model, img_rgb: np.ndarray, conf: float, iou: float, max_det: int) -> List[Tuple[int, int, int, int]]:
    results = model.predict(img_rgb, conf=conf, iou=iou, max_det=max_det, verbose=False)
    if not results:
        return []
    r0 = results[0]
    if r0.boxes is None or len(r0.boxes) == 0:
        return []
    xyxy = r0.boxes.xyxy.detach().cpu().numpy()
    return [(int(round(x1)), int(round(y1)), int(round(x2)), int(round(y2))) for x1, y1, x2, y2 in xyxy]


def opaque_card(title: str, text: str):
    st.markdown(
        f"""
        <div class="opaque-card">
          <h3>{title}</h3>
          <p class="muted">{text}</p>
        </div>
        """,
        unsafe_allow_html=True,
    )


def safe_switch_page(target: str):
    if hasattr(st, "switch_page"):
        try:
            st.switch_page(target)
        except Exception:
            st.info("Навигация недоступна в текущей среде. Используйте меню слева.")
    else:
        st.info("Навигация недоступна в текущей версии Streamlit. Используйте меню слева.")


# =============================
# Header
# =============================
st.markdown(
    """
    <div class="opaque-card">
      <h3>FaceScanner</h3>
      <p class="muted">Маскировка лиц на изображениях: пакетная загрузка, детекция YOLO, скачивание результата одним архивом.</p>
    </div>
    """,
    unsafe_allow_html=True,
)

top_left, top_right = st.columns([1, 1], gap="large")
with top_left:
    if st.button("← На главную", use_container_width=True):
        safe_switch_page("app.py")
with top_right:
    if DEFAULT_BG_JPG.exists():
        st.download_button(
            "Скачать фон (JPG)",
            data=DEFAULT_BG_JPG.read_bytes(),
            file_name=DEFAULT_BG_JPG.name,
            mime="image/jpeg",
            use_container_width=True,
        )
    else:
        st.caption(f"Фон не найден: `{DEFAULT_BG_JPG.as_posix()}`")


# =============================
# Sidebar: settings & artifacts
# =============================
st.sidebar.markdown("## Настройки")
st.sidebar.caption("Параметры инференса и маскировки.")

# Веса: по умолчанию берем из pages/facebook/best-13.pt
weights_path = st.sidebar.text_input(
    "Веса YOLO (pt)",
    value=DEFAULT_WEIGHTS.as_posix(),
    help="Файл должен существовать в репозитории (например, pages/facebook/best-13.pt).",
)

conf_th = st.sidebar.slider("Confidence threshold", 0.05, 0.95, 0.25, 0.05)
iou_th = st.sidebar.slider("IoU threshold", 0.10, 0.90, 0.50, 0.05)
max_det = st.sidebar.number_input("Max detections per image", min_value=1, max_value=500, value=50, step=1)

st.sidebar.divider()
st.sidebar.markdown("### Маскировка")
mask_mode = st.sidebar.selectbox("Режим", ["Blur", "Pixelate", "Solid"], index=0)
padding = st.sidebar.slider("Padding бокса (расширение)", 0.0, 0.5, 0.10, 0.02)

blur_radius = 12
pixel_size = 12
solid_color = (0, 0, 0)

if mask_mode == "Blur":
    blur_radius = st.sidebar.slider("Blur radius", 1, 40, 12, 1)
elif mask_mode == "Pixelate":
    pixel_size = st.sidebar.slider("Pixel size", 2, 40, 12, 1)
else:
    color_name = st.sidebar.selectbox("Цвет заливки", ["Black", "White", "Gray"], index=0)
    solid_color = {"Black": (0, 0, 0), "White": (255, 255, 255), "Gray": (120, 120, 120)}[color_name]

mask_cfg = MaskConfig(
    mode=mask_mode,
    blur_radius=blur_radius,
    pixel_size=pixel_size,
    solid_color=solid_color,
    padding=padding,
)

st.sidebar.divider()
st.sidebar.markdown("## Артефакты обучения")
st.sidebar.caption("Автоподхват из pages/facebook/")

st.sidebar.write(f"• args.yaml: `{DEFAULT_ARGS_YAML.name}`", "✅" if DEFAULT_ARGS_YAML.exists() else "❌")
st.sidebar.write(f"• results.csv: `{DEFAULT_RESULTS_CSV.name}`", "✅" if DEFAULT_RESULTS_CSV.exists() else "❌")
st.sidebar.write(f"• weights: `{Path(weights_path).name}`", "✅" if Path(weights_path).exists() else "❌")


# =============================
# Main: upload & run
# =============================
left, right = st.columns([1.25, 1.0], gap="large")

with left:
    opaque_card("Загрузка изображений", "Поддерживается пакетная загрузка (несколько файлов за раз).")

    uploads = st.file_uploader(
        "Файлы изображений",
        type=["png", "jpg", "jpeg", "bmp", "tif", "tiff"],
        accept_multiple_files=True,
        label_visibility="collapsed",
    )

    run_btn = st.button("Запустить обработку", type="primary", use_container_width=True)

with right:
    opaque_card("Информация о модели", "Метрики и параметры обучения подгружаются из files: args.yaml и results.csv.")

    with st.expander("Параметры обучения (args.yaml)", expanded=True):
        if DEFAULT_ARGS_YAML.exists():
            if yaml is None:
                st.warning("PyYAML не установлен. Добавьте `pyyaml>=6.0` в requirements.txt.")
                st.code(DEFAULT_ARGS_YAML.read_text(encoding="utf-8", errors="ignore"))
            else:
                try:
                    args_data = yaml.safe_load(DEFAULT_ARGS_YAML.read_text(encoding="utf-8", errors="ignore"))
                    st.json(args_data)
                except Exception as e:
                    st.error(f"Не удалось прочитать args.yaml: {e}")
        else:
            st.info(f"Файл не найден: `{DEFAULT_ARGS_YAML.as_posix()}`")

    with st.expander("Качество по эпохам (results.csv)", expanded=True):
        if DEFAULT_RESULTS_CSV.exists():
            try:
                df = pd.read_csv(DEFAULT_RESULTS_CSV)
                st.caption("Сводка по обучению (лог Ultralytics).")

                # Найдем полезные колонки “как получится” (Ultralytics иногда меняет имена)
                cols = {c.lower(): c for c in df.columns}
                epoch_col = cols.get("epoch", None)

                # Кандидаты на метрику “лучше всего”
                map5095 = None
                map50 = None
                for c in df.columns:
                    cl = c.lower()
                    if "map50-95" in cl or "map50_95" in cl or "map50-95(b)" in cl:
                        map5095 = c
                    if "map50" in cl and map5095 is None:
                        map50 = c

                score_col = map5095 or map50
                if epoch_col and score_col:
                    best_idx = int(df[score_col].astype(float).idxmax())
                    best_epoch = int(df.loc[best_idx, epoch_col])
                    best_score = float(df.loc[best_idx, score_col])
                    st.success(f"Лучшая эпоха по `{score_col}`: epoch={best_epoch}, score={best_score:.4f}")

                # Покажем ключевые линии, если есть
                plot_candidates = []
                for key in ["precision", "recall", "map50", "map50-95", "box_loss", "cls_loss", "dfl_loss"]:
                    for c in df.columns:
                        if key in c.lower():
                            plot_candidates.append(c)

                plot_candidates = list(dict.fromkeys(plot_candidates))[:6]  # до 6 графиков

                if epoch_col and plot_candidates:
                    import matplotlib.pyplot as plt

                    for c in plot_candidates:
                        fig = plt.figure()
                        plt.plot(df[epoch_col], df[c])
                        plt.xlabel("epoch")
                        plt.ylabel(c)
                        plt.title(c)
                        st.pyplot(fig, clear_figure=True)
                else:
                    st.dataframe(df.tail(20), use_container_width=True)

            except Exception as e:
                st.error(f"Не удалось прочитать results.csv: {e}")
        else:
            st.info(f"Файл не найден: `{DEFAULT_RESULTS_CSV.as_posix()}`")


# =============================
# Run inference
# =============================
if run_btn:
    if not uploads:
        st.warning("Загрузите хотя бы один файл.")
        st.stop()

    if YOLO is None:
        st.error("ultralytics не установлен. Проверьте requirements.txt.")
        st.stop()

    weights_file = Path(weights_path)
    if not weights_file.exists():
        st.error(
            "Файл весов не найден.\n\n"
            f"Путь: `{weights_file.as_posix()}`\n\n"
            "Проверьте, что `best-13.pt` лежит в `pages/facebook/` и закоммичен/задеплоен."
        )
        st.stop()

    with st.spinner("Загружаю модель..."):
        model = load_yolo_model(weights_file.as_posix())

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
            boxed = draw_boxes_pil(img, boxes)

            buf_masked = io.BytesIO()
            masked.save(buf_masked, format="PNG")
            buf_masked.seek(0)

            out_name = f"{Path(up.name).stem}_masked.png"
            results_for_zip.append((out_name, buf_masked.getvalue()))

            preview_rows.append((up.name, img, boxed, masked, len(boxes)))

        except Exception as e:
            st.error(f"Ошибка обработки файла {up.name}: {e}")

        prog.progress(int(idx / len(uploads) * 100))

    prog.empty()

    st.markdown("### Результаты")
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

    st.markdown("### Скачать")
    zip_buf = io.BytesIO()
    with zipfile.ZipFile(zip_buf, "w", compression=zipfile.ZIP_DEFLATED) as zf:
        for fname, fbytes in results_for_zip:
            zf.writestr(fname, fbytes)
    zip_buf.seek(0)

    st.download_button(
        label="Скачать ZIP с результатами",
        data=zip_buf,
        file_name="facescanner_results.zip",
        mime="application/zip",
        use_container_width=True,
    )
