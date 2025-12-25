# pages/facescanner.py
from __future__ import annotations

import base64
import io
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

# YAML (опционально; если pyyaml не установлен, покажем сырой текст)
try:
    import yaml  # type: ignore
except Exception:
    yaml = None


# =============================
# Paths (строго по вашим директориям/именам)
# =============================
THIS_DIR = Path(__file__).resolve().parent
FB_DIR = THIS_DIR / "facebook"

WEIGHTS_PATH = FB_DIR / "best-13.pt"
ARGS_PATH = FB_DIR / "args.yaml"
RESULTS_PATH = FB_DIR / "results.csv"

# Фон: любой *.jpg в pages/facebook
BG_JPG_LIST = sorted(FB_DIR.glob("*.jpg"))


# =============================
# Utils: background + high contrast opaque UI
# =============================
def apply_background_and_contrast(bg_path: Path | None):
    bg_css = ""
    if bg_path is not None and bg_path.exists():
        b64 = base64.b64encode(bg_path.read_bytes()).decode("utf-8")
        bg_css = f"""
        .stApp {{
            background-image: url("data:image/jpeg;base64,{b64}");
            background-size: cover;
            background-position: center;
            background-attachment: fixed;
        }}
        """

    css = f"""
    <style>
    {bg_css}

    /* Максимальный контраст текста */
    .stApp, .stMarkdown, .stText, .stCaption, .stWrite {{
        color: #F8FAFC;
    }}

    header[data-testid="stHeader"] {{
        background: rgba(0,0,0,0);
    }}

    /* Sidebar: непрозрачный */
    section[data-testid="stSidebar"] {{
        background: #0B1220;
        border-right: 1px solid rgba(255,255,255,0.10);
    }}
    section[data-testid="stSidebar"] * {{
        color: #F8FAFC !important;
    }}

    /* Оpaque card */
    .opaque-card {{
        background: #0B1220;
        border: 1px solid rgba(255,255,255,0.12);
        border-radius: 18px;
        padding: 16px 16px 14px 16px;
        box-shadow: 0 10px 24px rgba(0,0,0,0.40);
        margin-bottom: 14px;
    }}
    .opaque-card h3 {{
        margin: 0;
        font-size: 1.25rem;
        font-weight: 750;
        color: #F8FAFC;
    }}
    .opaque-card p {{
        margin: 6px 0 0 0;
        color: rgba(248,250,252,0.85);
        line-height: 1.35;
    }}

    /* Expander: непрозрачный */
    div[data-testid="stExpander"] > details {{
        background: #0B1220;
        border: 1px solid rgba(255,255,255,0.12);
        border-radius: 18px;
        padding: 10px 12px;
        box-shadow: 0 10px 24px rgba(0,0,0,0.30);
    }}
    div[data-testid="stExpander"] summary {{
        color: #F8FAFC !important;
        font-weight: 650;
    }}

    /* File uploader: непрозрачный */
    div[data-testid="stFileUploader"] section {{
        background: #0B1220;
        border: 1px solid rgba(255,255,255,0.12);
        border-radius: 18px;
        padding: 10px;
    }}

    /* Buttons */
    .stButton > button {{
        border-radius: 14px;
        border: 1px solid rgba(255,255,255,0.14);
    }}

    a {{
        color: #93C5FD !important;
    }}
    </style>
    """
    st.markdown(css, unsafe_allow_html=True)


def opaque_card(title: str, text: str):
    st.markdown(
        f"""
        <div class="opaque-card">
          <h3>{title}</h3>
          <p>{text}</p>
        </div>
        """,
        unsafe_allow_html=True,
    )


def safe_switch_page(target: str):
    if hasattr(st, "switch_page"):
        try:
            st.switch_page(target)
        except Exception:
            st.info("Переход недоступен в этой среде. Используйте меню слева.")
    else:
        st.info("Переход недоступен в этой версии Streamlit. Используйте меню слева.")


# =============================
# Page config
# =============================
st.set_page_config(
    page_title="FaceScanner — маскировка лиц",
    page_icon="🕵️",
    layout="wide",
)


# =============================
# Sidebar: background selector + downloads
# =============================
st.sidebar.markdown("## FaceScanner")
st.sidebar.caption("Артефакты лежат в `pages/facebook/`.")

# Выбор фона из списка *.jpg (если один — просто используем его)
bg_path: Path | None = None
if len(BG_JPG_LIST) == 0:
    st.sidebar.warning("Фон *.jpg не найден в `pages/facebook/`.")
else:
    if len(BG_JPG_LIST) == 1:
        bg_path = BG_JPG_LIST[0]
        st.sidebar.success(f"Фон: {bg_path.name}")
    else:
        bg_name = st.sidebar.selectbox(
            "Фон (выберите *.jpg)",
            options=[p.name for p in BG_JPG_LIST],
            index=0,
        )
        bg_path = FB_DIR / bg_name

# Применяем фон и контраст (после выбора)
apply_background_and_contrast(bg_path)

st.sidebar.divider()
st.sidebar.markdown("### Файлы модели")
st.sidebar.write(f"• weights: `{WEIGHTS_PATH.name}`", "✅" if WEIGHTS_PATH.exists() else "❌")
st.sidebar.write(f"• args: `{ARGS_PATH.name}`", "✅" if ARGS_PATH.exists() else "❌")
st.sidebar.write(f"• results: `{RESULTS_PATH.name}`", "✅" if RESULTS_PATH.exists() else "❌")

if bg_path is not None and bg_path.exists():
    st.sidebar.download_button(
        "Скачать фон (JPG)",
        data=bg_path.read_bytes(),
        file_name=bg_path.name,
        mime="image/jpeg",
        use_container_width=True,
    )

st.sidebar.divider()
st.sidebar.markdown("### Инференс")
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


# =============================
# Data structures / helpers
# =============================
@dataclass
class MaskConfig:
    mode: str
    blur_radius: int = 12
    pixel_size: int = 12
    solid_color: Tuple[int, int, int] = (0, 0, 0)
    padding: float = 0.10


mask_cfg = MaskConfig(
    mode=mask_mode,
    blur_radius=blur_radius,
    pixel_size=pixel_size,
    solid_color=solid_color,
    padding=padding,
)


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
        else:
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


# =============================
# Header (opaque)
# =============================
opaque_card(
    "FaceScanner",
    "Пакетная обработка изображений: детекция лиц (YOLO) и маскировка обнаруженных областей. "
    "Веса и отчёты берутся из `pages/facebook/`.",
)

top_l, top_r = st.columns([1, 1], gap="large")
with top_l:
    if st.button("← На главную", use_container_width=True):
        safe_switch_page("app.py")

with top_r:
    # Даем бизнес-пользователю быстрый индикатор “модель подключена”
    if WEIGHTS_PATH.exists():
        st.success(f"Веса подключены: {WEIGHTS_PATH.name}")
    else:
        st.error(f"Нет весов: {WEIGHTS_PATH.as_posix()}")


# =============================
# Main layout
# =============================
left, right = st.columns([1.25, 1.0], gap="large")

with left:
    opaque_card("Загрузка изображений", "Загрузите один или несколько файлов. После обработки доступно скачивание ZIP.")
    uploads = st.file_uploader(
        "Файлы изображений",
        type=["png", "jpg", "jpeg", "bmp", "tif", "tiff"],
        accept_multiple_files=True,
        label_visibility="collapsed",
    )
    run_btn = st.button("Запустить обработку", type="primary", use_container_width=True)

with right:
    opaque_card("Качество и обучение", "Артефакты: args.yaml (параметры) и results.csv (динамика по эпохам).")

    with st.expander("args.yaml (параметры обучения)", expanded=True):
        if ARGS_PATH.exists():
            raw = ARGS_PATH.read_text(encoding="utf-8", errors="ignore")
            if yaml is None:
                st.warning("PyYAML не установлен. Добавьте `pyyaml>=6.0` в requirements.txt для красивого парсинга.")
                st.code(raw)
            else:
                try:
                    data = yaml.safe_load(raw)
                    st.json(data)
                except Exception as e:
                    st.error(f"Не удалось прочитать args.yaml: {e}")
                    st.code(raw)
        else:
            st.info(f"Файл не найден: `{ARGS_PATH.as_posix()}`")

    with st.expander("results.csv (графики обучения)", expanded=True):
        if RESULTS_PATH.exists():
            try:
                df = pd.read_csv(RESULTS_PATH)
                st.dataframe(df.tail(20), use_container_width=True)

                # Подхватываем epoch + разумный набор метрик/лоссов (без угадывания имён файлов — только колонки)
                cols_lower = {c.lower(): c for c in df.columns}
                epoch_col = cols_lower.get("epoch")

                # кандидаты на линии
                candidates = []
                for key in ["precision", "recall", "map50", "map50-95", "map50_95", "box_loss", "cls_loss", "dfl_loss"]:
                    for c in df.columns:
                        if key in c.lower():
                            candidates.append(c)
                candidates = list(dict.fromkeys(candidates))[:6]

                # лучшая эпоха, если есть mAP
                score_col = None
                for c in df.columns:
                    cl = c.lower()
                    if "map50-95" in cl or "map50_95" in cl:
                        score_col = c
                        break
                if score_col is None:
                    for c in df.columns:
                        if "map50" in c.lower():
                            score_col = c
                            break

                if epoch_col and score_col:
                    best_idx = int(df[score_col].astype(float).idxmax())
                    best_epoch = int(df.loc[best_idx, epoch_col])
                    best_score = float(df.loc[best_idx, score_col])
                    st.success(f"Лучшая эпоха по `{score_col}`: epoch={best_epoch}, score={best_score:.4f}")

                if epoch_col and candidates:
                    import matplotlib.pyplot as plt

                    for c in candidates:
                        fig = plt.figure()
                        plt.plot(df[epoch_col], df[c])
                        plt.xlabel("epoch")
                        plt.ylabel(c)
                        plt.title(c)
                        st.pyplot(fig, clear_figure=True)

            except Exception as e:
                st.error(f"Не удалось прочитать results.csv: {e}")
        else:
            st.info(f"Файл не найден: `{RESULTS_PATH.as_posix()}`")


# =============================
# Inference
# =============================
if run_btn:
    if YOLO is None:
        st.error("ultralytics не установлен. Проверьте requirements.txt.")
        st.stop()

    if not WEIGHTS_PATH.exists():
        st.error(
            "Файл весов не найден.\n\n"
            f"Ожидается: `{WEIGHTS_PATH.as_posix()}`\n\n"
            "Проверьте, что best-13.pt действительно лежит в `pages/facebook/` и запушен в GitHub."
        )
        st.stop()

    if not uploads:
        st.warning("Загрузите хотя бы один файл.")
        st.stop()

    with st.spinner("Загружаю модель..."):
        model = load_yolo_model(WEIGHTS_PATH.as_posix())

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

            buf = io.BytesIO()
            masked.save(buf, format="PNG")
            buf.seek(0)

            out_name = f"{Path(up.name).stem}_masked.png"
            results_for_zip.append((out_name, buf.getvalue()))
            preview_rows.append((up.name, img, boxed, masked, len(boxes)))

        except Exception as e:
            st.error(f"Ошибка обработки файла {up.name}: {e}")

        prog.progress(int(idx / len(uploads) * 100))

    prog.empty()

    opaque_card("Результаты", "Просмотрите превью и скачайте ZIP с маскированными изображениями.")

    for name, orig, boxed, masked, n_boxes in preview_rows:
        with st.expander(f"{name} — детекций: {n_boxes}", expanded=False):
            c1, c2, c3 = st.columns(3)
            with c1:
                st.markdown("**Оригинал**")
                st.image(orig, use_container_width=True)
            with c2:
                st.markdown("**Детекции**")
                st.image(boxed, use_container_width=True)
            with c3:
                st.markdown("**Маскировано**")
                st.image(masked, use_container_width=True)

    zip_buf = io.BytesIO()
    with zipfile.ZipFile(zip_buf, "w", compression=zipfile.ZIP_DEFLATED) as zf:
        for fname, fbytes in results_for_zip:
            zf.writestr(fname, fbytes)
    zip_buf.seek(0)

    st.download_button(
        "Скачать ZIP с результатами",
        data=zip_buf,
        file_name="facescanner_results.zip",
        mime="application/zip",
        use_container_width=True,
    )
