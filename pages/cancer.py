# pages/cancer.py
import base64
import io
import zipfile
from pathlib import Path

import numpy as np
import pandas as pd
import streamlit as st
from PIL import Image, ImageDraw

try:
    from ultralytics import YOLO
except Exception:
    YOLO = None


# -----------------------------
# Paths (строго по вашим директориям/именам)
# -----------------------------
THIS_DIR = Path(__file__).resolve().parent
ART_DIR = THIS_DIR / "cancerbook"

WEIGHTS_PATH = ART_DIR / "best.pt"
ARGS_PATH = ART_DIR / "args.yaml"
RESULTS_PATH = ART_DIR / "results.csv"
BG_JPG_LIST = sorted(ART_DIR.glob("*.jpg"))  # у вас: screen.jpg


# -----------------------------
# Page config
# -----------------------------
st.set_page_config(page_title="Medical Scan Analyzer", page_icon="🧠", layout="wide")


# -----------------------------
# UI helpers
# -----------------------------
def opaque_card(title: str, text: str) -> None:
    st.markdown(
        f"""
        <div class="opaque-card">
          <h3>{title}</h3>
          <p>{text}</p>
        </div>
        """,
        unsafe_allow_html=True,
    )


def safe_switch_page(target: str) -> None:
    if hasattr(st, "switch_page"):
        try:
            st.switch_page(target)
        except Exception:
            st.info("Переход недоступен. Используйте меню слева.")
    else:
        st.info("Переход недоступен. Используйте меню слева.")


def apply_background_and_contrast(bg_path: Path | None) -> None:
    bg_css = ""
    if bg_path and bg_path.exists():
        b64 = base64.b64encode(bg_path.read_bytes()).decode("utf-8")
        bg_css = f"""
        .stApp {{
            background-image: url("data:image/jpeg;base64,{b64}");
            background-size: cover;
            background-position: center;
            background-attachment: fixed;
        }}
        """

    st.markdown(
        f"""
        <style>
        {bg_css}

        .stApp, .stMarkdown, .stText, .stCaption, .stWrite {{
            color: #F8FAFC;
        }}

        header[data-testid="stHeader"] {{ background: rgba(0,0,0,0); }}

        section[data-testid="stSidebar"] {{
            background: #0B1220;
            border-right: 1px solid rgba(255,255,255,0.10);
        }}
        section[data-testid="stSidebar"] * {{ color: #F8FAFC !important; }}

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

        div[data-testid="stFileUploader"] section {{
            background: #0B1220;
            border: 1px solid rgba(255,255,255,0.12);
            border-radius: 18px;
            padding: 10px;
        }}

        .stButton > button {{
            border-radius: 14px;
            border: 1px solid rgba(255,255,255,0.14);
        }}

        a {{ color: #93C5FD !important; }}
        </style>
        """,
        unsafe_allow_html=True,
    )


def parse_yaml_shallow(path: Path) -> dict:
    """
    Очень простой парсер key: value (без вложенности). Нам достаточно для таблицы ключевых параметров.
    """
    out = {}
    if not path.exists():
        return out
    for line in path.read_text(encoding="utf-8", errors="ignore").splitlines():
        s = line.strip()
        if (not s) or s.startswith("#") or (":" not in s):
            continue
        k, v = s.split(":", 1)
        k = k.strip()
        v = v.strip().strip("'").strip('"')
        if not v:
            continue
        out[k] = v
    return out


def pick_first(args: dict, keys: list) -> str:
    for k in keys:
        if k in args and str(args[k]).strip():
            return str(args[k]).strip()
    return "—"


def looks_like_lfs_pointer(p: Path) -> bool:
    if not p.exists():
        return False
    head = p.read_bytes()[:200]
    txt = head.decode("utf-8", errors="ignore")
    return ("git-lfs" in txt) and ("git-lfs.github.com/spec" in txt)


def ensure_weights_ok_or_stop(p: Path) -> None:
    if not p.exists():
        st.error(f"Не найден файл весов: `{p.as_posix()}`")
        st.stop()
    if looks_like_lfs_pointer(p):
        st.error("Файл весов выглядит как Git LFS pointer (ссылка), а не бинарный .pt.")
        st.stop()


@st.cache_resource(show_spinner=False)
def load_yolo_model(weights_path: str):
    if YOLO is None:
        raise RuntimeError("ultralytics не установлен.")
    return YOLO(weights_path)


def draw_boxes(i_
