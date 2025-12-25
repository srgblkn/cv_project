# app.py
from __future__ import annotations

import streamlit as st
from datetime import datetime


APP_TITLE = "Vision Suite"
APP_SUBTITLE = "Инструменты компьютерного зрения для повседневных бизнес-задач"


def page_link_or_button(page_path: str, label: str, icon: str = "→"):
    """
    Streamlit даёт разные способы навигации в зависимости от версии.
    - Если доступен st.page_link — используем его.
    - Иначе используем кнопку + st.switch_page.
    """
    if hasattr(st, "page_link"):
        st.page_link(page_path, label=f"{label}", icon=icon)
    else:
        # Fallback для более старых версий Streamlit
        if st.button(label, use_container_width=True):
            try:
                st.switch_page(page_path)
            except Exception:
                st.info("Навигация недоступна в текущей версии Streamlit. Используйте меню слева.")


def render_header():
    st.set_page_config(
        page_title=APP_TITLE,
        page_icon="🧩",
        layout="wide",
        initial_sidebar_state="expanded",
    )

    st.markdown(
        f"""
        <div style="padding: 0.2rem 0 0.6rem 0;">
          <div style="font-size: 2.2rem; font-weight: 700; line-height: 1.1;">{APP_TITLE}</div>
          <div style="font-size: 1.05rem; opacity: 0.85; margin-top: 0.35rem;">{APP_SUBTITLE}</div>
        </div>
        """,
        unsafe_allow_html=True,
    )


def render_sidebar():
    st.sidebar.markdown("### Разделы")
    st.sidebar.caption("Выберите модуль. Страницы в папке `pages/` появятся в меню автоматически.")

    st.sidebar.markdown("#### Быстрый переход")
    page_link_or_button("pages/facescanner.py", "FaceScanner — маскировка лиц", icon="🕵️")

    st.sidebar.divider()
    st.sidebar.markdown("#### О сервисе")
    st.sidebar.write("• Пакетная обработка файлов")
    st.sidebar.write("• Быстрый прототип → замена весов без переписывания UI")
    st.sidebar.write("• Понятный формат результатов для конечного пользователя")

    st.sidebar.divider()
    st.sidebar.caption(f"Сессия: {datetime.now().strftime('%Y-%m-%d %H:%M')}")


def render_hero_actions():
    c1, c2 = st.columns([1.2, 1.0], gap="large")

    with c1:
        st.markdown("### Выберите модуль")
        st.write(
            "Три независимых инструмента для типовых сценариев: "
            "анонимизация, медицинская классификация/детекция, мониторинг территорий."
        )

    with c2:
        st.markdown("### Начать работу")
        page_link_or_button("pages/facescanner.py", "Открыть FaceScanner", icon="🕵️")


def render_solution_cards():
    st.markdown("### Решения")
    a, b, c = st.columns(3, gap="large")

    with a:
        st.markdown("#### 1) FaceScanner")
        st.caption("Анонимизация изображений")
        st.write("Детекция лиц и маскировка области (blur/pixelate/заливка). Поддерживает загрузку нескольких файлов.")
        page_link_or_button("pages/facescanner.py", "Перейти к FaceScanner", icon="🕵️")

    with b:
        st.markdown("#### 2) BrainScan Detect")
        st.caption("Детекция опухолей мозга")
        st.write("Детекция объектов на изображениях. Планируется поддержка загрузки файлов и загрузки по прямой ссылке.")
        st.button("Скоро доступно", use_container_width=True, disabled=True)

    with c:
        st.markdown("#### 3) Forest Segmentation")
        st.caption("Сегментация лесных массивов")
        st.write("Семантическая сегментация спутниковых снимков (бинарные маски) для оценки покрытий и изменений.")
        st.button("Скоро доступно", use_container_width=True, disabled=True)


def render_how_it_works():
    st.markdown("### Как это работает")
    x1, x2, x3 = st.columns(3, gap="large")

    with x1:
        st.markdown("**1. Загрузка**")
        st.write("Загрузите один или несколько файлов. Для некоторых модулей будет доступна загрузка по ссылке.")

    with x2:
        st.markdown("**2. Обработка**")
        st.write("Сервис выполняет инференс модели и формирует результат в понятном для бизнеса виде.")

    with x3:
        st.markdown("**3. Результат**")
        st.write("Просмотр превью на странице и скачивание итогов (например, ZIP с обработанными файлами).")


def render_quality_block():
    st.markdown("### Качество и надёжность")
    st.write(
        "В каждом модуле предусмотрен блок с описанием модели, основными метриками и рекомендациями "
        "по настройкам (например, пороги confidence/IoU или подходящие сценарии применения)."
    )
    with st.expander("Что будет в блоке качества на страницах модулей", expanded=False):
        st.write("• Число эпох обучения")
        st.write("• Размер обучающей и валидационной выборок")
        st.write("• Метрики качества (для детекции: mAP, PR-кривая, confusion matrix; для сегментации: IoU/Dice и т.п.)")
        st.write("• Рекомендации по применению и ограничения")


def render_footer():
    st.divider()
    st.caption("Работу выполнили студенты Эльбруса — Игорь Никоновский и Сергей Белькин")


def main():
    render_header()
    render_sidebar()

    render_hero_actions()
    st.divider()

    render_solution_cards()
    st.divider()

    render_how_it_works()
    st.divider()

    render_quality_block()
    render_footer()


if __name__ == "__main__":
    main()
