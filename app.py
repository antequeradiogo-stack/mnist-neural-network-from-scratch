"""Interactive Streamlit demo for the saved from-scratch MNIST network."""

from __future__ import annotations

import hashlib
from pathlib import Path

import numpy as np
import streamlit as st
from streamlit_drawable_canvas import st_canvas

from evaluate_candidate import load_checkpoint
from src.demo_utils import predict_digit, preprocess_canvas_image, validate_candidate_network

ROOT = Path(__file__).resolve().parent
CHECKPOINT = ROOT / "models" / "mnist_glorot_candidate.npz"
REPOSITORY_URL = "https://github.com/antequeradiogo-stack/mnist-neural-network-from-scratch"

st.set_page_config(
    page_title="MNIST Neural Network — Built from Scratch",
    page_icon="✍️",
    layout="wide",
    initial_sidebar_state="collapsed",
)


@st.cache_resource
def load_candidate():
    """Load and validate the committed candidate checkpoint once per app process."""
    network = load_checkpoint(CHECKPOINT)
    validate_candidate_network(network)
    return network


def render_styles() -> None:
    st.markdown(
        """
        <style>
        .stApp { background: linear-gradient(180deg, #f5f8ff 0%, #f8fafc 270px, #ffffff 600px); }
        .block-container { max-width: 1180px; padding-top: 2rem; padding-bottom: 3rem; }
        .hero { padding: 1.7rem 1.9rem; border: 1px solid #dce5f4; border-radius: 22px;
                background: rgba(255,255,255,.92); box-shadow: 0 12px 34px rgba(25,48,87,.06); }
        .hero h1 { color: #172b4d; font-size: clamp(2rem, 4vw, 3rem); line-height: 1.12;
                   letter-spacing: -.035em; }
        .eyebrow { color: #4169a7; font-size: .8rem; font-weight: 700; letter-spacing: .12em;
                   text-transform: uppercase; margin-bottom: .4rem; }
        .hero p { color: #536276; font-size: 1.05rem; margin-bottom: .2rem; }
        .hero a { color: #315ea8; font-weight: 650; }
        .muted { color: #64748b; font-size: .92rem; }
        [data-testid="stMetric"] { background: #f3f7ff; padding: .85rem 1rem;
                                    border-radius: 14px; border: 1px solid #e1eafe; }
        div.stButton > button { border-radius: 11px; min-height: 2.8rem; font-weight: 650; }
        /* Keep the 320px drawable component centered instead of leaving the
           remainder of its Streamlit column visible beside the canvas. */
        iframe[src*="streamlit_drawable_canvas.st_canvas"] {
            display: block;
            width: min(320px, 100%) !important;
            max-width: 100%;
            margin-inline: auto;
        }
        </style>
        """,
        unsafe_allow_html=True,
    )


def main() -> None:
    render_styles()
    st.markdown(
        f"""
        <div class="hero">
          <div class="eyebrow">Interactive model demo · Python + NumPy</div>
          <h1>MNIST Neural Network — Built from Scratch</h1>
          <p>Draw a digit and explore how a compact 784 → 30 → 10 neural network responds.</p>
          <p><a href="{REPOSITORY_URL}" target="_blank">View the source code on GitHub ↗</a></p>
        </div>
        """,
        unsafe_allow_html=True,
    )
    st.write("")

    try:
        network = load_candidate()
    except Exception as error:
        st.error(f"Could not load the saved candidate checkpoint: {error}")
        st.stop()

    if "canvas_version" not in st.session_state:
        st.session_state.canvas_version = 0
    if "prediction" not in st.session_state:
        st.session_state.prediction = None
    if "canvas_digest" not in st.session_state:
        st.session_state.canvas_digest = None

    left, right = st.columns([1.05, 0.95], gap="large")
    with left:
        with st.container(border=True):
            st.subheader("Draw a digit")
            st.markdown(
                '<p class="muted">Use your mouse or touch screen. Draw one large digit in white on the dark canvas.</p>',
                unsafe_allow_html=True,
            )
            canvas_result = st_canvas(
                fill_color="rgba(255, 255, 255, 1)",
                stroke_width=24,
                stroke_color="#FFFFFF",
                background_color="#000000",
                update_streamlit=True,
                height=320,
                width=320,
                drawing_mode="freedraw",
                display_toolbar=True,
                key=f"mnist_canvas_{st.session_state.canvas_version}",
            )

            # streamlit-drawable-canvas 0.9.3 hard-codes a light-grey border
            # on its foreground canvas. Its iframe is same-origin with this
            # app, so remove that border on the actual drawing element (not
            # with an overlay) and center the component's own canvas stack.
            # Keep the bitmap attributes at 320 × 320; preprocessing still
            # receives exactly that image and resizes it to 28 × 28.
            st.iframe(
                """
                <script>
                (() => {
                  const patchCanvas = () => {
                    const frame = window.parent.document.querySelector(
                      'iframe[src*="streamlit_drawable_canvas.st_canvas"]'
                    );
                    if (!frame) return false;
                    try {
                      frame.style.setProperty("width", "min(320px, 100%)", "important");
                      frame.style.setProperty("max-width", "100%", "important");
                      frame.style.display = "block";
                      frame.style.marginInline = "auto";
                      if (frame.parentElement) {
                        frame.parentElement.style.display = "flex";
                        frame.parentElement.style.justifyContent = "center";
                      }
                      const doc = frame.contentDocument;
                      const canvas = doc && doc.querySelector("canvas#canvas");
                      const background = doc && doc.querySelector("canvas#backgroundimage-canvas");
                      if (!canvas || !background) {
                        if (doc && doc.body && !doc.body.dataset.mnistCanvasObserver) {
                          doc.body.dataset.mnistCanvasObserver = "active";
                          const canvasObserver = new MutationObserver(() => {
                            if (patchCanvas()) canvasObserver.disconnect();
                          });
                          canvasObserver.observe(doc.body, {childList: true, subtree: true});
                        }
                        return false;
                      }
                      canvas.style.border = "0";
                      canvas.style.maxWidth = "100%";
                      canvas.style.height = "auto";
                      background.style.backgroundColor = "#000000";
                      background.style.maxWidth = "100%";
                      background.style.height = "auto";
                      return true;
                    } catch (_) {
                      return false;
                    }
                  };
                  const observer = new MutationObserver(() => {
                    if (patchCanvas()) observer.disconnect();
                  });
                  observer.observe(window.parent.document.body, {
                    childList: true,
                    subtree: true
                  });
                  patchCanvas();
                  setTimeout(() => observer.disconnect(), 5000);
                })();
                </script>
                """,
                height="content",
                width="content",
            )

            image_data = canvas_result.image_data
            if image_data is not None:
                digest = hashlib.sha256(np.asarray(image_data).tobytes()).hexdigest()
                if digest != st.session_state.canvas_digest:
                    st.session_state.canvas_digest = digest
                    st.session_state.prediction = None

            predict_col, clear_col = st.columns(2, gap="small")
            predict_clicked = predict_col.button("Predict", type="primary", width="stretch")
            clear_clicked = clear_col.button("Clear", width="stretch")

        if clear_clicked:
            st.session_state.canvas_version += 1
            st.session_state.canvas_digest = None
            st.session_state.prediction = None
            st.rerun()

        if predict_clicked:
            if image_data is None:
                st.session_state.prediction = None
                st.warning("Draw a digit on the canvas before predicting.")
            else:
                try:
                    input_column = preprocess_canvas_image(image_data)
                    if not np.any(input_column):
                        st.session_state.prediction = None
                        st.warning("The canvas is blank. Draw a digit, then try again.")
                    else:
                        digit, scores = predict_digit(network, input_column)
                        st.session_state.prediction = {
                            "digit": digit,
                            "scores": scores[:, 0].tolist(),
                        }
                except (TypeError, ValueError) as error:
                    st.session_state.prediction = None
                    st.error(f"Could not process this drawing: {error}")

    with right:
        with st.container(border=True):
            st.subheader("Model response")
            prediction = st.session_state.prediction
            if prediction is None:
                st.info("Your prediction will appear here after you draw and select Predict.")
            else:
                st.markdown(
                    f"<div class='eyebrow'>Predicted digit</div><div style='font-size:5rem;line-height:1.1;"
                    f"font-weight:750;color:#234e91'>{prediction['digit']}</div>",
                    unsafe_allow_html=True,
                )
                st.markdown("#### Sigmoid output scores")
                scores = np.asarray(prediction["scores"], dtype=np.float64)
                st.bar_chart({"Output score": scores}, x_label="Digit", y_label="Score")
                with st.expander("Show all ten scores"):
                    st.dataframe(
                        [{"Digit": digit, "Sigmoid score": float(score)} for digit, score in enumerate(scores)],
                        hide_index=True,
                        width="stretch",
                    )
                st.caption(
                    "These sigmoid activations are model scores, not calibrated probabilities; "
                    "they need not sum to 1."
                )

    st.write("")
    metric_col, note_col = st.columns([0.35, 0.65], gap="large")
    with metric_col:
        st.metric("MNIST test-set accuracy (reported)", "95.82%")
    with note_col:
        st.markdown(
            '<p class="muted">This is the saved model’s reported evaluation on the held-out MNIST test set. '
            'It is not an accuracy score for drawings submitted in this demo.</p>',
            unsafe_allow_html=True,
        )

    with st.expander("How drawings are prepared"):
        st.write(
            "MNIST images use a dark background with brighter digit pixels, normalized to [0, 1]. "
            "This demo draws white ink on black, composites transparency over black, crops the ink, "
            "scales it proportionally to fit inside 20 × 20 pixels, places it on a 28 × 28 image, "
            "and shifts its intensity center toward the image center. It then flattens the image to "
            "a (784, 1) input column. This approximates common MNIST centering, but visitor drawings "
            "can differ in handwriting, stroke thickness, and positioning."
        )

    st.caption(
        f"Architecture: 784 → 30 → 10 · sigmoid activation · saved checkpoint: `{CHECKPOINT.relative_to(ROOT)}`"
    )


if __name__ == "__main__":
    main()
