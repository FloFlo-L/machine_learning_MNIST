import streamlit as st
from streamlit_drawable_canvas import st_canvas
from PIL import Image
import numpy as np
import requests
import io

API_URL = "http://localhost:8000/api/v1/predict"

st.markdown(
    """
    <style>
    .main {
        background-color: #f7f7f7;
    }
    .stButton>button {
        background: #262730;
        color: #fff;
        border-radius: 8px;
        border: none;
        padding: 0.5em 1.5em;
        font-weight: bold;
        margin-top: 1em;
    }
    .stButton>button:hover {
        background: #4f8bf9;
        color: #fff;
    }
    </style>
    """,
    unsafe_allow_html=True
)

st.title("🖌️ MNIST Digit Recognizer")
st.write("Draw a digit (0-9) on the left, visualize the image on the right, then send it to get the prediction from the model.")

st.markdown("---")

col1, col2 = st.columns([2, 1])

with col1:
    st.markdown("**Drawing zone**")
    canvas_result = st_canvas(
        fill_color="rgba(255, 255, 255, 1)",
        stroke_width=12,
        stroke_color="#000000",
        background_color="#FFFFFF",
        width=200,
        height=200,
        drawing_mode="freedraw",
        key="canvas",
    )

with col2:
    st.markdown("**Preview 28x28**")
    if canvas_result.image_data is not None:
        img = Image.fromarray((255 - canvas_result.image_data[:, :, 0]).astype(np.uint8))
        img = img.resize((28, 28))
        st.image(img, caption="Image to send", width=100)
    else:
        st.info("Draw a digit on the left.")

st.markdown("---")

# Zone de résultat
result_placeholder = st.empty()

# Désactivation du bouton pendant la soumission
if img is not None:
    with st.form("predict_form"):
        submit = st.form_submit_button("🚀 Send for prediction")
        if submit:
            result_placeholder.info("Sending image to the API...")
            # Sauvegarder l'image dans un buffer
            buf = io.BytesIO()
            img.save(buf, format="PNG")
            buf.seek(0)
            files = {"file": ("digit.png", buf, "image/png")}
            try:
                response = requests.post(API_URL, files=files, timeout=10)
                if response.status_code == 200:
                    data = response.json()
                    result_placeholder.success(
                        f"Predicted digit : **{data['predicted_digit']}** (confidence : {data['confidence']:.2%})"
                    )
                else:
                    result_placeholder.error(f"API error : {response.text}")
            except Exception as e:
                result_placeholder.error(f"Erreur lors de l'appel à l'API : {e}")
else:
    st.button("🚀 Send for prediction", disabled=True)