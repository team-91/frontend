import os
from datetime import datetime

import numpy as np
import pydicom
import requests
import streamlit as st
from PIL import Image

BACKEND_HOST = os.getenv("BACKEND_HOST", "http://localhost:8080")
FORWARD_URL = f"{BACKEND_HOST}/forward"
HISTORY_URL = f"{BACKEND_HOST}/history"

st.set_page_config(page_title="Chest X-ray Classification", layout="centered")
st.title("Chest X-ray Classification")

tab_classification, tab_history = st.tabs(["Classification", "History"])

with tab_classification:
    uploaded_file = st.file_uploader("Upload an X-ray image", type=["dcm", "dicom"])

    if uploaded_file is not None:
        col1, col2 = st.columns(2)

        dicom = pydicom.dcmread(uploaded_file)
        pixel_array = dicom.pixel_array

        pixel_array = pixel_array.astype(np.float32)
        pixel_array = (pixel_array - pixel_array.min()) / (
            pixel_array.max() - pixel_array.min() + 1e-8
        )
        pixel_array = (pixel_array * 255).astype(np.uint8)

        image = Image.fromarray(pixel_array)
        with col1:
            col1.image(image, caption="Uploaded X-ray Image", use_container_width=True)

        with col2:
            st.subheader("Classification Result")
            uploaded_file.seek(0)

            try:
                response = requests.post(FORWARD_URL, files={"file": uploaded_file})

                if response.ok:
                    result = response.json()

                    pred = result.get("prediction", "N/A")
                    if pred != "N/A":
                        pred = "Negative" if pred == 0 else "Positive"

                    st.metric("Prediction", pred)
                    probability = result.get("probability", 0)
                    st.metric("Probability", f"{probability:.2%}")
                else:
                    st.error(f"Error from server: {response.status_code} - {response.text}")

            except requests.exceptions.ConnectionError:
                st.error(f"Could not connect to backend at {FORWARD_URL}")
            except requests.exceptions.RequestException as e:
                st.error(f"Request failed: {e}")

with tab_history:
    st.subheader("Request History")

    if st.button("Refresh History"):
        st.rerun()

    try:
        response = requests.get(HISTORY_URL)

        if response.ok:
            data = response.json()
            requests_list = data.get("requests", [])

            if not requests_list:
                st.info("No history available yet.")
            else:
                requests_list = sorted(
                    requests_list,
                    key=lambda x: x.get("timestamp", ""),
                    reverse=True,
                )

                for req in requests_list:
                    with st.container(border=True):
                        col1, col2 = st.columns([2, 1])

                        with col1:
                            st.markdown(f"ID: `{req.get('id', 'N/A')}`")

                            timestamp_str = req.get("timestamp", "")
                            if timestamp_str:
                                try:
                                    ts = datetime.fromisoformat(
                                        timestamp_str.replace("Z", "+00:00")
                                    )
                                    formatted_ts = ts.strftime("%b %d, %Y, %I:%M %p")
                                except ValueError:
                                    formatted_ts = timestamp_str
                            else:
                                formatted_ts = "N/A"

                            st.text(f"Timestamp: {formatted_ts}")
                            st.text(
                                f"Image Size: {req.get('img_width', 'N/A')} x {req.get('img_height', 'N/A')}"
                            )

                        with col2:
                            result_value = req.get("result")
                            result_text = "Positive" if result_value else "Negative"
                            st.metric("Result", result_text)
        else:
            st.error(
                f"Error fetching history: {response.status_code} - {response.text}"
            )

    except requests.exceptions.ConnectionError:
        st.error(f"Could not connect to backend at {HISTORY_URL}")
    except requests.exceptions.RequestException as e:
        st.error(f"Request failed: {e}")
