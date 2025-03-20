import streamlit as st
import requests
import json
import logging
import cv2
import numpy as np
import matplotlib.pyplot as plt
from typing import Optional
from ultralytics import YOLO
from PIL import Image
import base64

# Constants
BASE_API_URL = "https://fond-jaybird-touching.ngrok-free.app"
FLOW_ID = "bf94ab78-019e-4853-879a-a1f6db2cfb7e"
ENDPOINT = "" # You can set a specific endpoint name in the flow settings

TWEAKS = {
  "ChatOutput-Ak4aa": {},
  "Prompt-ix3Ht": {},
  "OpenAIModel-Vl0ut": {},
  "TextInput-egPWI": {},
  "Memory-90Pf3": {},
  "Prompt-TzV8F": {},
  "ChatInput-vG3C5": {},
  "Chroma-Ev84W": {},
  "ParseData-T03YE": {},
  "OpenAIEmbeddings-V1XJg": {}
}
# Initialize logging
logging.basicConfig(level=logging.INFO)

def run_flow(message: str,
             endpoint: str = FLOW_ID,
             output_type: str = "chat",
             input_type: str = "chat",
             tweaks: Optional[dict] = None,
             api_key: Optional[str] = None) -> dict:
    """
    Run a flow with a given message and optional tweaks.

    :param message: The message to send to the flow
    :param endpoint: The ID or the endpoint name of the flow
    :param tweaks: Optional tweaks to customize the flow
    :return: The JSON response from the flow
    """
    api_url = f"{BASE_API_URL}/api/v1/run/{endpoint}"

    payload = {
        "input_value": message,
        "output_type": output_type,
        "input_type": input_type,
    }

    if tweaks:
        payload["tweaks"] = tweaks

    headers = {"x-api-key": api_key} if api_key else None
    response = requests.post(api_url, json=payload, headers=headers)

    # Log the response for debugging
    logging.info(f"Response Status Code: {response.status_code}")
    logging.info(f"Response Text: {response.text}")

    try:
        return response.json()
    except json.JSONDecodeError:
        logging.error("Failed to decode JSON from the server response.")
        return {}

def extract_message(response: dict) -> str:
    try:
        # Extract the response message
        return response['outputs'][0]['outputs'][0]['results']['message']['text']
    except (KeyError, IndexError):
        logging.error("No valid message found in response.")
        return "No valid message found in response."

def clear_chat():
    st.session_state["analysis_done"] = False  # Reset analysis status
    TWEAKS = {"Memory-90Pf3": {}, "ChatInput-vG3C5": {}}  # Clear tweaks state
    st.session_state["messages"] = []  # Clear message history
    st.session_state["uploaded_file_key"] += 1  # Increment key to force file uploader reset
    st.session_state["detected_objects"] = ""  # Reset detected objects
    st.session_state["uploaded_image"] = None  # Clear uploaded image state
    st.session_state["camera_image"] = None  # Clear camera image state
    st.rerun()  # Refresh the app to reflect changes

def display_bounding_boxes(image, results):
    """Draws bounding boxes and labels on an image."""
    image_np = np.array(image)
    for result in results:
        for box in result.boxes:
            x1, y1, x2, y2 = map(int, box.xyxy[0])
            confidence = box.conf[0].item()
            class_name = result.names[int(box.cls[0])]
            
            cv2.rectangle(image_np, (x1, y1), (x2, y2), (0, 255, 0), 2)
            cv2.putText(image_np, f"{class_name} ({confidence:.2f})", (x1, y1 - 10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
    return image_np

def set_bg(image_file):
    with open(image_file, "rb") as f:
        encoded_string = base64.b64encode(f.read()).decode()
    
    bg_css = f"""
    <style>
    .stApp {{
        background-image: url("data:image/png;base64,{encoded_string}");
        background-size: cover;
        background-position: center;
        background-attachment: fixed;
    }}
    </style>
    """
    st.markdown(bg_css, unsafe_allow_html=True)

def set_gradient_bg():
    bg_css = """
    <style>
    body, html, .stApp {
        height: 100%;
        width: 100%;
        margin: 0;
        padding: 0;
        overflow: hidden;
        background: linear-gradient(135deg, #d3d3d3, #f0f0f0);
        background-size: cover;
        background-position: center;
        background-attachment: fixed;
    }
    </style>
    """
    st.markdown(bg_css, unsafe_allow_html=True)

def main():
    st.set_page_config(page_title="Welding Defect Detector", layout="wide")
    st.title("🔎 AI-Powered Welding Inspection")
    st.write("Upload an image of a weld, and let the AI analyze defects and answer your questions!")
    set_gradient_bg()
    
    if "uploaded_file_key" not in st.session_state:
        st.session_state["uploaded_file_key"] = 0  # Initialize uploaded file key
    if "messages" not in st.session_state:
        st.session_state["messages"] = []
    
    with st.sidebar:
        if st.button('🔄 Restart'):
            clear_chat()
        uploaded_file = st.file_uploader("Upload an image", type=["jpg", "jpeg", "png"], key=f"uploaded_file_{st.session_state['uploaded_file_key']}")
        enable = st.checkbox("Enable camera")
        camera = st.camera_input("Take a picture", disabled=not enable)

    col1, col2 = st.columns(2)

    if "analysis_done" not in st.session_state:
        st.session_state["analysis_done"] = False  # Track if analysis has been done

    # Handle uploaded image
    if uploaded_file and not st.session_state["analysis_done"]:
        image = Image.open(uploaded_file)
        col1.image(image, caption="Uploaded Image", use_container_width=True)
        model = YOLO('best.pt')

        with st.spinner("Analyzing image..."):
            results = model(image, conf=0.5, iou=0.6)
            processed_image = display_bounding_boxes(image, results)
            TWEAKS["TextInput-egPWI"]["input_value"] = results[0].to_json()
            response = extract_message(run_flow('what can you see', tweaks=TWEAKS))
            
            if not st.session_state["analysis_done"]:
                st.session_state["messages"].append({"role": "assistant", "content": response, "avatar": "🤖"})
                st.session_state["analysis_done"] = True  # Mark as done

        col2.image(processed_image, caption="Detected Objects", use_container_width=True)

        detected_classes = [result.names[int(box.cls[0])] for result in results for box in result.boxes]
        st.session_state["detected_objects"] = ", ".join(set(detected_classes))
        st.success(f"Objects detected: {st.session_state['detected_objects']}")

    # Handle camera input
    if camera and not st.session_state["analysis_done"]:
        image = Image.open(camera)
        col1.image(image, caption="Captured Image", use_container_width=True)
        model = YOLO('best.pt')

        with st.spinner("Analyzing image..."):
            results = model(image, conf=0.5, iou=0.6)
            processed_image = display_bounding_boxes(image, results)
            TWEAKS["TextInput-egPWI"]["input_value"] = results[0].to_json()
            response = extract_message(run_flow('what can you see', tweaks=TWEAKS))
            
            if not st.session_state["analysis_done"]:
                st.session_state["messages"].append({"role": "assistant", "content": response, "avatar": "🤖"})
                st.session_state["analysis_done"] = True  # Mark as done

        col2.image(processed_image, caption="Detected Objects", use_container_width=True)

        detected_classes = [result.names[int(box.cls[0])] for result in results for box in result.boxes]
        st.session_state["detected_objects"] = ", ".join(set(detected_classes))
        st.success(f"Objects detected: {st.session_state['detected_objects']}")

    
    # Chat Section
    for msg in st.session_state["messages"]:
        with st.chat_message(msg["role"], avatar=msg["avatar"]):
            st.write(msg["content"])
    
    query = st.chat_input("Ask about the image...")
    if query:
        st.session_state["messages"].append({"role": "user", "content": query, "avatar": "🤔"})
        with st.chat_message("user", avatar="🤔"):
            st.write(query)
        
        with st.chat_message("assistant", avatar="🤖"):
            message_placeholder = st.empty()
            with st.spinner("Thinking..."):
                response = extract_message(run_flow(query, tweaks=TWEAKS))
                message_placeholder.write(response)
            st.session_state["messages"].append({"role": "assistant", "content": response, "avatar": "🤖"})


if __name__ == "__main__":
    main()