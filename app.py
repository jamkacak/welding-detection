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

# Constants
BASE_API_URL = "http://127.0.0.1:7860"
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
    """Clears chat history, resets uploaded file, and clears Langflow session."""
    TWEAKS = {"Memory-90Pf3": {}}
    st.session_state["messages"] = []
    st.session_state["uploaded_file_key"] += 1  # Increment key to force file uploader reset
    st.session_state["detected_objects"] = ""  # Reset detected objects
    st.rerun()

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

def main():
    st.set_page_config(page_title="Welding Defect Detector", layout="wide")
    st.title("🔎 AI-Powered Welding Inspection")
    st.write("Upload an image of a weld, and let the AI analyze defects and answer your questions!")

    
    if "uploaded_file_key" not in st.session_state:
        st.session_state["uploaded_file_key"] = 0  # Initialize uploaded file key
    if "messages" not in st.session_state:
        st.session_state["messages"] = []
    
    with st.sidebar:
        st.image('welder-happy.gif')
        if st.button('🔄 Restart'):
            clear_chat()
        uploaded_file = st.file_uploader("Upload an image", type=["jpg", "jpeg", "png"], key=f"uploaded_file_{st.session_state['uploaded_file_key']}")
    
    col1, col2 = st.columns(2)
    if uploaded_file:
        image = Image.open(uploaded_file)
        col1.image(image, caption="Uploaded Image", use_container_width=True)
        model = YOLO(r'C:\Users\suhaimi\Desktop\Nidzam\capstone\runs\detect\train6\weights\best.pt')
        
        with st.spinner("Analyzing image..."):
            results = model(image, conf=0.5, iou=0.6)
            processed_image = display_bounding_boxes(image, results)
            TWEAKS["TextInput-egPWI"]["input_value"] = results[0].to_json()
            # st.text(results[0].to_json())
        
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
