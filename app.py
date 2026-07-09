import streamlit as st
from ultralytics import YOLO
from PIL import Image

# Set up a clean layout
st.set_page_config(page_title="AI PCB Inspection", layout="centered")
st.title("🛡️ AI-Powered Smart PCB Inspection System")
st.write("Upload a board image to instantly evaluate assembly elements and structural defects.")

# Safely load your custom trained model weights from your repo
@st.cache_resource
def load_model():
    return YOLO("best.pt")

try:
    model = load_model()
    
    # Create the drag-and-drop file uploader component
    uploaded_file = st.file_uploader("Choose a PCB Image...", type=["jpg", "jpeg", "png"])

    if uploaded_file is not None:
        # Load the uploaded file as a clean image object
        image = Image.open(uploaded_file)
        st.image(image, caption="Uploaded PCB Board", use_container_width=True)
        
        st.write("🔄 System processing: Running neural inference on board layout...")
        results = model(image)
        
        # Render the visual bounding boxes onto the image
        res_plotted = results[0].plot()
        
        # Logic rule: If bounding boxes/anomalies are found, flag for rework
        if len(results[0].boxes) > 0:
            st.error("❌ ROUTING RESULT: REWORK REQUIRED (Anomalies / Tracked Components Flagged)")
        else:
            st.success("✅ ROUTING RESULT: PASSED INSPECTION")
            
        # Output the analyzed visual mapping for the user
        st.image(res_plotted, caption="Inspection Analysis Output Map", use_container_width=True)

except Exception as e:
    st.error(f"Initialization Error: Please verify that 'best.pt' is fully uploaded to your root GitHub repository. Details: {e}")
