import streamlit as st
import tensorflow as tf
from tensorflow import keras
import numpy as np
from PIL import Image
import io
import os
import logging
from langchain_groq import ChatGroq
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.chains import load_summarize_chain
from langchain.docstore.document import Document
from langchain_community.utilities import GoogleSerperAPIWrapper
from dotenv import load_dotenv
import json  # Add this import for JSON serialization

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Suppress TensorFlow logging
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

# Crop selection
st.title("Crop Disease Detection")
crop = st.selectbox("Select a crop:", ["Cotton", "Rice", "Wheat"])

# Load the ML model based on the selected crop
@st.cache_resource
def load_ml_model(crop):
    try:
        logger.info(f"Loading the model for {crop}...")
        model_paths = {
            "Cotton": r'',#your model path here,
            "Rice": r'', #your model path here,
            "Wheat": r'' #your model path here
        }
        model_path = model_paths.get(crop)
        if not model_path or not os.path.exists(model_path):
            logger.error(f"Model file not found for {crop} at: {model_path}")
            return None

        # Load Keras model for all crops with compile=False to suppress warnings
        model = keras.models.load_model(model_path, compile=False)
        logger.info(f"{crop} model loaded successfully!")
        return model
    except Exception as e:
        logger.error(f"Error loading {crop} model: {str(e)}")
        return None

def predict_with_model(model, crop, img_array):
    # Keras model prediction for all crops
    prediction = model.predict(img_array)
    return prediction

# Initialize the selected crop's model
model = load_ml_model(crop)
if model is None:
    st.error(f"Failed to load the model for {crop}. Please check the logs.")
else:
    st.success(f"{crop} model loaded successfully! Proceed with the analysis.")

# Initialize services
@st.cache_resource
def initialize_services():
    try:
        load_dotenv()
        logger.info("Environment variables loaded")

        groq_api_key = os.getenv("GROQ_API_KEY")
        serper_api_key = os.getenv("SERPER_API_KEY")
        
        if not groq_api_key or not serper_api_key:
            logger.error("Missing API keys!")
            return None, None, None

        llm = ChatGroq(model="llama-3.2-3b-preview", api_key=groq_api_key)
        text_splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=50)
        summarize_chain = load_summarize_chain(llm, chain_type="map_reduce")
        logger.info("Services initialized successfully!")
        return llm, text_splitter, summarize_chain
    except Exception as e:
        logger.error(f"Error initializing services: {str(e)}")
        return None, None, None

# Define class labels for each crop
class_labels_map = {
    "Cotton": ['Bacterial Blight', 'Leaf Curl Virus', 'Wilt Disease'],
    "Rice": ['Bacterial Blight Disease', 'Blast Disease', 'Brown Spot Disease', 'False Smut Disease'],
    "Wheat": ['Rust Disease', 'Powdery Mildew', 'Fusarium Head Blight']
}

def preprocess_image(image_bytes, crop):
    logger.info(f"Preprocessing image for {crop}...")
    img = Image.open(io.BytesIO(image_bytes))
    if crop == "Wheat":
        # Resize specifically for the wheat model's expected input
        img = img.resize((64, 64))  # Adjust to the wheat model's expected input size
    elif crop == "Cotton":
        # Resize specifically for the cotton ResNet model's expected input
        img = img.resize((224, 224))  # Adjust to the cotton ResNet model's expected input size
    else:
        # Default resizing for other models
        img = img.resize((224, 224))
    img_array = np.array(img) / 255.0
    img_array = np.expand_dims(img_array, axis=0)  # Add batch dimension
    return img_array

def llama_summarize(text, text_splitter, summarize_chain):
    texts = text_splitter.split_text(text)
    docs = [Document(page_content=t) for t in texts]
    summary = summarize_chain.run(docs)
    return summary

def search_and_summarize(disease_name, text_splitter, summarize_chain):
    logger.info(f"Getting information for disease: {disease_name}")
    search = GoogleSerperAPIWrapper(api_key=os.getenv("SERPER_API_KEY"))
    
    causes_query = f"List the plausible symptoms of {disease_name} in crops"
    treatments_query = f"List the relevant fertilizers, herbicides, insecticides, or pesticides for treating {disease_name}"
    cure_query = f"List the ways to prevent {disease_name} in crops"
    
    causes_results = search.run(causes_query)
    treatments_results = search.run(treatments_query)
    cure_results = search.run(cure_query)
    
    summarized_causes = llama_summarize(causes_results, text_splitter, summarize_chain)
    summarized_treatments = llama_summarize(treatments_results, text_splitter, summarize_chain)
    summarized_cure = llama_summarize(cure_results, text_splitter, summarize_chain)
    
    return {
        "symptoms": summarized_causes.split("\n"),
        "treatments": summarized_treatments.split("\n"),
        "prevention": summarized_cure.split("\n")
    }

def format_json_as_list(json_data):
    """Format JSON data as a list with a charcoal grey background for readability."""
    try:
        if isinstance(json_data, list):
            list_items = "\n".join([f"<li>{item}</li>" for item in json_data])
        elif isinstance(json_data, dict):
            list_items = "\n".join([f"<li><strong>{key}</strong>: {value}</li>" for key, value in json_data.items()])
        else:
            list_items = f"<li>{str(json_data)}</li>"
        
        return f"""
        <div style='background-color: #333333; color: white; padding: 10px; border-radius: 5px;'>
            <ul style='list-style-type: disc; padding-left: 20px;'>
                {list_items}
            </ul>
        </div>
        """
    except Exception as e:
        logger.error(f"Error formatting JSON data as list: {str(e)}")
        return "<p style='color: red;'>Error displaying the information.</p>"

def apply_custom_styles():
    """Apply custom CSS styles for the entire webpage."""
    st.markdown(
        """
        <style>
        body {
            background-color: #2c2f33;
            color: #ffffff;
        }
        h1 {
            color: #7289da;
            text-align: center;
        }
        .stFileUploader {
            background-color: #23272a;
            border: 1px solid #7289da;
            border-radius: 5px;
            padding: 10px;
        }
        .st-expander {
            background-color: #23272a;
            border: 1px solid #7289da;
            border-radius: 5px;
        }
        .st-expander-content {
            background-color: #333333;
            color: #ffffff;
        }
        .stButton > button {
            background-color: #7289da;
            color: #ffffff;
            border: none;
            border-radius: 5px;
            padding: 10px 20px;
        }
        .stButton > button:hover {
            background-color: #5b6eae;
        }
        </style>
        """,
        unsafe_allow_html=True
    )

# Streamlit UI
apply_custom_styles()  # Apply the custom styles

if model is not None:
    class_labels = class_labels_map.get(crop, [])
    uploaded_file = st.file_uploader(f"Upload an image of the {crop} crop", type=["jpg", "jpeg", "png"])
    
    if uploaded_file is not None:
        image_bytes = uploaded_file.read()
        img_array = preprocess_image(image_bytes, crop)  # Pass the crop name to preprocessing
        
        prediction = predict_with_model(model, crop, img_array)
        predicted_class_index = np.argmax(prediction)
        predicted_class_label = class_labels[predicted_class_index]
        
        st.write(f"Predicted Disease: **{predicted_class_label}**")
        st.write(f"Confidence: **{float(prediction[0][predicted_class_index]):.2f}**")
        
        with st.spinner("Fetching additional information..."):
            llm, text_splitter, summarize_chain = initialize_services()
            if llm and text_splitter and summarize_chain:
                disease_info = search_and_summarize(predicted_class_label, text_splitter, summarize_chain)
                
                with st.expander("Symptoms"):
                    st.markdown(format_json_as_list(disease_info['symptoms']), unsafe_allow_html=True)
                
                with st.expander("Treatments"):
                    st.markdown(format_json_as_list(disease_info['treatments']), unsafe_allow_html=True)
                
                with st.expander("Prevention"):
                    st.markdown(format_json_as_list(disease_info['prevention']), unsafe_allow_html=True)
            else:
                st.error("Failed to initialize services for fetching additional information.")
