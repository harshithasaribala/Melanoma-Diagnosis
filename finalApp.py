import streamlit as st
import numpy as np
import cv2
from pydicom import dcmread
from tensorflow.keras.models import load_model
from skimage.util import random_noise
from skimage import exposure
import tempfile

# Load DICOM image
def load_dicom(path):
    dicom = dcmread(path, force=True)  # Use `force=True` to handle malformed headers
    img = dicom.pixel_array
    return img, dicom

# Convert to grayscale
def convert_to_gray(img):
    gray_img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY) if len(img.shape) == 3 else img
    return gray_img

# Resize image
def resize_image(img, size=(224, 224)):
    resized_img = cv2.resize(img, size, interpolation=cv2.INTER_AREA)
    return resized_img

# Add noise
def add_noise(img):
    noisy_img = random_noise(img, mode='gaussian', var=0.01)
    noisy_img = (255 * noisy_img).astype(np.uint8)  # Scale back to 0-255
    return noisy_img

# Enhance contrast
def enhance_image(img):
    enhanced_img = exposure.equalize_adapthist(img, clip_limit=0.03)  # CLAHE for contrast
    enhanced_img = (255 * enhanced_img).astype(np.uint8)
    return enhanced_img

# Display images in a horizontal layout
def display_images(original, gray, resized, noisy, enhanced):
    st.write("### Preprocessed Images")
    col1, col2, col3, col4, col5 = st.columns(5)
    with col1:
        st.image(original, caption="Original", use_column_width=True, clamp=True)
    with col2:
        st.image(gray, caption="Gray", use_column_width=True, clamp=True)
    with col3:
        st.image(resized, caption="Resized", use_column_width=True, clamp=True)
    with col4:
        st.image(noisy, caption="Noisy", use_column_width=True, clamp=True)
    with col5:
        st.image(enhanced, caption="Enhanced", use_column_width=True, clamp=True)

# Load and preprocess DICOM image
def preprocess_dicom(file_path):
    original_img, dicom_data = load_dicom(file_path)
    gray_img = convert_to_gray(original_img)
    resized_img = resize_image(gray_img)
    noisy_img = add_noise(resized_img)
    enhanced_img = enhance_image(resized_img)
    return original_img, gray_img, resized_img, noisy_img, enhanced_img, dicom_data

# Load and normalize DICOM image for model prediction
def load_dicom_image(file_path, input_shape=(224, 224)):
    dicom_data = dcmread(file_path, force=True)
    dicom_img = dicom_data.pixel_array
    img = cv2.resize(dicom_img, input_shape)
    if len(img.shape) == 2:  # If grayscale, convert to 3-channel
        img = np.stack((img,) * 3, axis=-1)
    img = img / 255.0  # Normalize
    return img, dicom_data

# Primary prediction
def predict_image(model, file_path):
    img, dicom_data = load_dicom_image(file_path)
    img = np.expand_dims(img, axis=0)
    prediction = model.predict(img)
    pred_label = np.argmax(prediction, axis=1)[0]
    confidence = prediction[0][pred_label]
    class_names = ["Benign", "Malignant"]
    main_class_name = class_names[pred_label]
    return main_class_name, confidence, dicom_data

# Secondary prediction for Benign cases
def secondary_predict_image(model, file_path):
    img, _ = load_dicom_image(file_path)
    img = np.expand_dims(img, axis=0)
    secondary_prediction = model.predict(img)
    sub_label = np.argmax(secondary_prediction, axis=1)[0]
    sub_class_names = ["Nevus", "Unknown"]
    secondary_class_name = sub_class_names[sub_label]
    return secondary_class_name

# Streamlit Interface
st.title("Melanoma Diagnosis System")
st.write("Upload a DICOM image to diagnose melanoma and view detailed insights.")

# File uploader
uploaded_file = st.file_uploader("Upload a DICOM Image", type=["dcm"])

if uploaded_file is not None:
    with tempfile.NamedTemporaryFile(delete=False, suffix=".dcm") as temp_dicom_file:
        temp_dicom_file.write(uploaded_file.read())
        temp_file_path = temp_dicom_file.name

    with st.spinner("Processing the DICOM image..."):
        try:
            # Preprocess and display images
            # Load models
            efficientnet_model_path = r"E:\KARE\Capstone Project\Melanoma Daignosis\Models\efficientnet_model.h5"
            secondary_model_path = r"E:\KARE\Capstone Project\Melanoma Daignosis\Models\secondary_model22.h5"
            efficientnet_model = load_model(efficientnet_model_path)
            secondary_model = load_model(secondary_model_path)
            result, confidence, dicom_metadata = predict_image(efficientnet_model, temp_file_path)
            
            # Display metadata
            st.write(f"**Patient ID:** {dicom_metadata.PatientID if 'PatientID' in dicom_metadata else 'N/A'}")
            st.write(f"**Patient Age:** {dicom_metadata.PatientAge if 'PatientAge' in dicom_metadata else 'N/A'}")
            st.write(f"**Patient Sex:** {dicom_metadata.PatientSex if 'PatientSex' in dicom_metadata else 'N/A'}")

            # Predict using the primary model
            
            original_img, gray_img, resized_img, noisy_img, enhanced_img, dicom_data = preprocess_dicom(temp_file_path)
            display_images(original_img, gray_img, resized_img, noisy_img, enhanced_img)

            st.write(f"### Primary Diagnosis: **{result}**")

            # Secondary prediction for benign cases
            if result == "Benign":
                sub_diagnosis = secondary_predict_image(secondary_model, temp_file_path)
                st.write(f"### Sub Diagnosis: **{sub_diagnosis}**")
            else:
                st.write("***No further classification needed for Malignant cases***")

        except Exception as e:
            st.error(f"An error occurred: {e}")
