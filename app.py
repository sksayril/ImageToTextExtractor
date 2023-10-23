import streamlit as st
from paddleocr import PaddleOCR
from PIL import Image
import numpy as np
from textblob import TextBlob

# Initialize the OCR model
ocr = PaddleOCR()

# Define the Streamlit app
st.title("Image Text Extractor")

# Upload image through Streamlit
uploaded_image = st.file_uploader("Upload an image", type=["jpg", "png", "jpeg"])

if uploaded_image:
    # Preprocess the uploaded image
    image = Image.open(uploaded_image)
    
    # Convert the image to a numpy array
    image_np = np.array(image)

    # Add a loader while performing OCR
    with st.spinner('Extracting text...'):
        # Perform OCR on the preprocessed image
        result = ocr.ocr(image_np)

    # Display all extracted text as a template
    st.subheader("Extracted Text:")
    extracted_text = ""
    for item in result[0]:
        text = item[1]
        if text:
            extracted_text += f'{text[0]}\n'
    st.text_area("Extracted Text:", value=extracted_text, height=300)

    # Correct grammatical mistakes using TextBlob
    text_blob = TextBlob(extracted_text)
    corrected_text = text_blob.correct()

    # Display corrected text
    st.subheader("Corrected Text:")
    st.text_area("Corrected Text:", value=str(corrected_text), height=300)

# You can add more Streamlit components as needed
