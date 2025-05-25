import streamlit as st
import openai
from dotenv import load_dotenv
import os
from PIL import Image
import pytesseract



# Update this path if your Tesseract is installed somewhere else
pytesseract.pytesseract.tesseract_cmd = r"C:\Program Files\Tesseract-OCR\tesseract.exe"



# Load API key from .env file
load_dotenv()
openai.api_key = openai.api_key = os.getenv("OPENAI_API_KEY")

# Streamlit page config
st.set_page_config(page_title="Visual Medical Assistant", page_icon="🩺", layout="wide")
st.title("Visual Medical Assistant 👨‍⚕️ 🩺 🏥")
st.subheader("Upload a medical report image and get a GPT-based analysis.")

# File upload
file_uploaded = st.file_uploader("Upload Medical Report (image only)", type=["png", "jpg", "jpeg"])

if file_uploaded:
    st.image(file_uploaded, caption="Uploaded Medical Report", width=300)

submit = st.button("Analyze Report")

if submit and file_uploaded:
    with st.spinner("Extracting text and analyzing..."):
        # Load and extract text from image
        image = Image.open(file_uploaded)
        extracted_text = pytesseract.image_to_string(image)

        if not extracted_text.strip():
            st.error("No readable text found in the image.")
        else:
            # Prepare GPT prompt
            prompt = f"""
You are a highly knowledgeable medical expert. Your task is to analyze the following extracted text from a medical report.

Provide the response under the following four headings:
1. Detailed Analysis
2. Summary Report
3. Recommendations
4. Treatments

Also include this disclaimer at the end: "Consult with a Doctor before making any decisions."

Extracted Medical Report Text:
\"\"\"
{extracted_text}
\"\"\"
"""

            try:
                # OpenAI GPT API call
                response = openai.ChatCompletion.create(
                    model="gpt-3.5-turbo",
                    messages=[
                        {"role": "system", "content": "You are a professional medical assistant."},
                        {"role": "user", "content": prompt}
                    ]
                )

                result = response["choices"][0]["message"]["content"]
                st.success("Analysis completed successfully.")
                st.markdown(result)

            except Exception as e:
                st.error(f"Something went wrong: {e}")
