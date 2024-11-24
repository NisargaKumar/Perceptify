import os
import streamlit as st
from google.cloud import vision
import tempfile
import requests
from PIL import Image
from io import BytesIO
from google.cloud import texttospeech
import pyttsx3
import google.generativeai as genai
from langchain_google_genai import GoogleGenerativeAI
from gtts import gTTS

os.environ["GOOGLE_APPLICATION_CREDENTIALS"] = "config/your-google-cloud-vision.json" #Your gcloud vision file here

GEMINI_API_KEY = ""  #Your gemini API Key Here
os.environ["GOOGLE_API_KEY"] = GEMINI_API_KEY

llm = GoogleGenerativeAI(model="gemini-1.5-pro", api_key=GEMINI_API_KEY)

engine = pyttsx3.init()

st.set_page_config(
    page_title="Perceptify - AI for Visually Impaired",
    page_icon="🌍", 
    layout="wide", 
)

#Sidebar
st.sidebar.title("Features 🌟")
options = st.sidebar.radio(
    "Select a Feature:",
    ("Home 🏠", "Real-Time Scene Understanding 👁️", "Text-to-Speech Conversion 🎙️", "Object Detection 🔍", "Personalized Assistance 🤖")
)

client = vision.ImageAnnotatorClient()

def text_to_speech(text, language='en'):
    tts = gTTS(text=text, lang=language)
    tts.save("output.mp3")
    return "output.mp3"

def analyze_image(image_path):
    with open(image_path, 'rb') as image_file:
        content = image_file.read()
    image = vision.Image(content=content)

    response = client.text_detection(image=image)
    texts = response.text_annotations

    if not texts:
        response = client.object_localization(image=image)
        objects = response.localized_object_annotations
        detected_objects = [obj.name for obj in objects]
        return detected_objects
    else:

        extracted_text = texts[0].description
        return extracted_text

language_options = {
    "en": "English",
    "es": "Spanish",
    "fr": "French",
    "de": "German",
    "hi": "Hindi",
    "it": "Italian",
    "ja": "Japanese",
    "ko": "Korean",
    "pt": "Portuguese",
    "zh": "Chinese",
}

def home_page():
        st.title("Perceptify 🌍")
        st.subheader("AI-powered Solutions for the Visually Impaired 💡")
        st.write(
            """
            Welcome to **Perceptify**, an innovative platform that uses AI to help visually impaired individuals navigate the world around them. 
            Our powerful tools include real-time scene understanding, text-to-speech conversion, object detection, and personalized assistance for daily tasks.

            - **Real-Time Scene Understanding 👁️**: Understand your surroundings with AI-powered scene analysis.
            - **Text-to-Speech Conversion 🎙️**: Convert text from images into speech for easier comprehension.
            - **Object Detection 🔍**: Detect and identify objects in your environment.
            - **Personalized Assistance 🤖**: Get dynamic information about detected objects and enhance your daily tasks.

            Select a feature from the sidebar to begin your journey with Perceptify! 🚀
            """
        )

# Home page
if options == "Home 🏠":
    home_page()

# 1 - Real-Time Scene Understanding
if options == "Real-Time Scene Understanding 👁️":
    st.header("Real-Time Scene Understanding 👁️")
    st.write("This feature helps you analyze images in real-time and understand the content using AI.")

    uploaded_file = st.file_uploader("Upload an Image", type=["png", "jpg", "jpeg"])
    image_url = st.text_input("Or enter an Image URL:")

    if uploaded_file:
        st.image(uploaded_file, caption="Uploaded Image", use_container_width=True)
        with tempfile.NamedTemporaryFile(delete=False, suffix=".jpg") as temp_file:
            temp_file.write(uploaded_file.getbuffer())
            temp_image_path = temp_file.name
    elif image_url:
        response = requests.get(image_url)
        img = Image.open(BytesIO(response.content))
        st.image(img, caption="Image from URL", use_container_width=True)
        with tempfile.NamedTemporaryFile(delete=False, suffix=".jpg") as temp_file:
            img.save(temp_file, format="JPEG")
            temp_image_path = temp_file.name
    else:
        st.warning("Please upload an image or provide an image URL.")
        temp_image_path = None

    if temp_image_path:
        def analyze_image_for_labels(image_path):
            client = vision.ImageAnnotatorClient()
            with open(image_path, 'rb') as image_file:
                content = image_file.read()
            image = vision.Image(content=content)
            response = client.label_detection(image=image)
            labels = response.label_annotations
            label_text = ""
            if labels:
                st.write("Labels detected in the image 📜:") 
                for label in labels:
                    label_text += f"- {label.description} (confidence: {label.score:.2f})\n"
                st.write(label_text)
                return label_text
            else:
                st.write("No labels detected in the image 🚫.")
                return None

        label_text = analyze_image_for_labels(temp_image_path)

        input_prompt = """
        You are an AI assistant helping visually impaired individuals by describing the scene in the image. Provide:
        1. List of items detected in the image with their purpose.
        2. Overall description of the image.
        3. Suggestions for actions or precautions for the visually impaired.
        """

        if label_text:
            try:
                response = llm.generate(
                    prompts=[f"{input_prompt}\n{label_text}"],  
                    temperature=0.7,  
                    max_tokens=500  
                )

                generated_text = response.generations[0][0].text 

                st.subheader("AI Generated Scene Description (English)")
                st.write(generated_text)

                audio_file_path = text_to_speech(generated_text)
                audio_file = open(audio_file_path, "rb")
                st.write("Audio description of the scene:")
                st.audio(audio_file, format="audio/mp3")

            except Exception as e:
                st.error(f"Error generating text: {e}")

        language_name = st.selectbox("Select Language for Description", list(language_options.values())) 

        language_code = {v: k for k, v in language_options.items()}.get(language_name)

        if language_code:
            try:

                translated_text = llm.generate(
                    prompts=[f"Translate the following text into {language_name}:\n{generated_text}"],
                    temperature=0.5,
                    max_tokens=200
                ).generations[0][0].text

                st.write(f"Translated Description ({language_name}):")
                st.write(translated_text)

                audio_file_path_translated = text_to_speech(translated_text, language_code)
                audio_file_translated = open(audio_file_path_translated, "rb")
                st.audio(audio_file_translated, format="audio/mp3")

            except Exception as e:
                st.error(f"Error generating translation: {e}")

#2 - Text-to-Speech Conversion
if options == "Text-to-Speech Conversion 🎙️":
    st.header("Text-to-Speech Conversion 🎙️")
    st.write("Upload an image and we'll convert any text within the image into speech, and guide you through it.")

    uploaded_file = st.file_uploader("Upload an Image ", type=["png", "jpg", "jpeg"])

    if uploaded_file is None:
        image_url = st.text_input("Or, paste an image URL here 🌐:")
    else:
        image_url = None

    if uploaded_file:
        st.image(uploaded_file, caption="Uploaded Image ", use_container_width=True)
        with tempfile.NamedTemporaryFile(delete=False, suffix=".jpg") as temp_file:
            temp_file.write(uploaded_file.getbuffer())
            temp_image_path = temp_file.name

    elif image_url:
        try:
            response = requests.get(image_url)
            img = Image.open(BytesIO(response.content))
            st.image(img, caption="Image from URL 🌐", use_container_width=True)
            with tempfile.NamedTemporaryFile(delete=False, suffix=".jpg") as temp_file:
                img.save(temp_file, format="JPEG")
                temp_image_path = temp_file.name
        except Exception as e:
            st.error(f"Failed to load the image from URL: {e}")
    else:
        st.warning("Please upload an image or provide a valid URL.")

    if uploaded_file or image_url:
        def extract_text_from_image(image_path):
            client = vision.ImageAnnotatorClient()
            with open(image_path, 'rb') as image_file:
                content = image_file.read()
            image = vision.Image(content=content)
            response = client.text_detection(image=image)
            texts = response.text_annotations
            if texts:
                extracted_text = texts[0].description
                st.write(f"Extracted Text 📝: {extracted_text}\n")
                return extracted_text
            else:
                st.write("No text detected in the image 🚫.")
                return None

        extracted_text = extract_text_from_image(temp_image_path)

        input_prompt = f"""
        You are an AI assistant that helps visually impaired individuals by providing a guide to the extracted text. 
        Summarize the text into a brief and understandable format.
        """

        if extracted_text:
            try:
                response = llm.generate(
                    prompts=[f"{input_prompt}\n{extracted_text}"],
                    temperature=0.7,
                    max_tokens=150
                )
                summarized_text = response.generations[0][0].text

                st.subheader("AI Generated Text Summary")
                st.write(summarized_text)

                audio_file_path = text_to_speech(summarized_text)
                audio_file = open(audio_file_path, "rb")
                st.write("Audio Summary:")
                st.audio(audio_file, format="audio/mp3")

            except Exception as e:
                st.error(f"Error generating summary: {e}")

#3 - Object Detection 
if options == "Object Detection 🔍":
    st.header("Object Detection 🔍")
    st.write("Detects objects in images and provides a description and suggestions.")

    uploaded_file = st.file_uploader("Upload an Image", type=["png", "jpg", "jpeg"])

    if uploaded_file is None:
        image_url = st.text_input("Or, paste an image URL here 🌐:")
    else:
        image_url = None

    if uploaded_file:
        st.image(uploaded_file, caption="Uploaded Image", use_container_width=True)

        with tempfile.NamedTemporaryFile(delete=False, suffix=".jpg") as temp_file:
            temp_file.write(uploaded_file.getbuffer())
            temp_image_path = temp_file.name

    elif image_url:
        try:
            response = requests.get(image_url)
            img = Image.open(BytesIO(response.content))
            st.image(img, caption="Image from URL 🌐", use_container_width=True)

            with tempfile.NamedTemporaryFile(delete=False, suffix=".jpg") as temp_file:
                img.save(temp_file, format="JPEG")
                temp_image_path = temp_file.name
        except Exception as e:
            st.error(f"Failed to load the image from URL: {e}")
    else:
        st.warning("Please upload an image or provide a valid URL.")

    if uploaded_file or image_url:
        def detect_objects_in_image(image_path):
            client = vision.ImageAnnotatorClient()
            with open(image_path, 'rb') as image_file:
                content = image_file.read()
            image = vision.Image(content=content)
            response = client.object_localization(image=image)
            objects = response.localized_object_annotations
            if objects:
                detected_objects = [obj.name for obj in objects]
                st.write(f"Objects Detected in the image 🏷️: {', '.join(detected_objects)}\n")
                return detected_objects
            else:
                st.write("No objects detected in the image 🚫.")
                return None

        detected_objects = detect_objects_in_image(temp_image_path)

        input_prompt = f"""
        You are an AI assistant helping visually impaired individuals by describing detected objects. 
        Please describe the following objects in the image and suggest what actions to take.
        """

        if detected_objects:
            try:

                response = llm.generate(
                    prompts=[f"{input_prompt}\n{', '.join(detected_objects)}"],  
                    temperature=0.7,  
                    max_tokens=500  
                )

                st.subheader("AI Generated Object Description")
                st.write(response.generations[0][0].text)

                audio_file_path = text_to_speech(response.generations[0][0].text)
                audio_file = open(audio_file_path, "rb")
                st.audio(audio_file, format="audio/mp3")
            except Exception as e:
                st.error(f"Error generating text: {e}")

#4 - Personalized Assistance
if options == "Personalized Assistance 🤖":
    st.header("Personalized Assistance 🤖")
    st.write("This feature provides personalized assistance based on the content of the uploaded image or URL. You can ask questions related to the image.")

    uploaded_file = st.file_uploader("Upload an image for analysis:", type=["png", "jpg", "jpeg"])
    image_url = st.text_input("Or provide an image URL 🌐:")

    if uploaded_file:
        st.image(uploaded_file, caption="Uploaded Image", use_container_width=True)
        with tempfile.NamedTemporaryFile(delete=False, suffix=".jpg") as temp_file:
            temp_file.write(uploaded_file.getbuffer())
            temp_image_path = temp_file.name
    elif image_url:
        response = requests.get(image_url)
        img = Image.open(BytesIO(response.content))
        st.image(img, caption="Image from URL", use_container_width=True)
        with tempfile.NamedTemporaryFile(delete=False, suffix=".jpg") as temp_file:
            img.save(temp_file, format="JPEG")
            temp_image_path = temp_file.name
    else:
        st.warning("Please upload an image or provide a valid image URL.")
        temp_image_path = None

    if temp_image_path:

        analysis_result = analyze_image(temp_image_path)

        if isinstance(analysis_result, str):  
            st.subheader("Extracted Text from Image:")
            st.write(analysis_result)
        else:  
            st.subheader("Detected Objects in Image:")
            for obj in analysis_result:
                st.write(f"- {obj}")

        if "conversation_history" not in st.session_state:
            st.session_state.conversation_history = []

        user_question = st.text_input("Ask a question about the image:")

        if user_question:

            st.session_state.conversation_history.append(f"User: {user_question}")

            response_text = "I'm not sure about the answer to that question, but here is some info:"

            if isinstance(analysis_result, str):  
                response_text = f"The text in the image says: {analysis_result}. Can you please specify what you'd like to know more about?"
            elif isinstance(analysis_result, list):  
                response_text = f"The image contains these objects: {', '.join(analysis_result)}. Could you clarify what you want to know about them?"

            st.session_state.conversation_history.append(f"AI: {response_text}")

            st.subheader("Answer:")
            st.write(response_text)

            audio_file_path = text_to_speech(response_text)
            audio_file = open(audio_file_path, "rb")
            st.audio(audio_file, format="audio/mp3")

        if st.session_state.conversation_history:
            st.subheader("Conversation History:")
            for i in st.session_state.conversation_history:
                st.write(i)

            follow_up_question = st.text_input("Ask a follow-up question:")

            if follow_up_question:

                st.session_state.conversation_history.append(f"User: {follow_up_question}")

                follow_up_response = "I'm still processing the image. Could you please ask more specifically?"

                if isinstance(analysis_result, str):
                    follow_up_response = f"The text extracted was: {analysis_result}. What else would you like to know?"
                elif isinstance(analysis_result, list):
                    follow_up_response = f"The image contains: {', '.join(analysis_result)}. What would you like to ask about these?"

                st.session_state.conversation_history.append(f"AI: {follow_up_response}")

                st.subheader("Follow-up Answer:")
                st.write(follow_up_response)

                audio_file_path = text_to_speech(follow_up_response)
                audio_file = open(audio_file_path, "rb")
                st.audio(audio_file, format="audio/mp3")

#Footer
footer_text = """
    <div style="text-align: center; padding: 10px; font-size: 14px; color: #333;">
        Developed by <strong>Nisarga K</strong> | &copy; 2024 | 
        <a href="https://www.innomatics.in/" target="_blank" style="color: #007BFF;">Innomatics Research Labs</a>
    </div>
"""
st.markdown(footer_text, unsafe_allow_html=True)
