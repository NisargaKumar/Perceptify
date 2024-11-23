# Perceptify - AI for Visually Impaired

Perceptify is an AI-powered platform designed to assist visually impaired individuals with real-time scene understanding, text-to-speech conversion, object detection, and personalized assistance for daily tasks.

## Features

1. **Real-Time Scene Understanding 👁️**
   - AI identifies and describes objects in the environment.

2. **Text-to-Speech Conversion 🎙️**
   - Converts extracted text from images into speech.

3. **Object Detection 🔍**
   - Detects and classifies objects in uploaded images.

4. **Personalized Assistance 🤖**
   - Users can ask questions and get answers based on the image content.

## Installation & Setup

1. Clone the repository:
    ```bash
    git clone https://github.com/yourusername/perceptify.git
    cd perceptify
    ```

2. Install dependencies:
    ```bash
    pip install -r requirements.txt
    ```

3. Set up Google Cloud credentials and API keys for Vision and Text-to-Speech APIs.

4. Run the Streamlit app:
    ```bash
    streamlit run app.py
    ```

## Project Structure

perceptify/
├── app.py                 # Main Streamlit app
├── requirements.txt       # Dependencies
├── config/                # Google Cloud credentials
└── README.md              # Documentation

## API Usage

- **Scene Understanding**: Upload an image for scene analysis.
- **Text-to-Speech**: Upload an image containing text for speech conversion.
- **Object Detection**: Upload an image for object detection and classification.
- **Personalized Assistance**: Ask questions based on the uploaded image.
