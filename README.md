#!/bin/bash

# ===============================
# Perceptify - AI for Visually Impaired
# ===============================

# Welcome to **Perceptify**, an AI-powered platform designed to assist visually impaired individuals
# with real-time scene understanding, text-to-speech conversion, object detection, and personalized
# assistance for daily tasks. This script will help you set up and run the project locally.

# ===============================
# 1. Clone the Repository
# ===============================
echo "Cloning the Perceptify repository..."
git clone https://github.com/yourusername/perceptify.git
cd perceptify || exit

# ===============================
# 2. Install Dependencies
# ===============================
echo "Installing required dependencies..."
pip install -r requirements.txt

# ===============================
# 3. Set Up Google Cloud Credentials
# ===============================
# Ensure that you've set up your Google Cloud Vision and Text-to-Speech API credentials
# Create and download the JSON key file from Google Cloud Console, then set the environment variable
# for authentication as follows:

echo "Setting up Google Cloud credentials..."
export GOOGLE_APPLICATION_CREDENTIALS="config/your-google-cloud-credentials.json"

# ===============================
# 4. Set Up API Keys for Gemini (Google Generative AI)
# ===============================
# Replace with your own API key from Google Cloud for Gemini
export GOOGLE_API_KEY="your-google-api-key"

# ===============================
# 5. Run the Streamlit App
# ===============================
echo "Starting the Perceptify app with Streamlit..."
streamlit run app.py

# ===============================
# Project Structure
# ===============================
echo "
Project Structure:

perceptify/
├── app.py                 # Main app (Streamlit)
├── requirements.txt       # Python dependencies
├── config/                # Google Cloud credentials
├── README.md              # Project documentation
"

# ===============================
# Usage Instructions
# ===============================
echo "
Usage Instructions:

1. Clone the repository using 'git clone'.
2. Install the required dependencies with 'pip install -r requirements.txt'.
3. Set up Google Cloud Vision and Text-to-Speech API credentials.
4. Set your API keys for Google Gemini (Generative AI).
5. Run the app using 'streamlit run app.py'.
"
