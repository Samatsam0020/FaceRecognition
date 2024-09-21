# 🤖 Face Recognition App

Welcome to the Face Recognition App! This project utilizes facial recognition technology to identify known faces and count the number of people in an image. It’s built with Python and leverages various libraries to provide a seamless user experience.

## Features

- 🕵️‍♂️ **Face Recognition:** Identify faces from a database of known individuals.
- 🔢 **Counting Individuals:** Accurately count the number of people in images.
- 🔍 **Search Functionality:** Easily search for specific faces in your dataset.

## Files

- **`app.py`:** The main Streamlit application for the user interface.
- **`count.py`:** Module for counting the number of faces in an image.
- **`extract_face.py`:** Extracts faces from images for further processing.
- **`recognize.py`:** Implements the face recognition logic.
- **`requirements.txt`:** Lists the required Python packages for the project.
- **`search_face.py`:** Tests the search functionality against the database.

## Requirements

To get started, make sure you have Python installed. You can install the required packages using pip:

```bash
pip install -r requirements.txt

#Getting Started

Run the Streamlit App: To launch the application, run:
```bash
streamlit run app.py

Test the Search Function: To test the search functionality, run:
```bash
python search_face.py

Usage

    🖼️ Face Recognition: Upload an image to recognize faces against the known database.
    📊 Count Faces: The app will automatically count the number of faces detected in the uploaded image.
    🔍 Search Faces: Use the search function to find specific individuals in your dataset.

Contact

📬 For any inquiries, feel free to reach out to me at [ssamat0020@gmail.com].
