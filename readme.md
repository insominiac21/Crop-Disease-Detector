# Crop Disease Detection Web App

This web application helps detect crop diseases using machine learning models and provides additional information about the diseases, including symptoms, treatments, and prevention methods.

## Features
- Upload an image of a crop to detect diseases.
- Provides detailed information about the disease, including symptoms, treatments, and prevention.
- Supports crops like Cotton, Rice, and Wheat.

## Installation

1. Clone the repository:
   ```bash
   git clone <repository-url>
   cd <repository-folder>
   ```

2. Install the required dependencies:
   ```bash
   pip install -r requirements.txt
   ```

3. Download the machine learning models for Cotton, Rice, and Wheat from the following link:
   [Download Models](https://drive.google.com/drive/folders/1SSPWH4hvg6U-N2vOy9dpHnBOE4KryEes?usp=drive_link)

   Place the downloaded models in the appropriate paths specified in the `load_ml_model` function in `app.py`.

4. Create a `.env` file in the root directory with the following contents:
   ```plaintext
   GROQ_API_KEY=<your_groq_api_key>
   SERPER_API_KEY=<your_serper_api_key>
   ```

   Replace `<your_groq_api_key>` and `<your_serper_api_key>` with your actual API keys.

## Running the Application

1. Navigate to the directory containing the `app.py` file:
   ```bash
   cd a:\hackathon\Imagine\models\crops\streamlit
   ```

2. Run the Streamlit app:
   ```bash
   streamlit run app.py
   ```

3. Open the provided URL in your browser to access the web app.

## Notes
- Ensure that the machine learning models for Cotton, Rice, and Wheat are placed in the appropriate paths specified in the `load_ml_model` function in `app.py`.
- The `.env` file is required to initialize external services for fetching additional information about diseases.

## License
This project is licensed under the MIT License.