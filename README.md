# End-to-End Salary Prediction (Modular ML Pipeline)

## Overview
This is a production-grade Machine Learning project designed to predict salaries based on years of experience. The core strength of this project is its **Modular Architecture**, making it scalable and easy to maintain.

## Project Structure
- **Data Ingestion**: Handles reading data from sources and splitting into Train/Test sets.
- **Data Transformation**: Uses Scikit-Learn pipelines for scaling and feature engineering.
- **Model Trainer**: Evaluates multiple algorithms and exports the best performing model.
- **Web Interface**: A Flask-based web app for real-time user predictions.

## Tech Stack
- **Language**: Python 3.8+
- **ML Libraries**: Pandas, Numpy, Scikit-Learn
- **Web Framework**: Flask
- **Logging/Exception**: Custom components for robust error handling.

## How to Run
1. Install requirements: `pip install -r requirements.txt`
2. Run the pipeline: `python src/components/data_ingestion.py`
3. Start the Web App: `python app.py`