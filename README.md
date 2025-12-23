1. Project Overview

    This project implements an electricity load forecasting system using a combination of:

        Statistical models (ARIMA)

        Deep learning models (LSTM)

        Deep Reinforcement Learning (Recurrent PPO)

    The system includes:

        Model training notebooks

        Evaluation pipelines

        A Streamlit-based interactive dashboard for visualization and analysis

    The solution supports research evaluation (RQ1–RQ3) and real-time simulation for smart grid decision support.

2. Project Structure

SmartGridDRL/
│
├── data/
│   ├── cleaned/               # Preprocessed datasets
│   ├── results/               # Saved predictions (.npy)
│   └── models/                # Trained model files
│
├── notebooks/
│   ├── 01_eda.ipynb
│   ├── 02_baselines.ipynb
│   ├── 03_drl_training.ipynb
│   └── 04_external_feature_experiment.ipynb
│   └── 05_computational_feasibility.ipynb
│
├── reports/
│   ├── tables / baseline_results_DE.csv
│   ├── tables / rq2_external_feature_comparison.csv
│   └── tables / rq3_computational_feasibility.csv
│
├── dashboard/
│   └── app.py                 # Streamlit dashboard
│
├── requirements.txt
└── README.md

3. System Requirements

    3.1 Software Requirements

        Operating System: Windows / macOS / Linux

        Python Version: Python 3.10 or 3.11 (recommended)

    3.2 Python Libraries

        Main libraries used:

            numpy

            pandas

            matplotlib

            scikit-learn

            torch

            statsmodels

            gymnasium

            stable-baselines3

            streamlit

            streamlit-autorefresh

4. Environment Setup (Local Execution)

    Step 1: Install Dependencies

        Navigate to the project root:

            cd SmartGridDRL


        Install required packages:

            pip install -r requirements.txt

5. Running the Experiments (Notebooks)

    Open Jupyter Notebook:

        jupyter notebook


    Run notebooks in the following order:

        01_eda.ipynb

        02_baselines.ipynb

        03_drl_training.ipynb

        04_external_feature_experiment.ipynb

        05_computational_feasibility.ipynb

    These notebooks will:

        Train models

        Generate predictions

        Save outputs to data/results/

        Export evaluation CSVs to reports/

⚠️ Important: The dashboard depends on these generated files.

6. Running the Dashboard Locally

    Step 1: Navigate to Dashboard Folder
        cd dashboard

    Step 2: Launch Streamlit App
        streamlit run app.py

    The dashboard will be available at:

        http://localhost:8501

