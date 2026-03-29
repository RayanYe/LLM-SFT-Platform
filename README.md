# 🤖 LLM-SFT-Platform

> A minimal yet extensible visual platform for fine-tuning large
> language models (LLMs) with an intuitive UI.

------------------------------------------------------------------------

## 📌 Overview

LLM-SFT-Platform is a lightweight end-to-end system that enables users
to:

-   Upload and validate datasets\
-   Configure fine-tuning parameters\
-   Launch training jobs with a single click\
-   Monitor training progress in real time\
-   Interact with fine-tuned models

This project is designed to **lower the barrier of LLM fine-tuning**,
especially for users without deep ML engineering experience.

------------------------------------------------------------------------

## 🚀 Features

### 📂 Dataset Management

-   Upload `.jsonl` datasets via UI
-   Automatic validation

### ⚙️ Training Configuration

-   Select base model (Qwen / LLaMA / Mistral)
-   Adjustable hyperparameters

### 📊 Monitoring Dashboard

-   Real-time training status
-   Loss curve visualization

### 💬 Inference Interface

-   Prompt input panel
-   Output display (placeholder)

------------------------------------------------------------------------

## 🏗️ Architecture

Frontend (Streamlit) → Backend (FastAPI) → Training Engine

------------------------------------------------------------------------

## 📁 Project Structure

LLM-SFT-Platform/ ├── backend/ ├── frontend/ ├── data/ ├── outputs/ ├──
requirements.txt └── README.md

------------------------------------------------------------------------

## ⚡ Getting Started

### Install dependencies

pip install -r requirements.txt

### Run backend

uvicorn backend.app:app --reload

### Run frontend

streamlit run frontend/streamlit_app.py

------------------------------------------------------------------------

## 📄 Dataset Format

Each line must be JSON:

{"instruction": "...", "input": "...", "output": "..."}

------------------------------------------------------------------------

## 👤 Author

Rayan Ye
