# 🤖 LLM Fine-tuning Platform

A full-stack platform for dataset processing, model fine-tuning, evaluation, and interactive analysis using Large Language Models (LLMs).

---

## 📌 Overview

This project provides an end-to-end workflow for:

* Uploading and validating datasets
* Automatically splitting datasets into train/validation/test
* Fine-tuning LLMs (LoRA / QLoRA / full)
* Monitoring training progress
* Evaluating model performance
* Interacting with an AI assistant for analysis and suggestions

It is designed to be **reproducible, modular, and easy to run**.

---

## 🧱 Project Structure

```
.
├── backend/              # FastAPI backend
│   ├── app.py
│   └── core/
│       └── trainer.py
├── frontend/             # Streamlit UI
│   └── streamlit_app.py
├── script/               # Utility scripts
│   └── download_assistant_model.py
├── data/                 # Dataset storage (ignored in Git)
├── models_cache/         # Local model cache (ignored in Git)
├── outputs/              # Training outputs (ignored)
├── requirements.txt
├── README.md
└── .gitignore
```

---

## ⚙️ Installation

### 1. Clone the repository

```bash
git clone https://github.com/RayanYe/LLM-SFT-Platform.git
cd LLM-SFT-Platform
```

### 2. Create environment

```bash
python -m venv venv
source venv/bin/activate   # Mac/Linux
```

### 3. Install dependencies

```bash
pip install -r requirements.txt
```

---

## 🤖 Model Setup

This project uses a local assistant model:

**Default model:** `Qwen/Qwen2.5-0.5B-Instruct`

### Option 1: Automatic download (recommended)

```bash
python script/download_assistant_model.py
```

The model will be stored in:

```
models_cache/
```

### Option 2: Use existing local model

Set environment variable:

```bash
export ASSISTANT_MODEL_PATH="/your/local/model/path"
```

---

## 🚀 Running the Project

### 1. Start backend (FastAPI)

```bash
uvicorn backend.app:app --reload
```

Backend runs at:

```
http://127.0.0.1:8000
```

---

### 2. Start frontend (Streamlit)

```bash
streamlit run frontend/streamlit_app.py
```

Frontend runs at:

```
http://localhost:8501
```

---

## 🔄 Workflow

1. Upload dataset (CSV / JSON / JSONL)
2. System validates and splits data automatically
3. Configure training parameters
4. Start fine-tuning
5. Monitor training progress
6. Evaluate model performance
7. Use AI assistant for insights

---

## 📊 Features

* ✅ Dataset validation and normalization
* ✅ Automatic train/validation/test split
* ✅ LoRA / QLoRA / full fine-tuning
* ✅ Training monitoring (loss curves)
* ✅ Evaluation metrics (accuracy, token-level)
* ✅ Error analysis with examples
* ✅ AI assistant for evaluation insights

---

## ⚠️ Notes

* `models_cache/`, `data/`, and `outputs/` are **not included in GitHub**
* Models will be downloaded automatically if not found locally
* Some models (e.g., LLaMA) may require Hugging Face access

---

## 🧠 Reproducibility

The system ensures reproducibility by:

* Using deterministic seeds
* Supporting local + remote model loading
* Providing a model download script

---

## 📌 Tech Stack

* Backend: FastAPI
* Frontend: Streamlit
* ML: PyTorch, Transformers, PEFT
* Dataset: HuggingFace Datasets

---

## 📬 Contact

For questions or issues, please open an issue or contact the author.
