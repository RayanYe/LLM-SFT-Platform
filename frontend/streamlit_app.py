from __future__ import annotations

import time
from typing import Any, Dict, List

import pandas as pd
import requests
import streamlit as st
import streamlit.components.v1 as components


# ============================================================
# Page config
# ============================================================
st.set_page_config(
    page_title="LLM Fine-tuning Platform",
    page_icon="🤖",
    layout="wide",
    initial_sidebar_state="expanded",
)


# ============================================================
# Style
# ============================================================
st.markdown(
    """
    <style>
        .block-container {
            padding-top: 3rem;
            padding-bottom: 1.2rem;
        }

        .topbar {
            padding: 1rem 1.1rem;
            border-radius: 20px;
            background: linear-gradient(135deg, rgba(57, 92, 255, 0.12), rgba(0, 180, 216, 0.10));
            border: 1px solid rgba(120, 120, 120, 0.18);
            margin-bottom: 1rem;
        }

        .topbar-title {
            margin: 0;
            font-size: 1.85rem;
            line-height: 1.15;
            font-weight: 800;
        }

        .topbar-subtitle {
            margin: 0.25rem 0 0 0;
            opacity: 0.78;
            font-size: 0.96rem;
        }

        .status-pill {
            display: inline-block;
            padding: 0.28rem 0.7rem;
            border-radius: 999px;
            font-size: 0.8rem;
            font-weight: 700;
            border: 1px solid rgba(120, 120, 120, 0.22);
            vertical-align: middle;
            margin-left: 0.35rem;
        }
        .pill-idle { background: rgba(120,120,120,0.10); }
        .pill-running { background: rgba(255,193,7,0.14); }
        .pill-success { background: rgba(46,204,113,0.14); }
        .pill-failed { background: rgba(231,76,60,0.14); }

        .section-card {
            padding: 1rem 1rem 0.95rem 1rem;
            border-radius: 18px;
            border: 1px solid rgba(120, 120, 120, 0.18);
            background: rgba(255, 255, 255, 0.03);
            box-shadow: 0 8px 24px rgba(0, 0, 0, 0.04);
            height: 100%;
        }

        .section-title {
            font-size: 1.05rem;
            font-weight: 800;
            margin-bottom: 0.75rem;
        }

        .small-note {
            font-size: 0.86rem;
            opacity: 0.75;
            margin-top: 0.4rem;
        }

        .hint-box {
            padding: 0.8rem 0.9rem;
            border-radius: 14px;
            background: rgba(58, 133, 255, 0.10);
            border: 1px solid rgba(58, 133, 255, 0.18);
            margin-top: 0.6rem;
        }

        .sidebar-step {
            padding: 0.7rem 0.85rem;
            border-radius: 14px;
            border: 1px solid rgba(120,120,120,0.14);
            background: rgba(255,255,255,0.02);
            margin-bottom: 0.6rem;
            font-size: 0.92rem;
        }

        .sidebar-step strong {
            display: block;
            margin-bottom: 0.15rem;
        }

        .divider-soft {
            height: 1px;
            background: rgba(120,120,120,0.18);
            margin: 0.9rem 0;
        }

        div[data-testid="stMetric"] {
            background: rgba(255,255,255,0.02);
            border: 1px solid rgba(120, 120, 120, 0.14);
            border-radius: 16px;
            padding: 0.7rem 0.9rem;
        }

        /* Hide the internal Streamlit trigger button */
        div[data-testid="stButton"] button.internal-ai-trigger {
            display: none !important;
        }

        /* Floating AI launcher */
        .ai-fab-wrap {
            position: fixed;
            right: 22px;
            bottom: 22px;
            z-index: 99999;
        }
        .ai-fab {
            border: none;
            border-radius: 999px;
            padding: 0.85rem 1.05rem;
            font-weight: 800;
            background: linear-gradient(135deg, #395cff, #00b4d8);
            color: white;
            cursor: pointer;
            box-shadow: 0 12px 30px rgba(0, 0, 0, 0.22);
        }
        .ai-fab:hover {
            opacity: 0.95;
        }

        .chat-bubble-user {
            background: rgba(57, 92, 255, 0.10);
            border: 1px solid rgba(57, 92, 255, 0.16);
            padding: 0.7rem 0.85rem;
            border-radius: 14px;
            margin-bottom: 0.5rem;
        }
        .chat-bubble-assistant {
            background: rgba(0, 180, 216, 0.08);
            border: 1px solid rgba(0, 180, 216, 0.14);
            padding: 0.7rem 0.85rem;
            border-radius: 14px;
            margin-bottom: 0.5rem;
        }
    </style>
    """,
    unsafe_allow_html=True,
)

DEFAULT_BACKEND_URL = "http://127.0.0.1:8000"


def init_state() -> None:
    defaults = {
        "uploaded_dataset_path": "",
        "uploaded_dataset_name": "",
        "dataset_report": None,
        "split_paths": None,
        "upload_result": None,
        "validation_result": None,
        "train_submit_result": None,
        "training_status": None,
        "last_poll_time": 0.0,
        "last_error": "",
        "inference_output": "",
        "evaluation_output": None,
        "show_instruction": False,
        "show_ai_assistant": False,
        "backend_url": DEFAULT_BACKEND_URL,
        "assistant_messages": [],
        "assistant_answer": "",
    }
    for key, value in defaults.items():
        if key not in st.session_state:
            st.session_state[key] = value


init_state()


# ============================================================
# API helpers
# ============================================================
def api_post(url: str, payload: Dict[str, Any], timeout: int = 60) -> Dict[str, Any]:
    resp = requests.post(url, json=payload, timeout=timeout)
    resp.raise_for_status()
    return resp.json()


def api_get(url: str, timeout: int = 30) -> Dict[str, Any]:
    resp = requests.get(url, timeout=timeout)
    resp.raise_for_status()
    return resp.json()


def upload_dataset_api(backend_url: str, uploaded_file) -> Dict[str, Any]:
    url = f"{backend_url}/upload-dataset"
    files = {
        "file": (
            uploaded_file.name,
            uploaded_file.getvalue(),
            uploaded_file.type or "application/octet-stream",
        )
    }
    resp = requests.post(url, files=files, timeout=60)
    resp.raise_for_status()
    return resp.json()


def validate_dataset_api(backend_url: str, dataset_path: str) -> Dict[str, Any]:
    return api_post(f"{backend_url}/validate-dataset", {"dataset_path": dataset_path})


def train_api(backend_url: str, payload: Dict[str, Any]) -> Dict[str, Any]:
    return api_post(f"{backend_url}/train", payload)


def training_status_api(backend_url: str) -> Dict[str, Any]:
    return api_get(f"{backend_url}/training-status")


def evaluate_api(backend_url: str, payload: Dict[str, Any]) -> Dict[str, Any]:
    return api_post(f"{backend_url}/evaluate", payload)


def assistant_api(backend_url: str, payload: Dict[str, Any]) -> Dict[str, Any]:
    return api_post(f"{backend_url}/assistant/chat", payload, timeout=180)


def refresh_status(backend_url: str) -> None:
    try:
        st.session_state.training_status = training_status_api(backend_url)
        st.session_state.last_poll_time = time.time()
        st.session_state.last_error = ""
    except requests.exceptions.RequestException as e:
        st.session_state.last_error = str(e)


# ============================================================
# Instruction dialog
# ============================================================
@st.dialog("instruction_dialog")
def instruction_dialog():
    st.markdown(
        """
### What is this platform for?
This is an LLM fine-tuning platform designed for non-technical users. Think of it as a "Model Training and Testing Workbench."

---

### Workflow
1. Upload Dataset  
2. System automatically checks and randomly splits data into train / validation / test  
3. Check Dataset Overview to ensure data accuracy  
4. Click Check Dataset for basic validation  
5. Set training parameters and Start Training  
6. Go to Monitor to track training status and loss curves  
7. Go to Evaluation to test model performance on validation/test sets  
8. Use the AI Assistant in the bottom-right corner to ask for advice after evaluation  

---

### Data Format Requirements
Supported formats:
- CSV
- JSON
- JSONL

CSV must contain three columns:
- instruction
- input
- output
        """
    )

    if st.button("Close", type="primary", use_container_width=True):
        st.session_state.show_instruction = False
        st.rerun()


# ============================================================
# AI Assistant dialog
# ============================================================
@st.dialog("AI Assistant")
def ai_assistant_dialog():
    st.markdown("### AI Assistant")
    st.caption("Ask about the latest evaluation result, error patterns, or next-step tuning suggestions.")

    for msg in st.session_state.assistant_messages:
        with st.chat_message(msg["role"]):
            st.markdown(msg["content"])

    user_prompt = st.chat_input("Ask something about the evaluation...")
    if user_prompt:
        st.session_state.assistant_messages.append({"role": "user", "content": user_prompt})

        eval_result = st.session_state.get("evaluation_output") or {}
        assistant_payload = {
            "messages": st.session_state.assistant_messages,
            "evaluation_context": {
                "metrics": eval_result.get("metrics", {}),
                "examples": eval_result.get("examples", [])[:3],
                "generation_config": eval_result.get("generation_config", {}),
                "model": eval_result.get("model", {}),
                "split_name": eval_result.get("split_name", ""),
                "message": eval_result.get("message", ""),
            },
            "max_new_tokens": 256,
        }

        try:
            resp = assistant_api(st.session_state.backend_url, assistant_payload)
            answer = resp.get("answer", "")
            st.session_state.assistant_messages.append({"role": "assistant", "content": answer})
            st.rerun()
        except requests.exceptions.RequestException as e:
            st.error(f"Assistant request failed: {e}")

    if st.button("Close", use_container_width=True):
        st.session_state.show_ai_assistant = False
        st.rerun()


# ============================================================
# Floating launcher
# ============================================================
def open_ai_assistant():
    st.session_state.show_ai_assistant = True

def render_floating_launcher() -> None:
    st.button(
        "💬 AI Assistant",
        key="ai_assistant_launcher",
        on_click=open_ai_assistant,
    )


# ============================================================
# Header
# ============================================================
status_payload = st.session_state.training_status or {}
status_value = status_payload.get("status", "idle")
status_class = f"pill-{status_value}" if status_value in {"idle", "running", "success", "failed"} else "pill-idle"

top_left, top_right = st.columns([0.82, 0.18], vertical_alignment="center")

with top_left:
    st.markdown(
        f"""
        <div class="topbar">
            <div>
                <div class="topbar-title">🤖 LLM Fine-tuning Platform <span class="status-pill {status_class}">{status_value.upper()}</span></div>
                <div class="topbar-subtitle">Upload, split, validate, train, evaluate, infer, and ask the assistant.</div>
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )

with top_right:
    st.write("")
    if st.button("📘 Instruction", use_container_width=True):
        st.session_state.show_instruction = True

if st.session_state.show_instruction:
    instruction_dialog()

render_floating_launcher()

if st.session_state.show_ai_assistant:
    ai_assistant_dialog()


# ============================================================
# Sidebar
# ============================================================
with st.sidebar:
    st.header("Control Panel")
    backend_url = st.text_input("Backend URL", value=st.session_state.backend_url, key="backend_url")

    st.markdown(
        '<div class="sidebar-step"><strong>Step 1 · Dataset</strong>Upload your dataset here.</div>',
        unsafe_allow_html=True,
    )
    uploaded_file = st.file_uploader(
        "Upload dataset",
        type=["csv", "json", "jsonl"],
        help="CSV must contain columns: instruction, input, output. JSON can be a list of objects or {data:[...]}.",
    )
    upload_btn = st.button("Upload to Backend", use_container_width=True)

    st.markdown('<div class="divider-soft"></div>', unsafe_allow_html=True)
    st.markdown(
        '<div class="sidebar-step"><strong>Step 2 · Training</strong>Choose the base model and training parameters, then start training.</div>',
        unsafe_allow_html=True,
    )

    MODEL_CHOICES = {
        "Qwen2.5-0.5B (Base)": "Qwen/Qwen2.5-0.5B",
        "Qwen2.5-0.5B (Instruct)": "Qwen/Qwen2.5-0.5B-Instruct",
        "Qwen2.5-1.5B": "Qwen/Qwen2.5-1.5B",
        "Qwen2.5-1.5B (Instruct)": "Qwen/Qwen2.5-1.5B-Instruct",
        "Qwen2.5-3B": "Qwen/Qwen2.5-3B",
        "Qwen2.5-3B (Instruct)": "Qwen/Qwen2.5-3B-Instruct",
        "Qwen2.5-7B": "Qwen/Qwen2.5-7B",
        "Qwen2.5-7B (Instruct)": "Qwen/Qwen2.5-7B-Instruct",
        "Llama-3.1-8B": "meta-llama/Llama-3.1-8B",
        "Llama-3.1-8B (Instruct)": "meta-llama/Llama-3.1-8B-Instruct",
        "Mistral-7B": "mistralai/Mistral-7B-v0.1",
        "Mistral-7B (Instruct)": "mistralai/Mistral-7B-Instruct-v0.3",
        "Mixtral-8x7B": "mistralai/Mixtral-8x7B-v0.1",
        "Phi-3-mini": "microsoft/Phi-3-mini-4k-instruct",
        "ChatGLM3-6B": "THUDM/chatglm3-6b",
        "InternLM2-7B": "internlm/internlm2-7b",
        "Baichuan2-7B": "baichuan-inc/Baichuan2-7B-Base",
    }

    model_label = st.selectbox("Base Model", list(MODEL_CHOICES.keys()))
    base_model = MODEL_CHOICES[model_label]

    finetune_method = st.selectbox("Fine-tuning Method", ["LoRA", "QLoRA", "Full Fine-tuning"])
    output_dir = st.text_input("Output Directory", value="./outputs/finetuned_model")

    c1, c2 = st.columns(2)
    with c1:
        learning_rate = st.number_input("Learning Rate", value=2e-4, format="%.6f")
        epochs = st.number_input("Epochs", min_value=1, value=3, step=1)
    with c2:
        batch_size = st.number_input("Batch Size", min_value=1, value=4, step=1)
        lora_rank = st.number_input("LoRA Rank", min_value=1, value=8, step=1)

    max_length = st.slider("Max Length", min_value=64, max_value=4096, value=512, step=64)

    st.markdown('<div class="divider-soft"></div>', unsafe_allow_html=True)

    action_col1, action_col2 = st.columns(2)
    with action_col1:
        validate_btn = st.button("Check Dataset", use_container_width=True)
    with action_col2:
        train_btn = st.button("Start Fine-tuning", type="primary", use_container_width=True)

    refresh_btn = st.button("Refresh Status", use_container_width=True)

    st.caption("Tip: upload first, then validate, then train.")


# ============================================================
# Sidebar actions
# ============================================================
if upload_btn:
    if uploaded_file is None:
        st.warning("Please choose a dataset file first.")
    else:
        try:
            result = upload_dataset_api(backend_url, uploaded_file)
            st.session_state.upload_result = result
            st.session_state.uploaded_dataset_path = result["dataset_path"]
            st.session_state.uploaded_dataset_name = result["filename"]
            st.session_state.dataset_report = result.get("dataset_report")
            st.session_state.split_paths = result.get("split_paths")
            st.session_state.validation_result = None
            st.success("Dataset uploaded successfully.")
        except requests.exceptions.RequestException as e:
            st.session_state.last_error = str(e)
            st.error(f"Upload failed: {e}")

if validate_btn:
    if not st.session_state.uploaded_dataset_path:
        st.warning("Please upload a dataset first.")
    else:
        try:
            result = validate_dataset_api(backend_url, st.session_state.uploaded_dataset_path)
            st.session_state.validation_result = result
            st.success("Dataset validation finished.")
        except requests.exceptions.RequestException as e:
            st.session_state.last_error = str(e)
            st.error(f"Validation request failed: {e}")

if train_btn:
    if not st.session_state.uploaded_dataset_path:
        st.warning("Please upload a dataset first.")
    else:
        payload = {
            "base_model": base_model,
            "dataset_path": st.session_state.uploaded_dataset_path,
            "finetune_method": finetune_method,
            "output_dir": output_dir,
            "learning_rate": learning_rate,
            "batch_size": batch_size,
            "epochs": epochs,
            "lora_rank": lora_rank,
            "max_length": max_length,
        }
        try:
            result = train_api(backend_url, payload)
            st.session_state.train_submit_result = result
            st.success("Training job submitted.")
        except requests.exceptions.HTTPError as e:
            detail = ""
            try:
                detail = e.response.json().get("detail", "")
            except Exception:
                detail = str(e)
            st.error(f"Training request rejected: {detail or e}")
        except requests.exceptions.RequestException as e:
            st.session_state.last_error = str(e)
            st.error(f"Training request failed: {e}")

if refresh_btn or st.session_state.training_status is None:
    refresh_status(backend_url)


# ============================================================
# Top metrics
# ============================================================
status_payload = st.session_state.training_status or {}
status_value = status_payload.get("status", "idle")

metric_cols = st.columns(4)
metric_cols[0].metric("Dataset", st.session_state.uploaded_dataset_name or "None")
metric_cols[1].metric("Model", model_label)
metric_cols[2].metric("Method", finetune_method)
metric_cols[3].metric("Status", status_value)


# ============================================================
# Tabs
# ============================================================
setup_tab, monitor_tab, eval_tab, infer_tab = st.tabs(["Setup", "Monitor", "Evaluation", "Inference"])


# ============================================================
# Setup tab
# ============================================================
with setup_tab:
    left, right = st.columns([1.15, 0.95], gap="large")

    with left:
        st.markdown('<div class="section-card">', unsafe_allow_html=True)
        st.markdown('<div class="section-title">Dataset Overview</div>', unsafe_allow_html=True)

        report = st.session_state.get("dataset_report")

        if report:
            c1, c2, c3 = st.columns(3)
            c1.metric("Total Samples", report.get("num_samples", 0))
            c2.metric("Source Format", report.get("source_format", "unknown"))
            c3.metric("Columns", len(report.get("columns", [])))

            st.write("**Detected columns:** " + ", ".join(report.get("columns", [])))

            split_counts = report.get("split_counts", {})
            split_ratio = report.get("split_ratio", {})
            st.write("**Split completed:** Yes")
            st.write(
                f"**Split counts:** train={split_counts.get('train', 0)}, "
                f"validation={split_counts.get('validation', 0)}, "
                f"test={split_counts.get('test', 0)}"
            )
            st.write(
                f"**Split ratio:** train={split_ratio.get('train', 0.8)}, "
                f"validation={split_ratio.get('validation', 0.1)}, "
                f"test={split_ratio.get('test', 0.1)}"
            )

            if report.get("note"):
                st.markdown(
                    f'<div class="hint-box">{report.get("note")}</div>',
                    unsafe_allow_html=True,
                )

            avg_lengths = report.get("avg_field_lengths", {})
            if avg_lengths:
                with st.expander("Average Field Lengths", expanded=False):
                    st.json(avg_lengths)

            empty_counts = report.get("empty_field_counts", {})
            if empty_counts:
                with st.expander("Empty Field Counts", expanded=False):
                    st.json(empty_counts)

            with st.expander("Preview", expanded=False):
                st.json(report.get("preview", []))

            with st.expander("Split Paths", expanded=False):
                st.json(report.get("split_paths", {}))
        else:
            st.info("No dataset uploaded yet.")

        if st.session_state.upload_result:
            with st.expander("Upload Result", expanded=False):
                st.json(st.session_state.upload_result)

        st.markdown(
            '<div class="small-note">Uploaded data is normalized to JSONL and randomly split into train / validation / test on the backend.</div>',
            unsafe_allow_html=True,
        )
        st.markdown("</div>", unsafe_allow_html=True)

        st.write("")
        st.markdown('<div class="section-card">', unsafe_allow_html=True)
        st.markdown('<div class="section-title">Training Configuration</div>', unsafe_allow_html=True)

        report = st.session_state.get("dataset_report") or {}
        config_df = pd.DataFrame(
            [
                {
                    "base_model": base_model,
                    "finetune_method": finetune_method,
                    "dataset_format": report.get("source_format", ""),
                    "num_samples": report.get("num_samples", ""),
                    "split_done": report.get("split_done", False),
                    "output_dir": output_dir,
                    "learning_rate": learning_rate,
                    "batch_size": batch_size,
                    "epochs": epochs,
                    "lora_rank": lora_rank,
                    "max_length": max_length,
                }
            ]
        )
        st.dataframe(config_df, hide_index=True, use_container_width=True)
        st.markdown("</div>", unsafe_allow_html=True)

    with right:
        st.markdown('<div class="section-card">', unsafe_allow_html=True)
        st.markdown('<div class="section-title">Dataset Validation</div>', unsafe_allow_html=True)

        if st.session_state.validation_result:
            vr = st.session_state.validation_result
            if vr.get("valid"):
                st.success(vr.get("message", "Dataset is valid."))
            else:
                st.error(vr.get("message", "Dataset is invalid."))

            m1, m2, m3 = st.columns(3)
            m1.metric("Valid", str(vr.get("valid")))
            m2.metric("Samples", vr.get("num_samples", 0))
            m3.metric("Keys", len(vr.get("sample_keys", [])))

            st.caption(", ".join(vr.get("sample_keys", [])) or "No keys returned.")
        else:
            st.info("Click **Check Dataset** after uploading a file.")

        st.markdown("</div>", unsafe_allow_html=True)

        st.write("")
        st.markdown('<div class="section-card">', unsafe_allow_html=True)
        st.markdown('<div class="section-title">Submission Preview</div>', unsafe_allow_html=True)
        if st.session_state.train_submit_result:
            st.json(st.session_state.train_submit_result)
        else:
            st.caption("No training request submitted yet.")
        st.markdown("</div>", unsafe_allow_html=True)


# ============================================================
# Monitor tab
# ============================================================
with monitor_tab:
    a, b = st.columns([1.1, 0.9], gap="large")

    with a:
        st.markdown('<div class="section-card">', unsafe_allow_html=True)
        st.markdown('<div class="section-title">Training Status</div>', unsafe_allow_html=True)

        st.metric("Current Status", status_value)
        st.write(f"**Message:** {status_payload.get('message', 'No message.')}")

        last_refresh = (
            time.strftime("%Y-%m-%d %H:%M:%S", time.localtime(st.session_state.last_poll_time))
            if st.session_state.last_poll_time
            else "-"
        )
        st.write(f"**Last refresh:** {last_refresh}")

        if st.session_state.last_error:
            st.error(f"Last error: {st.session_state.last_error}")

        if status_payload.get("config"):
            with st.expander("Backend Config", expanded=False):
                st.json(status_payload["config"])

        if status_payload.get("result"):
            with st.expander("Training Result", expanded=True):
                st.json(status_payload["result"])

        st.markdown("</div>", unsafe_allow_html=True)

    with b:
        st.markdown('<div class="section-card">', unsafe_allow_html=True)
        st.markdown('<div class="section-title">Loss Curve</div>', unsafe_allow_html=True)

        result = status_payload.get("result") or {}
        loss_history = result.get("loss_history") or []
        if loss_history:
            loss_df = pd.DataFrame(loss_history)
            if {"step", "loss"}.issubset(loss_df.columns):
                st.line_chart(loss_df.set_index("step")["loss"], use_container_width=True)
            else:
                st.info("Loss history is present, but step/loss columns are missing.")
        else:
            st.info("No loss history yet.")

        st.markdown("</div>", unsafe_allow_html=True)

    st.write("")
    st.markdown('<div class="section-card">', unsafe_allow_html=True)
    st.markdown('<div class="section-title">Raw Status Payload</div>', unsafe_allow_html=True)
    st.json(status_payload if status_payload else {"status": "idle"})
    st.markdown("</div>", unsafe_allow_html=True)

    if status_value == "running":
        st.info("Training is running. Refreshing status automatically...")
        time.sleep(2)
        st.rerun()


# ============================================================
# Evaluation tab
# ============================================================
with eval_tab:
    st.markdown('<div class="section-card">', unsafe_allow_html=True)
    st.markdown('<div class="section-title">Evaluation</div>', unsafe_allow_html=True)

    dataset_report = st.session_state.get("dataset_report") or {}

    left, right = st.columns([0.95, 1.05], gap="large")

    with left:
        st.subheader("Evaluation Settings")

        split_options = ["validation", "test", "train"]
        selected_split = st.selectbox("Split", split_options, index=0)

        max_eval_samples = st.slider("Max Eval Samples", min_value=1, max_value=100, value=20, step=1)
        template = st.selectbox("Template", ["alpaca", "qwen", "llama2", "chatml", "custom"], index=0)

        st.write("### Generation Parameters")
        temperature = st.slider("Temperature", 0.0, 2.0, 0.7, 0.05)
        top_p = st.slider("Top P", 0.0, 1.0, 0.9, 0.01)
        top_k = st.number_input("Top K", min_value=0, value=50, step=1)
        max_new_tokens = st.number_input("Max New Tokens", min_value=16, value=256, step=16)
        repetition_penalty = st.slider("Repetition Penalty", 1.0, 2.0, 1.05, 0.01)
        do_sample = st.checkbox("Do Sample", value=True)
        num_beams = st.number_input("Num Beams", min_value=1, value=1, step=1)
        seed = st.number_input("Seed", min_value=0, value=42, step=1)

        eval_btn = st.button("Run Evaluation", type="primary", use_container_width=True)

    with right:
        st.subheader("Evaluation Result")

        if eval_btn:
            if not dataset_report.get("split_done"):
                st.warning("Please upload and split a dataset first.")
            else:
                payload = {
                    "split_name": selected_split,
                    "base_model": base_model,
                    "model_path": output_dir,
                    "template": template,
                    "max_eval_samples": int(max_eval_samples),
                    "temperature": float(temperature),
                    "top_p": float(top_p),
                    "top_k": int(top_k),
                    "max_new_tokens": int(max_new_tokens),
                    "repetition_penalty": float(repetition_penalty),
                    "do_sample": bool(do_sample),
                    "num_beams": int(num_beams),
                    "seed": int(seed),
                }
                try:
                    result = evaluate_api(backend_url, payload)
                    st.session_state.evaluation_output = result
                    st.session_state.assistant_messages = []
                    st.session_state.assistant_answer = ""
                except requests.exceptions.RequestException as e:
                    st.session_state.last_error = str(e)
                    st.error(f"Evaluation failed: {e}")

        eval_result = st.session_state.get("evaluation_output")
        if eval_result:
            st.success(eval_result.get("message", "Evaluation finished."))

            metrics = eval_result.get("metrics", {})
            m1, m2, m3, m4 = st.columns(4)
            m1.metric("Exact Match", f"{metrics.get('exact_match_accuracy', 0):.4f}")
            m2.metric("Token Accuracy", f"{metrics.get('token_accuracy', 0):.4f}")
            m3.metric("Avg Pred Len", f"{metrics.get('avg_prediction_length', 0):.2f}")
            m4.metric("Eval Samples", metrics.get("evaluated_samples", 0))

            m5, m6 = st.columns(2)
            m5.metric("Avg Ref Len", f"{metrics.get('avg_reference_length', 0):.2f}")
            m6.metric("Device", metrics.get("device", "-"))

            st.write(
                f"**Split:** {eval_result.get('split_name', '-')}"
                f" | **Total in split:** {eval_result.get('total_split_samples', 0)}"
                f" | **Evaluated:** {eval_result.get('evaluated_samples', 0)}"
                f" | **Truncated:** {eval_result.get('truncated', False)}"
            )

            examples = eval_result.get("examples", [])
            if examples:
                table_rows = []
                for ex in examples:
                    table_rows.append(
                        {
                            "idx": ex.get("index", 0),
                            "hit": ex.get("exact_match", False),
                            "token_accuracy": ex.get("token_accuracy", 0),
                            "instruction": ex.get("instruction", ""),
                            "reference": ex.get("reference", ""),
                            "prediction": ex.get("prediction", ""),
                        }
                    )
                df = pd.DataFrame(table_rows)
                st.dataframe(df, hide_index=True, use_container_width=True)

                error_examples = [ex for ex in examples if not ex.get("exact_match", False)]
                if error_examples:
                    with st.expander("Error Examples", expanded=False):
                        for ex in error_examples[:5]:
                            st.markdown(f"**Sample {ex.get('index', 0)}**")
                            st.write("Instruction")
                            st.code(ex.get("instruction", ""))
                            if ex.get("input"):
                                st.write("Input")
                                st.code(ex.get("input", ""))
                            st.write("Reference")
                            st.code(ex.get("reference", ""))
                            st.write("Prediction")
                            st.code(ex.get("prediction", ""))
                            st.divider()

            with st.expander("Generation Config", expanded=False):
                st.json(eval_result.get("generation_config", {}))

            with st.expander("Model Config", expanded=False):
                st.json(eval_result.get("model", {}))

            st.markdown("### AI Assistant")
            st.caption("Click the floating button in the lower-right corner to ask follow-up questions.")
        else:
            st.info("Run evaluation to see accuracy and error analysis.")

        st.markdown("</div>", unsafe_allow_html=True)


# ============================================================
# Inference tab
# ============================================================
with infer_tab:
    st.markdown('<div class="section-card">', unsafe_allow_html=True)
    st.markdown('<div class="section-title">Inference Test</div>', unsafe_allow_html=True)

    prompt = st.text_area(
        "Enter a prompt",
        placeholder="Type a test prompt here...",
        height=120,
    )

    ic1, ic2 = st.columns([1, 1])
    run_infer = ic1.button("Run Inference", type="primary", use_container_width=True)
    clear_logs = ic2.button("Clear Display", use_container_width=True)

    if clear_logs:
        st.session_state.upload_result = None
        st.session_state.validation_result = None
        st.session_state.train_submit_result = None
        st.session_state.training_status = None
        st.session_state.last_error = ""
        st.session_state.inference_output = ""
        st.session_state.evaluation_output = None
        st.session_state.assistant_messages = []
        st.session_state.assistant_answer = ""
        st.rerun()

    if run_infer:
        if not prompt.strip():
            st.warning("Please enter a prompt first.")
        else:
            st.session_state["inference_output"] = (
                f"Model response to: {prompt}\n\n"
                f"[Placeholder] This is where the fine-tuned model output will appear."
            )

    if st.session_state.get("inference_output"):
        st.markdown("### Inference Output")
        st.success(st.session_state["inference_output"])

    st.markdown("</div>", unsafe_allow_html=True)


# ============================================================
# Footer
# ============================================================
st.caption("Backend: FastAPI | Frontend: Streamlit | Assistant model: local Qwen2.5-0.5B-Instruct snapshot")