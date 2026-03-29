from __future__ import annotations

import time
from typing import Any, Dict

import pandas as pd
import requests
import streamlit as st


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
            padding-top: 1.1rem;
            padding-bottom: 1.5rem;
        }
        .hero {
            padding: 1.2rem 1.4rem;
            border-radius: 22px;
            background: linear-gradient(135deg, rgba(57, 92, 255, 0.12), rgba(0, 180, 216, 0.10));
            border: 1px solid rgba(120, 120, 120, 0.18);
            margin-bottom: 1rem;
        }
        .hero h1 {
            margin: 0;
            font-size: 2.0rem;
            line-height: 1.15;
        }
        .hero p {
            margin: 0.35rem 0 0 0;
            opacity: 0.8;
            font-size: 0.98rem;
        }
        .section-card {
            padding: 1rem 1rem 0.9rem 1rem;
            border-radius: 18px;
            border: 1px solid rgba(120, 120, 120, 0.18);
            background: rgba(255, 255, 255, 0.03);
            box-shadow: 0 8px 24px rgba(0, 0, 0, 0.04);
            height: 100%;
        }
        .section-title {
            font-size: 1.05rem;
            font-weight: 700;
            margin-bottom: 0.8rem;
        }
        .muted {
            opacity: 0.7;
            font-size: 0.92rem;
        }
        .status-pill {
            display: inline-block;
            padding: 0.28rem 0.7rem;
            border-radius: 999px;
            font-size: 0.82rem;
            font-weight: 600;
            border: 1px solid rgba(120, 120, 120, 0.22);
            margin-left: 0.4rem;
        }
        .pill-idle { background: rgba(120,120,120,0.10); }
        .pill-running { background: rgba(255,193,7,0.14); }
        .pill-success { background: rgba(46,204,113,0.14); }
        .pill-failed { background: rgba(231,76,60,0.14); }
        .small-note {
            font-size: 0.86rem;
            opacity: 0.76;
            margin-top: 0.4rem;
        }
    </style>
    """,
    unsafe_allow_html=True,
)


# ============================================================
# Constants / Session state
# ============================================================
DEFAULT_BACKEND_URL = "http://127.0.0.1:8000"


def init_state() -> None:
    defaults = {
        "uploaded_dataset_path": "",
        "uploaded_dataset_name": "",
        "upload_result": None,
        "validation_result": None,
        "train_submit_result": None,
        "training_status": None,
        "last_poll_time": 0.0,
        "last_error": "",
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


def refresh_status(backend_url: str) -> None:
    try:
        st.session_state.training_status = training_status_api(backend_url)
        st.session_state.last_poll_time = time.time()
        st.session_state.last_error = ""
    except requests.exceptions.RequestException as e:
        st.session_state.last_error = str(e)


# ============================================================
# Sidebar
# ============================================================
with st.sidebar:
    st.header("Control Panel")
    backend_url = st.text_input("Backend URL", value=DEFAULT_BACKEND_URL)

    st.divider()
    st.subheader("1. Dataset")
    uploaded_file = st.file_uploader(
        "Upload .jsonl dataset",
        type=["jsonl"],
        help="Expected fields per line: instruction, input, output.",
    )
    upload_btn = st.button("Upload to Backend", use_container_width=True)

    st.divider()
    st.subheader("2. Training")
    base_model = st.selectbox(
        "Base Model",
        ["Qwen2.5-0.5B", "Qwen2.5-1.5B", "Llama-3.1-8B", "Mistral-7B"],
    )
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

    st.divider()
    validate_btn = st.button("Check Dataset", use_container_width=True)
    train_btn = st.button("🚀 Start Fine-tuning", type="primary", use_container_width=True)
    refresh_btn = st.button("🔄 Refresh Status", use_container_width=True)

    st.caption("Tip: upload first, then validate, then train.")


# ============================================================
# Sidebar actions
# ============================================================
if upload_btn:
    if uploaded_file is None:
        st.warning("Please choose a .jsonl file first.")
    else:
        try:
            result = upload_dataset_api(backend_url, uploaded_file)
            st.session_state.upload_result = result
            st.session_state.uploaded_dataset_path = result["dataset_path"]
            st.session_state.uploaded_dataset_name = result["filename"]
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
# Header / Hero
# ============================================================
status_payload = st.session_state.training_status or {}
status_value = status_payload.get("status", "idle")
status_class = f"pill-{status_value}" if status_value in {"idle", "running", "success", "failed"} else "pill-idle"

st.markdown(
    f"""
    <div class="hero">
        <h1>🤖 LLM Fine-tuning Platform <span class="status-pill {status_class}">{status_value.upper()}</span></h1>
        <p>Upload a dataset, validate it, launch training, and monitor progress from a clean visual dashboard.</p>
    </div>
    """,
    unsafe_allow_html=True,
)


# ============================================================
# Top metrics
# ============================================================
metric_cols = st.columns(4)
metric_cols[0].metric("Dataset", st.session_state.uploaded_dataset_name or "None")
metric_cols[1].metric("Model", base_model)
metric_cols[2].metric("Method", finetune_method)
metric_cols[3].metric("Status", status_value)


# ============================================================
# Main tabs
# ============================================================
setup_tab, monitor_tab, infer_tab = st.tabs(["Setup", "Monitor", "Inference"])


# ----------------------------
# Setup tab
# ----------------------------
with setup_tab:
    left, right = st.columns([1.15, 0.95], gap="large")

    with left:
        st.markdown('<div class="section-card">', unsafe_allow_html=True)
        st.markdown('<div class="section-title">Dataset Overview</div>', unsafe_allow_html=True)

        if st.session_state.uploaded_dataset_path:
            st.code(st.session_state.uploaded_dataset_path)
            st.write(f"Uploaded file: **{st.session_state.uploaded_dataset_name}**")
        else:
            st.info("No dataset uploaded yet.")

        if st.session_state.upload_result:
            with st.expander("Upload Result", expanded=False):
                st.json(st.session_state.upload_result)

        st.markdown("<div class='small-note'>The backend stores the uploaded file and returns a server-side path for validation and training.</div>", unsafe_allow_html=True)
        st.markdown("</div>", unsafe_allow_html=True)

        st.write("")
        st.markdown('<div class="section-card">', unsafe_allow_html=True)
        st.markdown('<div class="section-title">Training Configuration</div>', unsafe_allow_html=True)

        config_df = pd.DataFrame(
            [
                {
                    "base_model": base_model,
                    "finetune_method": finetune_method,
                    "dataset_path": st.session_state.uploaded_dataset_path or "",
                    "output_dir": output_dir,
                    "learning_rate": learning_rate,
                    "batch_size": batch_size,
                    "epochs": epochs,
                    "lora_rank": lora_rank,
                    "max_length": max_length,
                }
            ]
        )
        st.dataframe(config_df, width="stretch", hide_index=True)
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


# ----------------------------
# Monitor tab
# ----------------------------
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
                st.line_chart(loss_df.set_index("step")["loss"], width="stretch")
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


# ----------------------------
# Inference tab
# ----------------------------
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
st.caption("Backend: FastAPI | Frontend: Streamlit | MVP focus: upload, validate, train, monitor.")
