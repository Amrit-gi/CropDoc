"""CropDoc - Streamlit Web Application Interface.

Provides an interactive user interface for farmers, agricultural extension workers,
and researchers to diagnose betel leaf diseases, view severity scoring, analyze Grad-CAM heatmaps,
perform multi-leaf plant canopy batch analysis, and access targeted treatment recommendations.

Run using: streamlit run app/streamlit_app.py
"""

import os
import sys

# Suppress TF memory allocation issues and C++ warnings at top level
os.environ["TF_ENABLE_ONEDNN_OPTS"] = "0"
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"

# Ensure repository root is in Python path for imports
repo_root = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if repo_root not in sys.path:
    sys.path.insert(0, repo_root)

import cv2
import numpy as np
import streamlit as st
from PIL import Image
import io
from src.pipeline import diagnose_leaf
from src.validate_image import validate_image
from config import MODEL_PATH

# Streamlit Page Config
st.set_page_config(
    page_title="CropDoc - Betel Leaf Disease Detection System",
    page_icon="🍃",
    layout="wide",
    initial_sidebar_state="collapsed"
)

# Custom CSS styling - Hide Streamlit headers, deploy button, sidebar controls, and footer
st.markdown("""
    <style>
    /* Hide Streamlit top header, hamburger menu, deploy button, and footer */
    #MainMenu {visibility: hidden;}
    header {visibility: hidden;}
    footer {visibility: hidden;}
    .stDeployButton {display: none !important;}
    [data-testid="stSidebar"] {display: none !important;}
    [data-testid="collapsedControl"] {display: none !important;}
    
    /* Clean custom app layout */
    .block-container { padding-top: 1.5rem !important; padding-bottom: 2rem !important; }
    .main-title { font-size: 2.4rem; color: #1e5631; font-weight: 700; text-align: center; margin-bottom: 0.2rem; }
    .subtitle { font-size: 1.05rem; color: #4b6b54; text-align: center; margin-bottom: 1.5rem; }
    .step-box { background-color: #f4f8f5; border-left: 5px solid #2e7d32; padding: 1rem; border-radius: 8px; margin-bottom: 1rem; }
    .canopy-box { background-color: #eef6f0; border: 2px solid #2e7d32; padding: 1.2rem; border-radius: 10px; margin-top: 1rem; }
    </style>
""", unsafe_allow_html=True)


def main():
    st.markdown('<div class="main-title">🍃 CropDoc: Betel Leaf Disease Detection & Canopy Health System</div>', unsafe_allow_html=True)
    st.markdown('<div class="subtitle">Disease Identification $\\rightarrow$ Severity Estimation $\\rightarrow$ Grad-CAM Explainability $\\rightarrow$ Targeted Treatment</div>', unsafe_allow_html=True)

    # Dataset & Model Information Expander
    with st.expander("📚 Dataset & Model Specifications (Mendeley Data DOI: 10.17632/vpzkntzjty.1)", expanded=False):
        d_col1, d_col2, d_col3 = st.columns(3)
        with d_col1:
            st.markdown("**Dataset Scope**\n- Total Images: **12,222 images**\n- Original Field Images: **2,037 images**\n- Augmented Dataset: **10,185 images**")
        with d_col2:
            st.markdown("**Pathogen Classes**\n- `Healthy_Leaf` (1,080 field images)\n- `Leaf_Rot` (*Phytophthora parasitica*)\n- `Leaf_Spot` (*Xanthomonas campestris*)")
        with d_col3:
            st.markdown("**Model Backbones**\n- Primary: **MobileNetV2** (Lightweight Edge)\n- Benchmark: **EfficientNetB0** (88.22% Acc)\n- Pipeline: Single & Multi-Leaf Canopy Batch")

    st.markdown("---")

    # Mode Selector: Single Leaf vs Multi-Leaf Canopy Assessment
    upload_mode = st.radio(
        "**Select Assessment Mode:**",
        ["🍃 Single Leaf Diagnosis", "🌿 Multi-Leaf Plant Canopy Assessment (Batch Upload)"],
        horizontal=True
    )

    if upload_mode == "🍃 Single Leaf Diagnosis":
        render_single_leaf_mode()
    else:
        render_multi_leaf_mode()


def render_single_leaf_mode():
    col_upload, col_display = st.columns([1, 1])

    with col_upload:
        st.subheader("1. Upload Betel Leaf Photo")
        uploaded_file = st.file_uploader(
            "Choose a clear betel leaf photo (JPG / PNG):",
            type=["jpg", "jpeg", "png"],
            key="single_uploader"
        )
        sample_btn = st.button("🖼️ Use Example Sample Leaf", help="Load sample image from repository")

    image_bytes = None
    input_filename = None
    if uploaded_file is not None:
        image_bytes = uploaded_file.getvalue()
        input_filename = uploaded_file.name
    elif sample_btn:
        sample_path = os.path.join(repo_root, "data", "sample_images")
        found_sample = None
        for root, dirs, files in os.walk(sample_path):
            for f in files:
                if f.lower().endswith(('.jpg', '.jpeg', '.png')):
                    found_sample = os.path.join(root, f)
                    break
            if found_sample:
                break

        if found_sample:
            with open(found_sample, "rb") as f:
                image_bytes = f.read()
            input_filename = os.path.basename(found_sample)
            st.success(f"Loaded sample image: `{input_filename}`")

    if image_bytes is not None:
        with st.spinner("Diagnosing leaf... Running botanical validation, classification, severity & Grad-CAM..."):
            res = diagnose_leaf(image_bytes, filename=input_filename)

        if not res["is_valid"]:
            st.error(f"❌ **AI & Botanical Validation Rejected:**\n\n{res['error_message']}")
            st.warning("⚠️ **Botanical Scope Notice:** CropDoc is calibrated strictly for **Betel Vine (*Piper betle*)**. To protect against false-positive diagnoses during field deployment and panel evaluation, all non-betel leaves (such as Banana, Peepal, Mango, Tomato, Neem) are actively detected and rejected.")
            with col_display:
                st.subheader("2. Visual Screening Result")
                try:
                    uploaded_pil = Image.open(io.BytesIO(image_bytes))
                    st.image(uploaded_pil, caption="⚠️ Rejected Non-Betel Specimen", width="stretch")
                    st.error("🛑 DIAGNOSTIC INFERENCE BLOCKED: Non-Betel Leaf Detected")
                except Exception:
                    pass
            return

        with col_display:
            st.subheader("2. Visual Diagnostic Analysis")
            img_col1, img_col2 = st.columns(2)
            with img_col1:
                st.caption("📷 Original Upload")
                rgb_orig = cv2.cvtColor(res["original_bgr"], cv2.COLOR_BGR2RGB)
                st.image(rgb_orig, width="stretch")
            with img_col2:
                st.caption("🔥 Grad-CAM Heatmap Overlay")
                rgb_gradcam = cv2.cvtColor(res["gradcam_overlay"], cv2.COLOR_BGR2RGB)
                st.image(rgb_gradcam, width="stretch")

        st.markdown("---")
        st.subheader("3. Sequential Pathology Diagnosis & Targeted Treatment")

        # Explicit 3-Step Process Panel
        s1, s2, s3 = st.columns(3)

        with s1:
            st.markdown("#### Step 1: Disease Name")
            disease_raw = res["disease_predicted"]
            if disease_raw == "Leaf_Rot":
                disease_title = "LEAF ROT"
                pathogen_detail = "Fungal Infection (*Phytophthora parasitica*)"
            elif disease_raw == "Leaf_Spot":
                disease_title = "LEAF SPOT"
                pathogen_detail = "Bacterial Infection (*Xanthomonas campestris*)"
            else:
                disease_title = "HEALTHY LEAF"
                pathogen_detail = "No Pathogen Infection Detected"

            st.metric("Predicted Disease Name", disease_title)
            st.caption(f"**Pathogen Type:** {pathogen_detail}")
            st.caption(f"Model Confidence: **{res['confidence'] * 100:.1f}%**")
            st.progress(min(1.0, res["confidence"]))

        with s2:
            st.markdown("#### Step 2: Severity Estimation")
            st.metric("Infected Surface Ratio", f"{res['severity_percent']:.1f}%")
            sev_level = res["severity_level"]
            if sev_level == "Healthy":
                st.success(f"Infection Grade: **{sev_level}**")
            elif sev_level == "Mild":
                st.info(f"Infection Grade: **{sev_level}**")
            elif sev_level == "Moderate":
                st.warning(f"Infection Grade: **{sev_level}**")
            else:
                st.error(f"Infection Grade: **{sev_level}**")

        with s3:
            st.markdown("#### Step 3: Class Probability Distribution")
            for cls_name, prob in res["all_probabilities"].items():
                st.text(f"{cls_name}: {prob * 100:.1f}%")

        # Targeted Prescription Card
        if res["treatment"]:
            st.markdown("---")
            st.subheader(f"💊 Disease-Targeted Treatment Plan for {disease_title}")
            tr = res["treatment"]

            st.info(f"**Pathogen Target & Status:** {tr.get('status', 'N/A')}")

            t_col1, t_col2 = st.columns(2)
            with t_col1:
                st.markdown(f"🧪 **Specific Chemical Fungicide / Bactericide:**\n{tr.get('chemical_treatment', 'N/A')}")
                st.markdown(f"🌱 **Targeted Organic Treatment:**\n{tr.get('organic_treatment', 'N/A')}")

            with t_col2:
                st.markdown(f"✂️ **Cultural Practices:**\n{tr.get('cultural_practices', 'N/A')}")
                st.markdown(f"🛡️ **Preventive Field Measures:**\n{tr.get('preventive_measures', 'N/A')}")


def render_multi_leaf_mode():
    st.subheader("🌿 Multi-Leaf Plant Canopy Assessment")
    st.info("Upload multiple leaf photos (3 to 10 leaves) from different vines of the same plant or garden to generate a **Canopy-Wide Health Index**.")

    uploaded_files = st.file_uploader(
        "Choose multiple betel leaf photos:",
        type=["jpg", "jpeg", "png"],
        accept_multiple_files=True,
        key="multi_uploader"
    )

    if uploaded_files:
        results_list = []
        valid_count = 0

        st.markdown("---")
        st.write(f"### 🔍 Analyzing {len(uploaded_files)} Canopy Leaf Samples...")

        cols = st.columns(min(3, len(uploaded_files)))

        for idx, file in enumerate(uploaded_files):
            img_bytes = file.getvalue()
            res = diagnose_leaf(img_bytes, filename=file.name)

            with cols[idx % len(cols)]:
                st.markdown(f"**Leaf #{idx+1}: `{file.name}`**")
                if res["is_valid"]:
                    valid_count += 1
                    results_list.append(res)
                    rgb_orig = cv2.cvtColor(res["original_bgr"], cv2.COLOR_BGR2RGB)
                    st.image(rgb_orig, caption=f"{res['disease_predicted']} ({res['severity_percent']:.1f}% Severity)", width="stretch")
                else:
                    st.error(f"❌ Invalid: {res['error_message']}")

        if results_list:
            st.markdown("---")
            st.markdown('<div class="canopy-box">', unsafe_allow_html=True)
            st.subheader("📊 Plant Canopy Health Aggregator Summary")

            diseases = [r["disease_predicted"] for r in results_list]
            severities = [r["severity_percent"] for r in results_list]

            avg_severity = np.mean(severities)
            max_severity = np.max(severities)

            # Find dominant disease
            disease_counts = {d: diseases.count(d) for d in set(diseases)}
            dominant_disease_raw = max(disease_counts, key=disease_counts.get)
            if dominant_disease_raw == "Leaf_Rot":
                dominant_disease = "LEAF ROT"
            elif dominant_disease_raw == "Leaf_Spot":
                dominant_disease = "LEAF SPOT"
            else:
                dominant_disease = "HEALTHY LEAF"

            m1, m2, m3, m4 = st.columns(4)
            m1.metric("Valid Leaves Analyzed", f"{valid_count} / {len(uploaded_files)}")
            m2.metric("Dominant Disease Name", dominant_disease)
            m3.metric("Average Canopy Severity", f"{avg_severity:.1f}%")
            m4.metric("Peak Leaf Severity", f"{max_severity:.1f}%")

            st.markdown("---")
            st.subheader(f"🌿 Garden-Wide Treatment Plan (Based on Dominant Disease: {dominant_disease})")

            # Get treatment for worst case or dominant case
            sample_res = next(r for r in results_list if r["disease_predicted"] == dominant_disease_raw)
            tr = sample_res["treatment"]

            tc1, tc2 = st.columns(2)
            with tc1:
                st.markdown(f"🧪 **Garden Chemical Spray Protocol:**\n{tr.get('chemical_treatment', 'N/A')}")
                st.markdown(f"🌱 **Canopy Organic Bio-Treatment:**\n{tr.get('organic_treatment', 'N/A')}")
            with tc2:
                st.markdown(f"✂️ **Vine Pruning & Cultural Practices:**\n{tr.get('cultural_practices', 'N/A')}")
                st.markdown(f"🛡️ **Field Prevention:**\n{tr.get('preventive_measures', 'N/A')}")

            st.markdown('</div>', unsafe_allow_html=True)


if __name__ == "__main__":
    main()
