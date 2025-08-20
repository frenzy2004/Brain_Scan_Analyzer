import streamlit as st
import numpy as np
import cv2
import tensorflow as tf
from tensorflow import keras
import nibabel as nib
import matplotlib.pyplot as plt
import tempfile
import os

# Page config
st.set_page_config(page_title="🧠 Brain Tumor Detection", layout="wide")

st.title("🧠 Brain MRI Tumor Segmentation")
st.write("Upload a brain MRI scan to see AI-detected tumor regions")

# Constants
IMG_SIZE = 128
MODEL_FILE = "brain_tumor_unet_final.h5"  # Using the actual model filename

# Custom functions for model
import tensorflow.keras.backend as K

def dice_coef(y_true, y_pred, smooth=1.0):
    class_num = 4
    for i in range(class_num):
        y_true_f = K.flatten(y_true[:,:,:,i])
        y_pred_f = K.flatten(y_pred[:,:,:,i])
        intersection = K.sum(y_true_f * y_pred_f)
        loss = ((2. * intersection + smooth) / (K.sum(y_true_f) + K.sum(y_pred_f) + smooth))
        if i == 0:
            total_loss = loss
        else:
            total_loss = total_loss + loss
    total_loss = total_loss / class_num
    return total_loss

def dice_coef_necrotic(y_true, y_pred, epsilon=1e-6):
    intersection = K.sum(K.abs(y_true[:,:,:,1] * y_pred[:,:,:,1]))
    return (2. * intersection) / (K.sum(K.square(y_true[:,:,:,1])) + K.sum(K.square(y_pred[:,:,:,1])) + epsilon)

def dice_coef_edema(y_true, y_pred, epsilon=1e-6):
    intersection = K.sum(K.abs(y_true[:,:,:,2] * y_pred[:,:,:,2]))
    return (2. * intersection) / (K.sum(K.square(y_true[:,:,:,2])) + K.sum(K.square(y_pred[:,:,:,2])) + epsilon)

def dice_coef_enhancing(y_true, y_pred, epsilon=1e-6):
    intersection = K.sum(K.abs(y_true[:,:,:,3] * y_pred[:,:,:,3]))
    return (2. * intersection) / (K.sum(K.square(y_true[:,:,:,3])) + K.sum(K.square(y_pred[:,:,:,3])) + epsilon)

def precision(y_true, y_pred):
    true_positives = K.sum(K.round(K.clip(y_true * y_pred, 0, 1)))
    predicted_positives = K.sum(K.round(K.clip(y_pred, 0, 1)))
    return true_positives / (predicted_positives + K.epsilon())

def sensitivity(y_true, y_pred):
    true_positives = K.sum(K.round(K.clip(y_true * y_pred, 0, 1)))
    possible_positives = K.sum(K.round(K.clip(y_true, 0, 1)))
    return true_positives / (possible_positives + K.epsilon())

def specificity(y_true, y_pred):
    true_negatives = K.sum(K.round(K.clip((1-y_true) * (1-y_pred), 0, 1)))
    possible_negatives = K.sum(K.round(K.clip(1-y_true, 0, 1)))
    return true_negatives / (possible_negatives + K.epsilon())

# Load model
@st.cache_resource
def load_model():
    try:
        model = keras.models.load_model(
            MODEL_FILE,
            custom_objects={
                "dice_coef": dice_coef,
                "dice_coef_necrotic": dice_coef_necrotic,
                "dice_coef_edema": dice_coef_edema,
                "dice_coef_enhancing": dice_coef_enhancing,
                "precision": precision,
                "sensitivity": sensitivity,
                "specificity": specificity
            },
            compile=False
        )
        return model
    except Exception as e:
        st.error(f"Error loading model: {e}")
        return None

# Sidebar
with st.sidebar:
    st.header("📊 About This AI")
    st.info("""
    **Model Performance:**
    - Dice Score: 0.82
    - Accuracy: 89%
    
    **Detects 3 tumor types:**
    - 🔴 Necrotic Core
    - 🟡 Edema
    - 🔵 Enhancing Tumor
    """)

# Main area
st.subheader("Upload Brain MRI")
uploaded_file = st.file_uploader("Choose a NIfTI file (.nii or .nii.gz)", type=["nii", "gz"])

if uploaded_file is not None:
    # Load model
    with st.spinner("Loading AI model..."):
        model = load_model()
    
    if model is not None:
        # Save uploaded file
        with tempfile.NamedTemporaryFile(delete=False, suffix=".nii") as tmp:
            tmp.write(uploaded_file.read())
            temp_path = tmp.name
        
        # Load MRI
        try:
            mri = nib.load(temp_path)
            mri_data = mri.get_fdata()
            
            st.success(f"✅ Loaded MRI with shape: {mri_data.shape}")
            
            # Select slice
            if len(mri_data.shape) == 3:
                slice_num = st.slider("Select slice to analyze", 
                                    0, mri_data.shape[2]-1, 
                                    mri_data.shape[2]//2)
                
                # Get slice
                slice_data = mri_data[:, :, slice_num]
                
                # Resize to model input
                slice_resized = cv2.resize(slice_data, (IMG_SIZE, IMG_SIZE))
                
                # Normalize
                if slice_resized.max() > 0:
                    slice_norm = slice_resized / slice_resized.max()
                else:
                    slice_norm = slice_resized
                
                # Prepare input (2 channels)
                input_data = np.zeros((1, IMG_SIZE, IMG_SIZE, 2))
                input_data[0, :, :, 0] = slice_norm
                input_data[0, :, :, 1] = slice_norm
                
                # Predict
                with st.spinner("🧠 AI is analyzing..."):
                    prediction = model.predict(input_data, verbose=0)
                
                # Get predicted classes
                pred_classes = np.argmax(prediction[0], axis=-1)
                
                # Create visualization
                col1, col2, col3 = st.columns(3)
                
                fig1, ax1 = plt.subplots(figsize=(5, 5))
                ax1.imshow(slice_norm.T, cmap="gray", origin="lower")
                ax1.set_title("Original MRI")
                ax1.axis("off")
                col1.pyplot(fig1)
                
                fig2, ax2 = plt.subplots(figsize=(5, 5))
                im = ax2.imshow(pred_classes.T, cmap="jet", origin="lower", vmin=0, vmax=3)
                ax2.set_title("AI Segmentation")
                ax2.axis("off")
                col2.pyplot(fig2)
                
                fig3, ax3 = plt.subplots(figsize=(5, 5))
                ax3.imshow(slice_norm.T, cmap="gray", origin="lower")
                masked = np.ma.masked_where(pred_classes.T == 0, pred_classes.T)
                ax3.imshow(masked, cmap="jet", alpha=0.5, origin="lower", vmin=1, vmax=3)
                ax3.set_title("Overlay")
                ax3.axis("off")
                col3.pyplot(fig3)
                
                # Show statistics
                st.subheader("📊 Analysis Results")
                
                total_pixels = pred_classes.size
                tumor_pixels = np.sum(pred_classes > 0)
                
                if tumor_pixels > 0:
                    st.warning(f"⚠️ Tumor detected! Covers {(tumor_pixels/total_pixels)*100:.1f}% of this slice")
                    
                    # Count each class
                    necrotic = np.sum(pred_classes == 1)
                    edema = np.sum(pred_classes == 2)
                    enhancing = np.sum(pred_classes == 3)
                    
                    col1, col2, col3 = st.columns(3)
                    col1.metric("Necrotic/Core", f"{(necrotic/total_pixels)*100:.1f}%")
                    col2.metric("Edema", f"{(edema/total_pixels)*100:.1f}%")
                    col3.metric("Enhancing", f"{(enhancing/total_pixels)*100:.1f}%")
                else:
                    st.success("✅ No tumor detected in this slice")
                
        except Exception as e:
            st.error(f"Error processing MRI: {e}")
        finally:
            # Clean up
            if os.path.exists(temp_path):
                os.unlink(temp_path)
else:
    # Show demo info when no file uploaded
    st.info("👆 Please upload a brain MRI file (NIfTI format) to start analysis")
    
    st.subheader("📖 How it works:")
    st.write("""
    1. Upload a brain MRI scan in NIfTI format (.nii or .nii.gz)
    2. Select which slice to analyze using the slider
    3. The AI will automatically detect and highlight tumor regions
    4. View the segmentation results and statistics
    """)
