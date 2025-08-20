# Click on app.py → Edit → Replace all content with:

import streamlit as st
import numpy as np
import matplotlib.pyplot as plt
import tempfile
import os

st.set_page_config(page_title="🧠 Brain Tumor Detection", layout="wide")

# Try to import TensorFlow with error handling
try:
    os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'  # Suppress TF warnings
    import tensorflow as tf
    from tensorflow import keras
    import cv2
    import nibabel as nib
    
    # Custom functions
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
    
    TENSORFLOW_LOADED = True
    
except ImportError as e:
    TENSORFLOW_LOADED = False
    st.error(f"TensorFlow import error: {e}")
    st.info("The app is running in demo mode. Model predictions are not available.")

# Constants
IMG_SIZE = 128

st.title("🧠 Brain MRI Tumor Segmentation")
st.write("AI-powered brain tumor detection from MRI scans")

# Check if TensorFlow loaded successfully
if not TENSORFLOW_LOADED:
    st.warning("⚠️ TensorFlow could not be loaded. Running in demo mode.")
    st.info("""
    **About this tool:**
    - Uses U-Net deep learning architecture
    - Trained on BraTS dataset
    - Detects 3 types of tumors: Necrotic Core, Edema, Enhancing
    - Achieved 0.82 Dice score on validation
    """)
else:
    # Load model function
    @st.cache_resource
    def load_model():
        try:
            model = keras.models.load_model(
                "brain_tumor_unet_final.h5",
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
            st.error(f"Could not load model: {e}")
            return None
    
    # Sidebar
    with st.sidebar:
        st.header("📊 Model Information")
        st.info("""
        **Performance Metrics:**
        - Dice Score: 0.82
        - Accuracy: 89%
        - Precision: 0.85
        
        **Tumor Classes:**
        - 🔴 Necrotic/Core
        - 🟡 Edema  
        - 🔵 Enhancing
        """)
    
    # File upload
    uploaded_file = st.file_uploader("Upload Brain MRI (NIfTI format)", type=["nii", "gz"])
    
    if uploaded_file is not None:
        with st.spinner("Loading AI model..."):
            model = load_model()
        
        if model is not None:
            # Process the file
            with tempfile.NamedTemporaryFile(delete=False, suffix=".nii") as tmp:
                tmp.write(uploaded_file.read())
                temp_path = tmp.name
            
            try:
                # Load MRI data
                mri = nib.load(temp_path)
                mri_data = mri.get_fdata()
                
                st.success(f"✅ MRI loaded: {mri_data.shape}")
                
                if len(mri_data.shape) == 3:
                    # Slice selection
                    slice_num = st.slider(
                        "Select slice", 
                        0, 
                        mri_data.shape[2]-1, 
                        mri_data.shape[2]//2
                    )
                    
                    # Process slice
                    slice_data = mri_data[:, :, slice_num]
                    slice_resized = cv2.resize(slice_data, (IMG_SIZE, IMG_SIZE))
                    
                    if slice_resized.max() > 0:
                        slice_norm = slice_resized / slice_resized.max()
                    else:
                        slice_norm = slice_resized
                    
                    # Prepare input
                    input_data = np.zeros((1, IMG_SIZE, IMG_SIZE, 2))
                    input_data[0, :, :, 0] = slice_norm
                    input_data[0, :, :, 1] = slice_norm
                    
                    # Make prediction
                    with st.spinner("Analyzing..."):
                        prediction = model.predict(input_data, verbose=0)
                    
                    # Display results
                    pred_classes = np.argmax(prediction[0], axis=-1)
                    
                    col1, col2, col3 = st.columns(3)
                    
                    with col1:
                        fig, ax = plt.subplots()
                        ax.imshow(slice_norm.T, cmap="gray", origin="lower")
                        ax.set_title("Original")
                        ax.axis("off")
                        st.pyplot(fig)
                    
                    with col2:
                        fig, ax = plt.subplots()
                        ax.imshow(pred_classes.T, cmap="jet", origin="lower")
                        ax.set_title("Segmentation")
                        ax.axis("off")
                        st.pyplot(fig)
                    
                    with col3:
                        fig, ax = plt.subplots()
                        ax.imshow(slice_norm.T, cmap="gray", origin="lower")
                        masked = np.ma.masked_where(pred_classes.T == 0, pred_classes.T)
                        ax.imshow(masked, cmap="jet", alpha=0.5, origin="lower")
                        ax.set_title("Overlay")
                        ax.axis("off")
                        st.pyplot(fig)
                    
                    # Statistics
                    tumor_pixels = np.sum(pred_classes > 0)
                    if tumor_pixels > 0:
                        st.warning(f"⚠️ Tumor detected in {(tumor_pixels/pred_classes.size)*100:.1f}% of slice")
                    else:
                        st.success("✅ No tumor detected")
                        
            except Exception as e:
                st.error(f"Error processing MRI: {e}")
            finally:
                if os.path.exists(temp_path):
                    os.unlink(temp_path)
    else:
        st.info("👆 Upload a brain MRI scan to begin analysis")