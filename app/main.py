import os
import json
import pandas as pd
import plotly.express as px
from PIL import Image
import numpy as np
import tensorflow as tf
import streamlit as st

# Set page config FIRST
st.set_page_config(
    page_title="Plant Disease Prediction",
    page_icon="🌿",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS
st.markdown("""
<style>
    .stApp {
        background-color: #f5f5f5;
    }
    .prediction-card {
        background-color: #e8f5e9;
        border-radius: 10px;
        padding: 20px;
        margin: 10px 0;
        box-shadow: 0 4px 8px 0 rgba(0,0,0,0.2);
    }
    .stat-card {
        background-color: #e3f2fd;
        border-radius: 10px;
        padding: 20px;
        margin: 10px 0;
        box-shadow: 0 4px 8px 0 rgba(0,0,0,0.2);
    }
</style>
""", unsafe_allow_html=True)


# Load model and class indices
@st.cache_resource
def load_model():
    working_dir = os.path.dirname(os.path.abspath(__file__))
    model_path = f"{working_dir}/trained_model/plant_disease_prediction_model.h5"
    model = tf.keras.models.load_model(model_path)
    class_indices = json.load(open(f"{working_dir}/class_indices.json"))
    return model, class_indices


try:
    model, class_indices = load_model()
except Exception as e:
    st.error(f"Error loading model: {e}")
    st.stop()

# Sample disease statistics (replace with your actual data)
disease_stats = {
    "Healthy": {"count": 1200, "accuracy": 0.95},
    "Powdery Mildew": {"count": 850, "accuracy": 0.89},
    "Leaf Spot": {"count": 720, "accuracy": 0.87},
    "Rust": {"count": 680, "accuracy": 0.91},
    "Bacterial Blight": {"count": 540, "accuracy": 0.85}
}

# Convert to DataFrame for visualization
df_stats = pd.DataFrame.from_dict(disease_stats, orient='index').reset_index()
df_stats = df_stats.rename(columns={"index": "Disease"})


# Image processing functions
def load_and_preprocess_image(image_path, target_size=(224, 224)):
    img = Image.open(image_path)
    img = img.resize(target_size)
    img_array = np.array(img)
    img_array = np.expand_dims(img_array, axis=0)
    img_array = img_array.astype('float32') / 255.
    return img_array


def predict_image_class(model, image_path, class_indices):
    preprocessed_img = load_and_preprocess_image(image_path)
    predictions = model.predict(preprocessed_img)
    predicted_class_index = np.argmax(predictions, axis=1)[0]
    predicted_class_name = class_indices[str(predicted_class_index)]
    confidence = np.max(predictions) * 100
    return predicted_class_name, confidence, predictions[0]


# App layout
st.title("🌱 Plant Disease Prediction")
st.markdown("Upload an image of a plant leaf to detect potential diseases")

# Main columns
col1, col2 = st.columns([1, 1])

with col1:
    uploaded_image = st.file_uploader(
        "Choose a leaf image...",
        type=["jpg", "jpeg", "png"],
        key="file_uploader"
    )

    if uploaded_image is not None:
        image = Image.open(uploaded_image)
        st.image(image, caption="Uploaded Leaf Image", use_column_width=True)

with col2:
    if uploaded_image is not None:
        if st.button("🔍 Analyze Image", type="primary", use_container_width=True):
            with st.spinner('Analyzing the leaf image...'):
                try:
                    prediction, confidence, all_predictions = predict_image_class(
                        model, uploaded_image, class_indices
                    )

                    # Display prediction results
                    st.markdown(f"""
                    <div class="prediction-card">
                        <h3>Detection Results</h3>
                        <p><strong>Prediction:</strong> {prediction}</p>
                        <p><strong>Confidence:</strong> {confidence:.2f}%</p>
                    </div>
                    """, unsafe_allow_html=True)

                    # Display disease statistics
                    st.markdown("### 📊 Overall Model Testing Accuracy")

                    tab1, tab2 = st.tabs(["Accuracy by Disease", "Detection Frequency"])

                    with tab1:
                        fig_acc = px.bar(
                            df_stats,
                            x="Disease",
                            y="accuracy",
                            title="Model Accuracy by Disease Type",
                            color="accuracy",
                            color_continuous_scale="Greens"
                        )
                        st.plotly_chart(fig_acc, use_container_width=True)

                    with tab2:
                        fig_freq = px.pie(
                            df_stats,
                            names="Disease",
                            values="count",
                            title="Disease Detection Frequency",
                            hole=0.3
                        )
                        st.plotly_chart(fig_freq, use_container_width=True)

                    # Display prediction distribution
                    st.markdown("### 📈 Prediction Distribution")
                    classes = list(class_indices.values())
                    pred_df = pd.DataFrame({
                        "Disease": classes,
                        "Probability": all_predictions
                    })
                    pred_df = pred_df.sort_values("Probability", ascending=False)

                    fig_pred = px.bar(
                        pred_df.head(5),
                        x="Disease",
                        y="Probability",
                        title="Top 5 Predicted Diseases",
                        color="Probability",
                        color_continuous_scale="Tealrose"
                    )
                    st.plotly_chart(fig_pred, use_container_width=True)

                    # Add recommendations
                    with st.expander("💡 Recommendations"):
                        if "healthy" in prediction.lower():
                            st.success("""
                            **Healthy Plant Recommendations:**
                            - Continue current care routine
                            - Monitor for any changes
                            - Maintain proper watering and sunlight
                            """)
                        else:
                            st.warning(f"""
                            **Treatment Recommendations for {prediction}:**
                            - Isolate affected plants
                            - Remove severely infected leaves
                            - Apply appropriate fungicide/pesticide
                            - Improve air circulation
                            - Consult local agricultural expert
                            """)

                except Exception as e:
                    st.error(f"An error occurred during analysis: {e}")

# Sample images section
st.markdown("---")
st.markdown("### 📸 Sample Images for Testing")
sample_cols = st.columns(4)
sample_images = [
    "plant-disease-prediction-cnn-deep-leanring-project-main/sample images/cherry.jpg",
    "plant-disease-prediction-cnn-deep-leanring-project-main/sample images/grape healthy.jpg",
    "plant-disease-prediction-cnn-deep-leanring-project-main/sample images/orange.jpg",
    "plant-disease-prediction-cnn-deep-leanring-project-main/sample images/peach.jpg"
]

for col, img_url in zip(sample_cols, sample_images):
    with col:
        st.image(img_url, use_column_width=True)
        st.caption(f"Sample: {img_url.split('/')[-1].split('.')[0].replace('_', ' ').title()}")

# Footer
st.markdown("---")
st.markdown("""
<div style="text-align: center; padding: 20px;">
    <p>🌿 Early detection saves crops • Regular monitoring prevents outbreaks 🌿</p>
</div>
""", unsafe_allow_html=True)