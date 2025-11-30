"""
Streamlit web interface for Speech Emotion Detection.

To run this app, use:
    streamlit run app/app.py

Do NOT run this file directly with Python (python app/app.py).
"""
# Run with: streamlit run app.py
import streamlit as st
import numpy as np
import librosa
import tempfile
import os
import sys
from pathlib import Path

# Add parent directory to path
sys.path.append(str(Path(__file__).parent.parent))

# Note: If you see warnings about "missing ScriptRunContext", 
# you're running this file directly with Python. 
# To run the Streamlit app correctly, use:
#     streamlit run app/app.py

from src.feature_extraction import FeatureExtractor
from src.data_loader import EmotionDatasetLoader
from src.models.speaker_ml import SpeakerMLTrainer
from src.models.speaker_dl import SpeakerDLTrainer
from src.models.emotion_ml import EmotionMLTrainer
from src.models.emotion_dl import EmotionDLTrainer
from src.visualization import (
    plot_speaker_probabilities, plot_feature_comparison, plot_radar_chart,
    plot_similarity_scores, get_top_k_similar_speakers, load_speaker_metadata,
    get_speaker_feature_stats
)
import joblib
import matplotlib.pyplot as plt
import json
import torch
import pandas as pd


# Page config
st.set_page_config(
    page_title="Speech Analysis System",
    page_icon="🎤",
    layout="wide"
)

# Title
st.title("🎤 Speech Analysis System")
st.markdown("Analyze speech for emotion detection and speaker identification")

# Create tabs
tab1, tab2 = st.tabs(["🎭 Emotion Detection", "👤 Speaker Identification"])

# Initialize components
@st.cache_resource
def load_emotion_models():
    """Load emotion detection models (ML and DL)."""
    emotion_ml_trainer = None
    emotion_dl_trainer = None
    
    try:
        # Try loading ML models
        try:
            emotion_ml_trainer = EmotionMLTrainer()
            # Check for individual ML model files
            ml_models = ['random_forest', 'logistic_regression', 'svm', 'xgboost']
            scaler_path = 'models/scaler.pkl'
            
            found_ml = False
            if os.path.exists(scaler_path):
                for model_name in ml_models:
                    model_path = f'models/{model_name}.pkl'
                    if os.path.exists(model_path):
                        found_ml = True
                        # Load model and scaler
                        emotion_ml_trainer.model = joblib.load(model_path)
                        emotion_ml_trainer.scaler = joblib.load(scaler_path)
                        break
            
            if not found_ml:
                emotion_ml_trainer = None
        except Exception as e:
            emotion_ml_trainer = None
        
        # Try loading DL models
        try:
            # Check if DL models exist
            cnn_path = 'models/cnn_best.pth'
            lstm_path = 'models/lstm_best.pth'
            rnn_path = 'models/rnn_best.pth'
            # Also check for alternative naming
            if not os.path.exists(cnn_path):
                cnn_path = 'models/cnn.pth'
            if not os.path.exists(lstm_path):
                lstm_path = 'models/lstm.pth'
            if not os.path.exists(rnn_path):
                rnn_path = 'models/rnn.pth'
            
            if os.path.exists(cnn_path) or os.path.exists(lstm_path) or os.path.exists(rnn_path):
                # Models exist, initialize trainer (models will be loaded on demand)
                emotion_dl_trainer = EmotionDLTrainer()
        except Exception as e:
            emotion_dl_trainer = None
        
        # Return models if at least one type exists
        if emotion_ml_trainer is None and emotion_dl_trainer is None:
            return None, None
        
        return emotion_ml_trainer, emotion_dl_trainer
    except Exception as e:
        return None, None

@st.cache_resource
def load_speaker_models():
    """Load speaker identification models."""
    speaker_identifier = None
    dl_identifier = None
    
    try:
        # Try loading ML models
        try:
            speaker_identifier = SpeakerMLTrainer()
            # Try to load all models from metadata file
            metadata_path = 'models/speaker_models_metadata.pkl'
            if os.path.exists(metadata_path):
                try:
                    speaker_identifier.load_model(filepath=metadata_path)
                except Exception as e:
                    speaker_identifier = None
            else:
                # No metadata file found, check for individual model files
                individual_models = ['random_forest', 'logistic_regression', 'svm', 'xgboost']
                found_models = False
                for model_name in individual_models:
                    model_path = f'models/speaker_{model_name}.pkl'
                    if os.path.exists(model_path):
                        found_models = True
                        try:
                            speaker_identifier.load_model(model_name=model_name)
                            break  # Load first available model
                        except:
                            continue
                
                if not found_models:
                    speaker_identifier = None
        except Exception as e:
            speaker_identifier = None
        
        # Try loading DL models
        try:
            # Check if models exist (new naming: speaker_{model_type}.pth)
            cnn_path = 'models/speaker_cnn.pth'
            lstm_path = 'models/speaker_lstm.pth'
            rnn_path = 'models/speaker_rnn.pth'
            if os.path.exists(cnn_path) or os.path.exists(lstm_path) or os.path.exists(rnn_path):
                # Models exist, initialize trainer (models will be loaded on demand)
                dl_identifier = SpeakerDLTrainer()
        except Exception as e:
            dl_identifier = None
        
        # Return models if at least one type exists
        if speaker_identifier is None and dl_identifier is None:
            return None, None, None
        
        return speaker_identifier, dl_identifier, load_speaker_metadata()
    except Exception as e:
        return None, None, None

# Emotion Detection Tab
with tab1:
    st.header("🎭 Emotion Detection")
    st.markdown("Upload an audio file to detect emotions!")
    
    emotion_ml_trainer, emotion_dl_trainer = load_emotion_models()
    
    # File upload FIRST
    uploaded_file = st.file_uploader(
        "Choose an audio file",
        type=['wav', 'mp3', 'flac', 'm4a'],
        key="emotion_upload"
    )
    
    if uploaded_file is not None:
        # Save uploaded file temporarily
        with tempfile.NamedTemporaryFile(delete=False, suffix='.wav') as tmp_file:
            tmp_file.write(uploaded_file.read())
            tmp_path = tmp_file.name
        
        try:
            # Load and preprocess audio
            loader = EmotionDatasetLoader()
            audio = loader.preprocess_audio(tmp_path)
            
            if len(audio) > 0:
                # Display audio info
                col1, col2 = st.columns(2)
                with col1:
                    st.audio(tmp_path)
                with col2:
                    st.metric("Duration", f"{len(audio)/22050:.2f}s")
                    st.metric("Sample Rate", "22050 Hz")
                
                # Extract features (needed for both ML and DL)
                extractor = FeatureExtractor()
                features = extractor.extract_all_features(audio)
                
                # Model selection AFTER file upload
                st.header("Select Model")
                
                # Determine available model types
                available_types = []
                if emotion_ml_trainer is not None:
                    available_types.append("ML Models")
                if emotion_dl_trainer is not None:
                    available_types.append("DL Models")
                if emotion_ml_trainer is not None:
                    available_types.append("Best Model")
                
                if len(available_types) == 0:
                    st.warning("⚠️ No models available. Please train models first.")
                    st.info("""
                    **To train emotion detection models:**
                    
                    1. **Train ML models:**
                       - `python src/models/emotion_ml.py`
                       - OR: `python main.py --mode train_emotion_ml`
                    
                    2. **Train DL models:**
                       - `python src/models/emotion_dl.py`
                       - OR: `python main.py --mode train_emotion_dl`
                    
                    3. Models will be saved in the `models/` directory
                    """)
                    os.unlink(tmp_path)
                    st.stop()
                
                model_type = st.selectbox(
                    "Select Model Type",
                    available_types,
                    key="emotion_model_type"
                )
                
                selected_model = None
                model_available = False
                
                if model_type == "ML Models":
                    if emotion_ml_trainer is None:
                        st.warning("⚠️ ML models are not available. Please train ML models first.")
                        st.info("""
                        **To train ML models:**
                        - `python src/models/emotion_ml.py`
                        - OR: `python main.py --mode train_emotion_ml`
                        """)
                        model_available = False
                    else:
                        available_models = ['random_forest', 'logistic_regression', 'svm', 'xgboost']
                        selected_model = st.selectbox(
                            "Select ML Model",
                            available_models,
                            key="emotion_ml_model_select"
                        )
                        model_available = True
                        
                elif model_type == "DL Models":
                    if emotion_dl_trainer is None:
                        st.warning("⚠️ DL models are not available. Please train DL models first.")
                        st.info("""
                        **To train DL models:**
                        - `python src/models/emotion_dl.py`
                        - OR: `python main.py --mode train_emotion_dl`
                        """)
                        model_available = False
                    else:
                        selected_model = st.selectbox(
                            "Select DL Model",
                            ['cnn', 'lstm', 'rnn'],
                            key="emotion_dl_model_select"
                        )
                        model_available = True
                        
                elif model_type == "Best Model":
                    if emotion_ml_trainer is None:
                        st.warning("⚠️ Best model (ML) is not available. Please train ML models first.")
                        st.info("""
                        **To train ML models:**
                        - `python src/models/emotion_ml.py`
                        - OR: `python main.py --mode train_emotion_ml`
                        """)
                        model_available = False
                    else:
                        model_available = True
                
                # Only proceed with prediction if model is available
                if model_available:
                    # Predict
                    st.header("Prediction Results")
                    
                    emotions = ['angry', 'disgust', 'fear', 'happy', 'neutral', 'sad', 'surprise']
                    
                    if model_type == "ML Models" and selected_model and emotion_ml_trainer:
                        # Use specific ML model
                        try:
                            # Load the selected model
                            model_path = f'models/{selected_model}.pkl'
                            scaler_path = 'models/scaler.pkl'
                            if os.path.exists(model_path) and os.path.exists(scaler_path):
                                model = joblib.load(model_path)
                                scaler = joblib.load(scaler_path)
                                
                                # Scale features
                                features_scaled = scaler.transform(features.reshape(1, -1))
                                
                                # Predict
                                prediction = model.predict(features_scaled)[0]
                                probabilities = model.predict_proba(features_scaled)[0]
                            else:
                                st.error(f"Model file {model_path} not found")
                                st.stop()
                        except Exception as e:
                            st.error(f"Error with {selected_model}: {e}")
                            st.stop()
                            
                    elif model_type == "DL Models" and selected_model and emotion_dl_trainer:
                        # Use DL model
                        try:
                            # Load the DL model
                            model_path = f'models/{selected_model}_best.pth'
                            if not os.path.exists(model_path):
                                model_path = f'models/{selected_model}.pth'
                            
                            if not os.path.exists(model_path):
                                st.error(f"DL model file {model_path} not found")
                                st.stop()
                            
                            # Load model
                            model = emotion_dl_trainer.load_model(model_path)
                            emotion_dl_trainer.trained_models[selected_model] = model
                            
                            # Prepare data for DL model
                            if selected_model == 'cnn':
                                spectrograms = emotion_dl_trainer.prepare_spectrograms([audio], extractor)
                                # Convert to tensor and predict
                                if len(spectrograms.shape) == 4 and spectrograms.shape[-1] == 1:
                                    spectrograms = np.transpose(spectrograms, (0, 3, 1, 2))
                                X_tensor = torch.from_numpy(spectrograms).float().to(emotion_dl_trainer.device)
                            else:  # lstm or rnn
                                sequences = emotion_dl_trainer.prepare_sequences([audio], extractor)
                                X_tensor = torch.from_numpy(sequences).float().to(emotion_dl_trainer.device)
                            
                            # Predict
                            model.eval()
                            with torch.no_grad():
                                outputs = model(X_tensor)
                                probabilities_tensor = torch.softmax(outputs, dim=1)
                                probabilities = probabilities_tensor.cpu().numpy()[0]
                                prediction = np.argmax(probabilities)
                            
                            # Decode label if label encoder available
                            if hasattr(emotion_dl_trainer, 'label_encoder') and emotion_dl_trainer.label_encoder is not None:
                                try:
                                    predicted_emotion = emotion_dl_trainer.label_encoder.inverse_transform([prediction])[0]
                                    pred_idx = emotions.index(predicted_emotion) if predicted_emotion in emotions else int(prediction)
                                except:
                                    pred_idx = int(prediction)
                                    predicted_emotion = emotions[pred_idx] if pred_idx < len(emotions) else emotions[0]
                            else:
                                pred_idx = int(prediction)
                                predicted_emotion = emotions[pred_idx] if pred_idx < len(emotions) else emotions[0]
                            
                        except Exception as e:
                            st.error(f"Error with DL model: {e}")
                            st.stop()
                            
                    elif model_type == "Best Model" and emotion_ml_trainer:
                        # Use best ML model (default: random_forest)
                        try:
                            model_path = 'models/random_forest.pkl'
                            scaler_path = 'models/scaler.pkl'
                            if os.path.exists(model_path) and os.path.exists(scaler_path):
                                model = joblib.load(model_path)
                                scaler = joblib.load(scaler_path)
                                
                                # Scale features
                                features_scaled = scaler.transform(features.reshape(1, -1))
                                
                                # Predict
                                prediction = model.predict(features_scaled)[0]
                                probabilities = model.predict_proba(features_scaled)[0]
                            else:
                                st.error("Best model (random_forest) not found")
                                st.stop()
                        except Exception as e:
                            st.error(f"Error with best model: {e}")
                            st.stop()
                    else:
                        st.error("No suitable model available for prediction")
                        st.stop()
                    
                    # Handle prediction format
                    if model_type != "DL Models":
                        # ML models prediction
                        if isinstance(prediction, str):
                            predicted_emotion = prediction
                            try:
                                pred_idx = emotions.index(predicted_emotion)
                            except ValueError:
                                pred_idx = 0
                        else:
                            pred_idx = int(prediction)
                            predicted_emotion = emotions[pred_idx] if pred_idx < len(emotions) else emotions[0]
                    
                    # Display prediction
                    col1, col2 = st.columns(2)
                    with col1:
                        st.success(f"**Predicted Emotion: {predicted_emotion.upper()}**")
                    with col2:
                        confidence = probabilities[pred_idx] * 100 if pred_idx < len(probabilities) else 0
                        st.metric("Confidence", f"{confidence:.2f}%")
                    
                    # Probability bar chart
                    st.subheader("Emotion Probabilities")
                    fig, ax = plt.subplots(figsize=(10, 6))
                    ax.barh(emotions, probabilities * 100)
                    ax.set_xlabel('Probability (%)')
                    ax.set_title('Emotion Prediction Probabilities')
                    ax.set_xlim(0, 100)
                    st.pyplot(fig)
                    
                    # Audio waveform
                    st.subheader("Audio Waveform")
                    fig, ax = plt.subplots(figsize=(12, 4))
                    time_axis = np.linspace(0, len(audio)/22050, len(audio))
                    ax.plot(time_axis, audio)
                    ax.set_xlabel('Time (s)')
                    ax.set_ylabel('Amplitude')
                    ax.set_title('Audio Waveform')
                    st.pyplot(fig)
                
                # Clean up
                os.unlink(tmp_path)
        except Exception as e:
            st.error(f"Error processing audio: {e}")
            if os.path.exists(tmp_path):
                os.unlink(tmp_path)

# Speaker Identification Tab
with tab2:
    st.header("👤 Speaker Identification")
    st.markdown("Upload an audio file to identify the speaker!")
    
    speaker_identifier, dl_identifier, speaker_metadata = load_speaker_models()
    
    # File upload FIRST
    uploaded_file = st.file_uploader(
        "Choose an audio file",
        type=['wav', 'mp3', 'flac', 'm4a'],
        key="speaker_upload"
    )
    
    if uploaded_file is not None:
        # Save uploaded file temporarily
        with tempfile.NamedTemporaryFile(delete=False, suffix='.wav') as tmp_file:
            tmp_file.write(uploaded_file.read())
            tmp_path = tmp_file.name
        
        try:
            # Load and preprocess audio
            loader = EmotionDatasetLoader()
            audio = loader.preprocess_audio(tmp_path)
            
            if len(audio) > 0:
                # Display audio info
                col1, col2 = st.columns(2)
                with col1:
                    st.audio(tmp_path)
                with col2:
                    st.metric("Duration", f"{len(audio)/22050:.2f}s")
                    st.metric("Sample Rate", "22050 Hz")
                
                # Extract features (needed for both ML and DL)
                extractor = FeatureExtractor()
                features = extractor.extract_all_features(audio)
                
                # Model selection AFTER file upload
                st.header("Select Model")
                
                # Determine available model types
                available_types = []
                if speaker_identifier is not None:
                    available_types.append("ML Models")
                if dl_identifier is not None:
                    available_types.append("DL Models")
                if speaker_identifier is not None:
                    available_types.append("Best Model")
                
                if len(available_types) == 0:
                    st.warning("⚠️ No models available. Please train models first.")
                    st.info("""
                    **To train speaker identification models:**
                    
                    1. **Train ML models:**
                       - `python src/models/speaker_ml.py`
                       - OR: `python main.py --mode train_speaker_ml`
                    
                    2. **Train DL models:**
                       - `python src/models/speaker_dl.py`
                       - OR: `python main.py --mode train_speaker_dl`
                    
                    3. Models will be saved in the `models/` directory
                    """)
                    os.unlink(tmp_path)
                    st.stop()
                
                model_type = st.selectbox(
                    "Select Model Type",
                    available_types,
                    key="speaker_model_type"
                )
                
                selected_model = None
                model_available = False
                
                if model_type == "ML Models":
                    if speaker_identifier is None:
                        st.warning("⚠️ ML models are not available. Please train ML models first.")
                        st.info("""
                        **To train ML models:**
                        - `python src/models/speaker_ml.py`
                        - OR: `python main.py --mode train_speaker_ml`
                        """)
                        model_available = False
                    else:
                        available_models = ['random_forest', 'logistic_regression', 'svm', 'xgboost']
                        selected_model = st.selectbox(
                            "Select ML Model",
                            available_models,
                            key="ml_model_select"
                        )
                        model_available = True
                        
                elif model_type == "DL Models":
                    if dl_identifier is None:
                        st.warning("⚠️ DL models are not available. Please train DL models first.")
                        st.info("""
                        **To train DL models:**
                        - `python src/models/speaker_dl.py`
                        - OR: `python main.py --mode train_speaker_dl`
                        """)
                        model_available = False
                    else:
                        selected_model = st.selectbox(
                            "Select DL Model",
                            ['cnn', 'lstm', 'rnn'],
                            key="dl_model_select"
                        )
                        model_available = True
                        
                elif model_type == "Best Model":
                    if speaker_identifier is None:
                        st.warning("⚠️ Best model (ML) is not available. Please train ML models first.")
                        st.info("""
                        **To train ML models:**
                        - `python src/models/speaker_ml.py`
                        - OR: `python main.py --mode train_speaker_ml`
                        """)
                        model_available = False
                    else:
                        model_available = True
                
                # Only proceed with prediction if model is available
                if model_available:
                    # Predict
                    st.header("Prediction Results")
                    
                    if model_type == "ML Models" and selected_model and speaker_identifier:
                        # Use specific ML model
                        try:
                            predictions, probabilities = speaker_identifier.predict(
                                features.reshape(1, -1), model_name=selected_model
                            )
                            predicted_speaker = predictions[0]
                            confidence = probabilities[0].max() * 100
                            all_probs = probabilities[0]
                        except Exception as e:
                            st.error(f"Error with {selected_model}: {e}")
                            if speaker_identifier:
                                predictions, probabilities = speaker_identifier.predict(features.reshape(1, -1))
                                predicted_speaker = predictions[0]
                                confidence = probabilities[0].max() * 100
                                all_probs = probabilities[0]
                            else:
                                st.error("ML models not available")
                                st.stop()
                    elif model_type == "DL Models" and selected_model and dl_identifier:
                        # Use DL model
                        try:
                            # Prepare data for DL model
                            if selected_model == 'cnn':
                                spectrograms = dl_identifier.prepare_spectrograms([audio], extractor)
                                predictions, probabilities = dl_identifier.predict(spectrograms, 'cnn')
                            else:  # lstm or rnn
                                sequences = dl_identifier.prepare_sequences([audio], extractor)
                                predictions, probabilities = dl_identifier.predict(sequences, selected_model)
                            
                            predicted_speaker = predictions[0]
                            confidence = probabilities[0].max() * 100
                            all_probs = probabilities[0]
                        except Exception as e:
                            st.error(f"Error with DL model: {e}")
                            st.stop()
                    elif model_type == "Best Model" and speaker_identifier:
                        # Use best ML model
                        predictions, probabilities = speaker_identifier.predict(features.reshape(1, -1))
                        predicted_speaker = predictions[0]
                        confidence = probabilities[0].max() * 100
                        all_probs = probabilities[0]
                    else:
                        st.error("No suitable model available for prediction")
                        st.stop()
                    
                    # Display prediction
                    col1, col2 = st.columns(2)
                    with col1:
                        if confidence < 50:
                            st.warning(f"**Predicted Speaker: {predicted_speaker}** (Low Confidence)")
                        else:
                            st.success(f"**Predicted Speaker: {predicted_speaker}**")
                    with col2:
                        st.metric("Confidence", f"{confidence:.2f}%")
                    
                    # Warning for low confidence
                    if confidence < 50:
                        st.warning("⚠️ Low confidence prediction. The audio may not match any known speaker well.")
                    
                    # Get all speaker IDs from label encoder
                    # Use DL label encoder if ML models not available
                    if speaker_identifier and hasattr(speaker_identifier, 'label_encoder') and speaker_identifier.label_encoder is not None:
                        all_speakers = speaker_identifier.label_encoder.classes_.tolist()
                    elif dl_identifier and hasattr(dl_identifier, 'label_encoder') and dl_identifier.label_encoder is not None:
                        all_speakers = dl_identifier.label_encoder.classes_.tolist()
                    else:
                        # Try to get from predictions if label encoder not available
                        all_speakers = list(set([predicted_speaker] + [f"Speaker_{i}" for i in range(len(all_probs))]))
                    
                    # Create probability dictionary
                    speaker_probs = {speaker: prob for speaker, prob in zip(all_speakers, all_probs)}
                    
                    # Top-K speakers
                    st.subheader("Top-K Most Likely Speakers")
                    top_k = st.slider("Number of top speakers to show", 3, 10, 5, key="top_k_slider")
                    sorted_speakers = sorted(speaker_probs.items(), key=lambda x: x[1], reverse=True)[:top_k]
                    
                    # Display top-K
                    cols = st.columns(min(top_k, 5))
                    for i, (speaker, prob) in enumerate(sorted_speakers):
                        with cols[i % 5]:
                            st.metric(speaker, f"{prob*100:.1f}%")
                    
                    # Probability distribution chart
                    st.subheader("Probability Distribution")
                    fig = plot_speaker_probabilities(speaker_probs, top_k=top_k)
                    st.pyplot(fig)
                    plt.close(fig)
                    
                    # Full Data of ID Holder Section
                    st.header("Full Data of Predicted Speaker")
                    
                    if speaker_metadata and 'speakers' in speaker_metadata:
                        if predicted_speaker in speaker_metadata['speakers']:
                            speaker_data = speaker_metadata['speakers'][predicted_speaker]
                            
                            # Training Data Info
                            st.subheader("Training Data Information")
                            col1, col2, col3 = st.columns(3)
                            with col1:
                                st.metric("Training Samples", speaker_data.get('num_samples', 'N/A'))
                            with col2:
                                # Determine dataset
                                dataset = "Unknown"
                                if predicted_speaker.startswith('Actor_'):
                                    dataset = "RAVDESS"
                                elif predicted_speaker.isdigit() and len(predicted_speaker) == 4:
                                    dataset = "CREMA-D"
                                elif predicted_speaker in ['DC', 'JE', 'JK', 'KL']:
                                    dataset = "SAVEE"
                                else:
                                    dataset = "TESS"
                                st.metric("Dataset Source", dataset)
                            with col3:
                                if speaker_metadata.get('datasets'):
                                    total = sum(speaker_metadata['datasets'].values())
                                    st.metric("Total Dataset Samples", total)
                            
                            # Feature Statistics
                            st.subheader("Feature Statistics")
                            if 'feature_mean' in speaker_data:
                                feature_stats = {
                                    'Mean': speaker_data.get('feature_mean', []),
                                    'Std': speaker_data.get('feature_std', []),
                                    'Min': speaker_data.get('feature_min', []),
                                    'Max': speaker_data.get('feature_max', [])
                                }
                                
                                # Show summary statistics
                                stats_df = {
                                    'Statistic': ['Mean', 'Std', 'Min', 'Max'],
                                    'Value': [
                                        np.mean(speaker_data.get('feature_mean', [])),
                                        np.mean(speaker_data.get('feature_std', [])),
                                        np.mean(speaker_data.get('feature_min', [])),
                                        np.mean(speaker_data.get('feature_max', []))
                                    ]
                                }
                                import pandas as pd
                                st.dataframe(pd.DataFrame(stats_df), use_container_width=True)
                        else:
                            st.info(f"Metadata not available for speaker: {predicted_speaker}")
                    else:
                        st.info("Speaker metadata not available. Please train models to generate metadata.")
                    
                    # Model Confidence Scores
                    st.subheader("Model Confidence Scores")
                    if model_type == "ML Models":
                        # Get predictions from all ML models
                        try:
                            if speaker_identifier and hasattr(speaker_identifier, 'predict_all_models'):
                                all_predictions = speaker_identifier.predict_all_models(features.reshape(1, -1))
                            else:
                                all_predictions = {}
                            model_confidences = {}
                            for model_name, pred_data in all_predictions.items():
                                pred_speaker = pred_data['predictions'][0]
                                conf = pred_data['probabilities'][0].max() * 100
                                model_confidences[model_name] = {
                                    'speaker': pred_speaker,
                                    'confidence': conf
                                }
                            
                            # Display in columns
                            cols = st.columns(len(model_confidences))
                            for i, (model_name, data) in enumerate(model_confidences.items()):
                                with cols[i]:
                                    st.metric(model_name.replace('_', ' ').title(), 
                                            f"{data['confidence']:.1f}%")
                                    st.caption(f"Speaker: {data['speaker']}")
                        except:
                            st.info("Could not load all model predictions")
                    
                    # Comparison Section
                    st.header("Comparison with Other Speakers")
                    
                    # Top-K Similar Speakers
                    st.subheader("Top-K Similar Speakers")
                    if speaker_metadata and 'speakers' in speaker_metadata:
                        # Get average features for all speakers
                        all_speaker_features = {}
                        for speaker_id, data in speaker_metadata['speakers'].items():
                            if 'feature_mean' in data:
                                all_speaker_features[speaker_id] = np.array(data['feature_mean'])
                        
                        if all_speaker_features:
                            similar_speakers = get_top_k_similar_speakers(
                                features, all_speaker_features, k=top_k
                            )
                            
                            # Remove the predicted speaker from the list
                            similar_speakers = [s for s in similar_speakers if s[0] != predicted_speaker][:top_k]
                            
                            if similar_speakers:
                                fig = plot_similarity_scores(similar_speakers)
                                st.pyplot(fig)
                                plt.close(fig)
                                
                                # Feature comparison with top similar speaker
                                if len(similar_speakers) > 0:
                                    st.subheader(f"Feature Comparison: {predicted_speaker} vs {similar_speakers[0][0]}")
                                    compare_speakers = {
                                        predicted_speaker: features,
                                        similar_speakers[0][0]: all_speaker_features.get(similar_speakers[0][0], features)
                                    }
                                    
                                    try:
                                        fig = plot_feature_comparison(compare_speakers, top_features=10)
                                        st.pyplot(fig)
                                        plt.close(fig)
                                    except:
                                        st.info("Feature comparison visualization not available")
                            else:
                                st.info("No similar speakers found")
                        else:
                            st.info("Speaker feature data not available for comparison")
                    else:
                        st.info("Speaker metadata required for comparison. Please train models.")
                
                # Clean up
                os.unlink(tmp_path)
        except Exception as e:
            st.error(f"Error processing audio: {e}")
            if os.path.exists(tmp_path):
                os.unlink(tmp_path)

# Footer
st.markdown("---")
st.markdown("**Speech Analysis System** | Built with Streamlit")

# Auto-launch streamlit if run directly with Python (for IDE run button)
if __name__ == "__main__":
    # Check if we're being run by streamlit (streamlit sets __file__ differently)
    # If not, launch streamlit automatically
    try:
        # Try to access streamlit's runtime - if this fails, we're not in streamlit
        from streamlit.runtime.scriptrunner import get_script_run_ctx
        if get_script_run_ctx() is None:
            raise ImportError("Not in streamlit context")
    except (ImportError, AttributeError):
        # Not running in streamlit, launch it
        import subprocess
        import sys
        file_path = os.path.abspath(__file__)
        print(f"Launching Streamlit for {file_path}...")
        subprocess.run([sys.executable, "-m", "streamlit", "run", file_path])

