import librosa
import numpy as np
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout, Flatten, Conv1D, MaxPooling1D, LSTM
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from tensorflow.keras.utils import to_categorical
from imblearn.over_sampling import SMOTE
import os

# --- 1. Feature Extraction (MFCCs, Chroma, Mel-spectrograms) ---
def extract_features(file_path):
    """
    Extracts a combination of features from an audio file.
    """
    try:
        audio, sample_rate = librosa.load(file_path, res_type='kaiser_fast')
        mfccs = np.mean(librosa.feature.mfcc(y=audio, sr=sample_rate, n_mfcc=40).T, axis=0)
        chroma = np.mean(librosa.feature.chroma_stft(y=audio, sr=sample_rate).T, axis=0)
        mel = np.mean(librosa.feature.melspectrogram(y=audio, sr=sample_rate).T, axis=0)
        
        combined_features = np.hstack([mfccs, chroma, mel])
    except Exception as e:
        print(f"Error extracting features from file: {file_path}")
        print(f"Details: {e}")
        return None
    return combined_features

# --- 2. Load and Preprocess the RAVDESS Dataset ---
def load_ravdess_data(data_path):
    """
    Loads audio files from the RAVDESS dataset, extracts features, and labels.
    """
    features = []
    labels = []
    
    # Define a helper function to extract emotion from filename
    def get_emotion(filename):
        emotion_code = int(filename.split('-')[2])
        emotion_map = {
            1: 'neutral', 2: 'calm', 3: 'happy', 4: 'sad', 
            5: 'angry', 6: 'fearful', 7: 'disgust', 8: 'surprised'
        }
        return emotion_map.get(emotion_code)

    # Walk through the dataset directory to find all audio files
    for root, dirs, files in os.walk(data_path):
        for audio_file in files:
            if audio_file.endswith('.wav'):
                file_path = os.path.join(root, audio_file)
                try:
                    label = get_emotion(audio_file)
                    mfccs = extract_features(file_path)
                    if mfccs is not None:
                        features.append(mfccs)
                        labels.append(label)
                except:
                    continue
    
    return np.array(features), np.array(labels)

# --- MAIN SCRIPT EXECUTION ---
while True:
    ravdess_path = input("Please enter the full path to your RAVDESS dataset folder: ")
    if os.path.exists(ravdess_path) and os.path.isdir(ravdess_path):
        break
    else:
        print(f"Error: The path '{ravdess_path}' does not exist or is not a directory. Please enter a valid path.")

print("Loading and preprocessing RAVDESS dataset...")
X_raw, y_raw = load_ravdess_data(ravdess_path)

if len(X_raw) == 0:
    print("No audio files found in the provided path. Please check the path and file structure.")
else:
    print(f"\nLoaded {len(X_raw)} samples from the dataset.")

    print("Balancing the dataset using SMOTE...")
    smote = SMOTE(random_state=42)
    X_balanced, y_balanced = smote.fit_resample(X_raw, y_raw)
    print(f"Balanced dataset size: {len(X_balanced)} samples.")

    label_encoder = LabelEncoder()
    y_encoded = label_encoder.fit_transform(y_balanced)
    y_categorical = to_categorical(y_encoded)
    X_train, X_test, y_train, y_test = train_test_split(X_balanced, y_categorical, test_size=0.2, random_state=42)

    X_train = np.expand_dims(X_train, axis=2)
    X_test = np.expand_dims(X_test, axis=2)

    # --- 3. Build and Train the CNN-LSTM Model ---
    model = Sequential()
    model.add(Conv1D(filters=128, kernel_size=5, activation='relu', input_shape=(X_train.shape[1], 1)))
    model.add(MaxPooling1D(pool_size=2))
    model.add(Dropout(0.3))
    model.add(LSTM(64, return_sequences=True))
    model.add(Dropout(0.3))
    model.add(LSTM(64))
    model.add(Dense(256, activation='relu'))
    model.add(Dropout(0.3))
    model.add(Dense(y_categorical.shape[1], activation='softmax'))
    model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])

    print("\nTraining the model...")
    model.fit(X_train, y_train, epochs=50, batch_size=64, validation_data=(X_test, y_test), verbose=1)

    # --- 4. Make a Prediction on a New Audio File ---
    while True:
        audio_file_path = input("\n\nPlease enter the full path to your audio file for prediction: ")
        if os.path.exists(audio_file_path) and os.path.isfile(audio_file_path):
            break
        else:
            print(f"Error: The file path '{audio_file_path}' does not exist or is not a file.")

    new_audio_features = extract_features(audio_file_path)

    if new_audio_features is not None:
        new_audio_features = np.expand_dims(new_audio_features, axis=0)
        new_audio_features = np.expand_dims(new_audio_features, axis=2)
        
        prediction = model.predict(new_audio_features)
        predicted_label_index = np.argmax(prediction)
        predicted_emotion = label_encoder.inverse_transform([predicted_label_index])[0]
        
        print(f"\nThe predicted emotion for your audio is: {predicted_emotion}")
    else:
        print("\nCould not process the audio file. Please check the path and format.")