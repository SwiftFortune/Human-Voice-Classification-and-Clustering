import streamlit as st
import pandas as pd
import numpy as np
import joblib
import os
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier

# ---------- Page Config ----------
st.set_page_config(page_title="Voice Gender Prediction", layout="centered")

# ---------- Load Data ----------
@st.cache_data
def load_data():
    # 👇 Change to your actual CSV file path or allow file uploader
    return pd.read_csv(r"C:\sachin\Python\Classification project\vocal_gender_features_new.csv")

data = load_data()

# ---------- Prepare Features ----------
top_12_features_info = {
    'mfcc_4_std': (12.0, 59.0),
    'mfcc_5_std': (4.0, 44.0),
    'mfcc_4_mean': (-7.0, 92.0),
    'mfcc_5_mean': (-50.0, 22.0),
    'mean_spectral_flatness': (0.0017, 0.0721),
    'mfcc_1_mean': (-448.0, -162.0),
    'mfcc_7_mean': (-34.0, 18.0),
    'mfcc_1_std': (52.0, 206.0),
    'mfcc_8_mean': (-34.0, 18.0),
    'zero_crossing_rate': (0.02, 0.27),
    'mfcc_2_std': (19.0, 109.0),
    'mfcc_10_mean': (-20.0, 19.0)
}

features = list(top_12_features_info.keys())

# ---------- Train and Save Model if not exist ----------
@st.cache_resource
def train_and_save_model():
    X = data[features]
    y = data['label']  # assumed to be encoded 0/1

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)

    model = RandomForestClassifier(n_estimators=100, random_state=42)
    model.fit(X_train_scaled, y_train)

    # Save to files
    joblib.dump(model, "voice_gender_model.pkl")
    joblib.dump(scaler, "voice_prediction.pkl")

    return model, scaler

# Load or train model
if os.path.exists("voice_gender_model.pkl") and os.path.exists("voice_prediction.pkl"):
    model = joblib.load("voice_gender_model.pkl")
    scaler = joblib.load("voice_prediction.pkl")
else:
    model, scaler = train_and_save_model()

# ---------- Generate Sample Data for Visualization ----------
def generate_sample_data(num_samples=100):
    return pd.DataFrame({
        feature: np.random.uniform(low, high, num_samples)
        for feature, (low, high) in top_12_features_info.items()
    })

# ---------- Sidebar Navigation ----------
st.sidebar.title("🔍 Navigation")
selection = st.sidebar.radio("Select Page", ["📘 Intro", "📊 Visualization", "🧠 Feature Ranges", "🔮 Predict Gender", "👤 About Me"])

# ---------- Intro ----------
if selection == "📘 Intro":
    st.title("🔊 Voice Gender Prediction App")
    st.markdown("""
This interactive Streamlit application is designed to predict gender (Male or Female) based on audio-derived features from voice recordings. It provides a user-friendly interface to explore the data, visualize important features, and make real-time predictions.

🛠️ How the App Works
📂 Loads Voice Data
The app begins by loading a dataset containing audio features such as MFCCs, zero-crossing rate, and spectral flatness. These features are known to carry meaningful patterns for gender classification.

🧪 Trains a Machine Learning Model
A RandomForestClassifier is used as the core model. This ensemble-based classifier is well-suited for handling high-dimensional data and works effectively with both linear and non-linear relationships.

⚖️ Applies Feature Scaling
Features are scaled using StandardScaler to normalize the input values, ensuring the model performs optimally and is not biased by features with larger numeric ranges.

💾 Saves the Trained Model and Scaler
Once training is complete, both the model and the scaler are saved locally using joblib. This allows the app to reuse the trained model for future predictions without retraining.

🎛️ Interactive Gender Prediction
Users can input feature values using intuitive sliders and get instant gender predictions. The app transforms and scales the inputs before feeding them into the trained model to produce accurate results.


    """)

# ---------- Visualization ----------
elif selection == "📊 Visualization":
    st.title("📊 Feature Distribution & Statistics")
    st.markdown("""
    Select a feature below to explore its distribution and summary statistics.  
    This helps you understand how each feature varies across samples.
    """)

    df = generate_sample_data()
    feature = st.selectbox("🔍 Choose a feature to visualize", df.columns)

    col1, col2 = st.columns(2)

    with col1:
        st.markdown(f"### 📈 Histogram of `{feature}`")
        fig, ax = plt.subplots()
        ax.hist(df[feature], bins=30, color='cornflowerblue', edgecolor='black')
        ax.set_xlabel(f"{feature}")
        ax.set_ylabel("Frequency")
        ax.grid(True, linestyle='--', alpha=0.5)
        st.pyplot(fig)

    with col2:
        st.markdown("### 📊 Summary Statistics")
        stats = df[feature].describe().rename({
            "mean": "Mean", "std": "Std Dev", "min": "Minimum", 
            "25%": "25th Percentile", "50%": "Median", 
            "75%": "75th Percentile", "max": "Maximum"
        })
        st.dataframe(stats.to_frame(name="Value"))

    # Optional boxplot
    st.markdown("---")
    with st.expander("📦 Show Boxplot"):
        fig2, ax2 = plt.subplots()
        ax2.boxplot(df[feature], vert=False, patch_artist=True,
                    boxprops=dict(facecolor='lightblue', color='blue'),
                    medianprops=dict(color='red'))
        ax2.set_xlabel(f"{feature}")
        st.pyplot(fig2)


# ---------- Feature Ranges ----------
elif selection == "🧠 Feature Ranges":
    st.title("🧠 Top 12 Audio Feature Ranges")
    st.markdown("""
    Below are the **12 most important audio features** selected for predicting voice gender.  
    The table includes the expected **minimum and maximum values** based on the training dataset.
    """)

    # Create DataFrame for display
    feature_df = pd.DataFrame(top_12_features_info).T.reset_index()
    feature_df.columns = ['Feature Name', 'Min Value', 'Max Value']
    feature_df = feature_df.sort_values(by='Feature Name')

    st.dataframe(feature_df.style.format({
        "Min Value": "{:.4f}",
        "Max Value": "{:.4f}"
    }).background_gradient(cmap='Blues', subset=["Min Value", "Max Value"]))

    st.markdown("---")

    with st.expander("ℹ️ What do these features represent?"):
        st.markdown("""
        #### 🎵 **MFCCs (Mel-Frequency Cepstral Coefficients)**  
        Represent the short-term power spectrum of a sound. They capture the **tone and timbre** of the voice, which are crucial in distinguishing different speakers.

        #### 🔉 **Spectral Flatness**  
        Measures how noise-like a sound is. A higher value means the sound is more like noise; a lower value means it's more musical or tonal.

        #### 📈 **Zero-Crossing Rate (ZCR)**  
        Indicates how frequently the signal crosses zero — i.e., changes sign. It’s useful for identifying **noisy** vs. **stable (voiced)** sounds.

        ---

        ### 🧠 **Why They Matter for Gender Prediction**
        Male and female voices differ in **pitch**, **tone**, and **energy** patterns.  
        These features help the machine learning model capture those subtle acoustic differences to **accurately predict gender** from voice data.
        """)


# ---------- Predict Gender ----------
    # --- Collect input with sliders ---
elif selection == "🔮 Predict Gender":
    st.title("🔮 Predict Gender from Voice Features")
    st.markdown("""
    Adjust the sliders below to input voice feature values, then click **🚀 Predict Gender**  
    to let the model determine whether the voice is more likely to be **Male** or **Female**.
    """)

    # --- Collect input with sliders ---
    st.subheader("🎛️ Input Audio Features")
    st.markdown("Use the sliders below to simulate or fine-tune audio input values.")

    input_data = {}
    cols = st.columns(2)  # Two-column layout for sliders

    for i, (feature, (min_val, max_val)) in enumerate(top_12_features_info.items()):
        step = (max_val - min_val) / 100.0
        default_val = (min_val + max_val) / 2
        with cols[i % 2]:
            input_data[feature] = st.slider(
                label=f"{feature}",
                min_value=float(min_val),
                max_value=float(max_val),
                value=float(default_val),
                step=step
            )

    # --- Predict Button ---
    st.markdown("---")
    if st.button("🚀 Predict Gender"):
        try:
            # Prepare input
            input_array = np.array([list(input_data.values())])
            input_scaled = scaler.transform(input_array)

            # Predict
            prediction = model.predict(input_scaled)[0]
            probas = model.predict_proba(input_scaled)[0] if hasattr(model, "predict_proba") else [None, None]
            gender = "Male" if prediction == 1 else "Female"
            gender_color = "blue" if gender == "Male" else "magenta"

            # --- Display result ---
            st.markdown(f"### 🎯 **Predicted Gender:** <span style='color:{gender_color}'>{gender}</span>", unsafe_allow_html=True)

            # Show probability if available
            if probas[0] is not None:
                confidence = probas[prediction] * 100
                st.markdown(f"🧪 **Model Confidence**: `{confidence:.2f}%`")

            # Display relevant image
            if gender == "Female":
                st.image("https://cdn-icons-png.flaticon.com/512/2922/2922561.png", width=150, caption="Predicted: Female")
            else:
                st.image("https://cdn-icons-png.flaticon.com/512/2922/2922510.png", width=150, caption="Predicted: Male")

            # Show input details
            with st.expander("🔍 View Your Input Data"):
                st.dataframe(pd.DataFrame(input_data, index=["Input"]).T)

                
        except Exception as e:
            st.error(f"Prediction failed: {e}")

# ------------ About Me ------------------
elif selection == "👤 About Me":
    st.title("👤 About Me")
    st.markdown("""
    Hello! My name is **Sachin**, and I'm passionate about technology, data science, and AI-powered solutions.  
I love diving into machine learning projects, building intelligent models, and staying updated with the latest advancements in artificial intelligence.

### 🌐 Real-World Applications:
- This approach can be used in **call center analytics**, **virtual assistants**, **voice biometrics**, and **accessibility tools**.
- Can also support **emotion detection** or **speaker verification** by extending the current setup.

---

### 🔧 Future Improvements:
- Integrate **deep learning models** like CNNs or LSTMs for even better audio classification.  
- Add real-time **voice recording and feature extraction** within the app.  
- Track model performance over time using tools like **MLflow** or **Weights & Biases**.  
- Extend the model to classify **more voice attributes**, such as age group or emotional tone.

---

Overall, this project has been a great step forward in learning how to apply machine learning to real-world voice data, and I'm excited to keep improving it!
    """)
    
    # Optional: Add more details
    st.subheader("📌 Interests & Expertise")
    st.markdown("""
    - 🤖 Machine Learning  
    - 📊 Data Analytics  
    - 🎯 Predictive Modeling  
    - 🏆 Continuous Learning  
    """)
    
    st.image("https://cdn-icons-png.flaticon.com/512/3135/3135715.png", width=150, caption="Sachin's Profile")
    
    # Expand for additional details
    with st.expander("🔍 More About Me"):
        st.markdown("""
        - 🔥 Enthusiastic about deep learning  
        - 💡 Love solving data-driven challenges  
        - 🎯 Always eager to learn new things  
        """)


           
