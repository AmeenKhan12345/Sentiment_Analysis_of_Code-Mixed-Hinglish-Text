import streamlit as st
import torch
import torch.nn.functional as F
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import plotly.express as px
import datetime
import time
import os
from transformers import AutoTokenizer, AutoModelForSequenceClassification
from lime.lime_text import LimeTextExplainer
from wordcloud import WordCloud

# Page configuration
st.set_page_config(
    page_title="Sentiment Analysis Dashboard",
    page_icon="😀",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Enhanced CSS for a more professional look
st.markdown("""
<style>
    /* Main Layout and Colors */
    .main {
        background-color: #f8f9fa;
        padding: 1rem;
    }
    
    /* Header Styling */
    h1, h2, h3 {
        color: #1E3A8A;
        font-family: 'Segoe UI', Arial, sans-serif;
    }
    h1 {
        font-weight: 600;
        border-bottom: 2px solid #4e73df;
        padding-bottom: 0.5rem;
        margin-bottom: 1rem;
    }
    h2 {
        font-weight: 500;
        margin-top: 1.5rem;
    }
    h3 {
        font-weight: 500;
        color: #2c3e50;
        margin-top: 1rem;
    }
    
    /* Button Styling */
    .stButton button {
        background-color: #4e73df;
        color: white;
        border-radius: 6px;
        border: none;
        padding: 0.5rem 1.2rem;
        transition: all 0.3s;
        font-weight: 500;
        box-shadow: 0 2px 5px rgba(0,0,0,0.1);
    }
    .stButton button:hover {
        background-color: #2e59d9;
        box-shadow: 0 4px 8px rgba(0,0,0,0.2);
        transform: translateY(-1px);
    }
    .stButton button:active {
        transform: translateY(0px);
    }
    
    /* Tab Styling */
    .stTabs [data-baseweb="tab-list"] {
        gap: 2px;
        border-bottom: 1px solid #e0e0e0;
    }
    .stTabs [data-baseweb="tab"] {
        background-color: #f1f3f9;
        border-radius: 6px 6px 0px 0px;
        padding: 12px 24px;
        border: none;
        font-weight: 500;
    }
    .stTabs [aria-selected="true"] {
        background-color: #4e73df !important;
        color: white !important;
    }
    
    /* Metric Styling */
    div[data-testid="stMetricValue"] {
        font-size: 2.2rem;
        font-weight: 600;
        color: #2c3e50;
    }
    div[data-testid="stMetricLabel"] {
        font-size: 1rem;
        font-weight: 500;
    }
    [data-testid="stMetricDelta"] > div {
        font-size: 1rem;
    }
    
    /* Container and Card Styling */
    div.stDataFrame {
        border: 1px solid #e0e0e0;
        border-radius: 8px;
        padding: 1px;
        box-shadow: 0 2px 4px rgba(0,0,0,0.05);
    }
    
    /* Sidebar improvements */
    section[data-testid="stSidebar"] {
        background-color: #f1f5fd;
        border-right: 1px solid #e0e0e0;
    }
    section[data-testid="stSidebar"] h1 {
        color: #1E3A8A;
        font-size: 1.8rem;
        padding-bottom: 0.5rem;
    }
    section[data-testid="stSidebar"] h2 {
        color: #2c3e50;
        font-size: 1.3rem;
        margin-top: 2rem;
    }
    
    /* Input Fields */
    div[data-testid="stTextInput"] label,
    div[data-testid="stTextArea"] label,
    div[data-testid="stSelectbox"] label {
        font-weight: 500;
        color: #2c3e50;
    }
    div[data-testid="stTextInput"] input,
    div[data-testid="stTextArea"] textarea,
    div[data-testid="stSelectbox"] div[data-baseweb="select"] {
        border-radius: 6px;
        border: 1px solid #cfd7e6;
    }
    
    /* Progress Bar */
    div[data-testid="stProgressBar"] {
        height: 10px;
        border-radius: 10px;
    }
    
    /* Custom Cards */
    .dashboard-card {
        background-color: white;
        border-radius: 8px;
        padding: 1.5rem;
        box-shadow: 0 4px 6px rgba(0, 0, 0, 0.05);
        margin-bottom: 1rem;
        border-left: 4px solid #4e73df;
    }
    .sentiment-negative {
        border-left: 4px solid #ff6b6b;
    }
    .sentiment-neutral {
        border-left: 4px solid #51cf66;
    }
    .sentiment-positive {
        border-left: 4px solid #339af0;
    }
    
    /* Download Button */
    .stDownloadButton button {
        background-color: #28a745;
        color: white;
        border-radius: 6px;
        border: none;
        padding: 0.5rem 1.2rem;
        font-weight: 500;
    }
    .stDownloadButton button:hover {
        background-color: #218838;
        box-shadow: 0 4px 8px rgba(0,0,0,0.1);
    }
    
    /* Footer */
    footer {
        border-top: 1px solid #e0e0e0;
        padding-top: 1rem;
        color: #6c757d;
        font-size: 0.9rem;
        text-align: center;
    }
    
    /* Animations for live data */
    @keyframes pulse {
        0% { opacity: 1; }
        50% { opacity: 0.7; }
        100% { opacity: 1; }
    }
    .live-indicator {
        animation: pulse 1.5s infinite;
        display: inline-block;
        width: 10px;
        height: 10px;
        background-color: #ff0000;
        border-radius: 50%;
        margin-right: 8px;
    }
    
    /* Custom chart container */
    .chart-container {
        background-color: white;
        border-radius: 8px;
        padding: 1rem;
        box-shadow: 0 2px 6px rgba(0,0,0,0.05);
        margin-bottom: 1.5rem;
    }
    
    /* Enhanced text area */
    textarea {
        border: 1px solid #ddd !important;
        border-radius: 8px !important;
        padding: 10px !important;
        font-family: 'Segoe UI', Arial, sans-serif !important;
        font-size: 16px !important;
        transition: border 0.3s !important;
    }
    textarea:focus {
        border: 1px solid #4e73df !important;
        box-shadow: 0 0 0 2px rgba(78, 115, 223, 0.1) !important;
    }
    
    /* Data table styling */
    .dataframe {
        font-family: 'Segoe UI', Arial, sans-serif !important;
    }
    .dataframe thead th {
        background-color: #f1f5fd !important;
        color: #1E3A8A !important;
        font-weight: 600 !important;
    }
    
    /* Card with hover effect */
    .hover-card {
        transition: transform 0.3s, box-shadow 0.3s;
    }
    .hover-card:hover {
        transform: translateY(-5px);
        box-shadow: 0 10px 20px rgba(0, 0, 0, 0.1);
    }
</style>
""", unsafe_allow_html=True)

# Function to load model and tokenizer - used by both tabs
@st.cache_resource
def load_model_and_tokenizer(model_path):
    """Load and cache the model and tokenizer to avoid reloading"""
    tokenizer = AutoTokenizer.from_pretrained(model_path)
    model = AutoModelForSequenceClassification.from_pretrained(model_path)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)
    model.eval()
    return tokenizer, model, device

# Set your saved model path (update as needed)
MODEL_PATH = "C:\\Users\\ASUS\\Downloads\\PBL2\\New\\fine_tuned_xlm_roberta99"

# Load model, tokenizer, and device
tokenizer, model, device = load_model_and_tokenizer(MODEL_PATH)

# Define sentiment classes
sentiment_mapping = {0: "Negative", 1: "Neutral", 2: "Positive"}
class_names = list(sentiment_mapping.values())

# Define prediction function for individual text samples
def predict_sentiment(text):
    """Predict sentiment for a single text input"""
    inputs = tokenizer(text, return_tensors="pt", padding=True, truncation=True, max_length=128)
    inputs = {k: v.to(device) for k, v in inputs.items()}
    with torch.no_grad():
        outputs = model(**inputs)
    probs = F.softmax(outputs.logits, dim=1).cpu().numpy()[0]
    predicted_class = int(np.argmax(probs))
    return sentiment_mapping[predicted_class], probs

# Define prediction function for LIME that returns class probabilities
def predict_proba(texts):
    """Batch prediction function compatible with LIME"""
    # Ensure texts is a list of strings
    if isinstance(texts, str):
        texts = [texts]
    # If pre-tokenized (list of lists), join tokens back into a string
    if len(texts) > 0 and isinstance(texts[0], list):
        texts = [" ".join(token_list) for token_list in texts]
    inputs = tokenizer(texts, return_tensors="pt", padding=True, truncation=True, max_length=128)
    # Move inputs to the same device as the model
    inputs = {k: v.to(device) for k, v in inputs.items()}
    with torch.no_grad():
        outputs = model(**inputs)
    probs = F.softmax(outputs.logits, dim=1)
    return probs.cpu().numpy()

# Function to load and prepare tweet data
@st.cache_data
def load_tweet_data(file_path):
    """Load and prepare tweet data with cache to avoid reloading"""
    try:
        df = pd.read_csv(file_path, encoding="latin-1")
        if "created_at" not in df.columns:
            start_time = pd.to_datetime("2023-01-01 00:00:00")
            df["created_at"] = start_time + pd.to_timedelta(np.arange(len(df)), unit="m")
        df["text"] = df["text"].astype(str).fillna("").str.strip()
        df = df[df["text"] != ""]
        return df
    except Exception as e:
        st.error(f"Error loading tweet data: {str(e)}")
        # Return empty dataframe with expected columns
        return pd.DataFrame(columns=["text", "created_at"])

# Function to fetch tweets from dataframe
def fetch_tweets_from_df(query, df, max_tweets=20):
    """Filter dataframe for tweets containing the query"""
    df_filtered = df[df["text"].str.contains(query, case=False, na=False)].copy()
    df_filtered.sort_values("created_at", inplace=True)
    return df_filtered.head(max_tweets)

# Function to generate word cloud
def generate_wordcloud(df, query, sentiment_filter=None):
    """Generate word cloud from filtered tweets"""
    try:
        # Filter the dataset by the user query (case-insensitive)
        df_filtered = df[df["text"].str.contains(query, case=False, na=False)].copy()
        
        if len(df_filtered) == 0:
            st.warning(f"No tweets found containing '{query}'. Try a different keyword.")
            return None
        
        # If sentiment filter is requested, analyze and filter by sentiment
        if sentiment_filter and sentiment_filter != "All":
            # Add sentiment column if it doesn't exist
            if "Sentiment" not in df_filtered.columns:
                st.info(f"Analyzing sentiment for {len(df_filtered)} tweets...")
                df_filtered["Sentiment"] = df_filtered["text"].apply(
                    lambda x: predict_sentiment(x)[0]
                )
            
            # Apply sentiment filter
            df_filtered = df_filtered[df_filtered["Sentiment"] == sentiment_filter]
            
            if len(df_filtered) == 0:
                st.warning(f"No {sentiment_filter} tweets found containing '{query}'. Try a different filter or keyword.")
                return None
                
            title = f"Word Cloud for {sentiment_filter} Tweets containing '{query}'"
        else:
            title = f"Word Cloud for Tweets containing '{query}'"
        
        # Concatenate tweet texts
        text_data = " ".join(df_filtered["text"].tolist())
        
        if not text_data or text_data.strip() == "":
            st.warning("No text data available for the selected filter and query.")
            return None
        
        # Print for debugging
        st.text(f"Processing {len(df_filtered)} tweets with {len(text_data)} characters")
        
        # Generate word cloud with error handling
        try:
            # Enhanced wordcloud with better design
            wordcloud = WordCloud(
                width=800, 
                height=400, 
                background_color="white", 
                max_words=200, 
                contour_width=3, 
                contour_color='steelblue',
                colormap='viridis',
                collocations=False,
                min_font_size=10,
                max_font_size=80,
                random_state=42
            ).generate(text_data)
            
            fig, ax = plt.subplots(figsize=(12, 6))
            ax.imshow(wordcloud, interpolation="bilinear")
            ax.axis("off")
            ax.set_title(title, fontsize=16, pad=20)
            fig.tight_layout(pad=3)
            
            return fig
        except Exception as e:
            st.error(f"Error generating word cloud: {str(e)}")
            return None
            
    except Exception as e:
        st.error(f"Error in word cloud generation: {str(e)}")
        return None

# Function to save feedback to CSV
def save_feedback(input_text, predicted_sentiment, corrected_sentiment, timestamp):
    """Save user feedback to CSV file"""
    feedback_file = "feedback.csv"
    
    # Create feedback DataFrame
    feedback_data = {
        "text": [input_text],
        "predicted_sentiment": [predicted_sentiment],
        "corrected_sentiment": [corrected_sentiment],
        "timestamp": [timestamp]
    }
    feedback_df = pd.DataFrame(feedback_data)
    
    # Check if feedback file exists
    if os.path.exists(feedback_file):
        # Append to existing file
        feedback_df.to_csv(feedback_file, mode='a', header=False, index=False)
    else:
        # Create new file with header
        feedback_df.to_csv(feedback_file, index=False)
    
    return True

# Create LIME explainer
explainer = LimeTextExplainer(class_names=class_names)

# Load tweet data
tweet_data_path = "C:\\Users\\ASUS\\Downloads\\PBL2\\New\\emotions_tweets.csv"
tweet_df = load_tweet_data(tweet_data_path)

# Sidebar for global settings and filters
with st.sidebar:
    st.title("Dashboard Settings")
    st.markdown("---")
    
    # Word Cloud Options
    st.header("Word Cloud Options")
    query_wc = st.text_input("Enter keyword for Word Cloud", "Hinglish")
    sentiment_option = st.selectbox("Select Sentiment for Word Cloud", 
                                  options=["All", "Negative", "Neutral", "Positive"])
    
    show_wc_button = st.button("Show Word Cloud")

# Main content area with tabs
tab1, tab2, tab3 = st.tabs(["🔍 Single Text Analysis", "📊 Tweet Dashboard", "📝 Feedback Summary"])

with tab1:
    st.title("Sentiment Analysis with LIME Explanations")
    st.write("Enter text for sentiment prediction and see a LIME explanation below.")
    
    # Text input and analysis
    user_input = st.text_area("Input Text", "Ye product bahot bekar hai", height=100)
    
    col1, col2 = st.columns([1, 3])
    with col1:
        predict_button = st.button("Predict and Explain")
    
    if predict_button:
        # Store input and prediction for potential feedback
        st.session_state.current_input = user_input
        with st.spinner("Analyzing sentiment..."):
            # Run model prediction
            predicted_sentiment, probs = predict_sentiment(user_input)
            st.session_state.current_prediction = predicted_sentiment
            
            # Display results in columns
            res_col1, res_col2, res_col3 = st.columns(3)
            with res_col1:
                st.metric("Predicted Sentiment", predicted_sentiment)
            with res_col2:
                st.metric("Negative Probability", f"{probs[0]:.2%}")
            with res_col3:
                st.metric("Positive Probability", f"{probs[2]:.2%}")
            
            # Show detailed probabilities
            st.write("### Detailed Class Probabilities:")
            prob_df = pd.DataFrame({
                "Sentiment": class_names,
                "Probability": probs
            })
            fig = px.bar(prob_df, x="Sentiment", y="Probability", color="Sentiment",
                         color_discrete_map={"Negative": "#ff6b6b", "Neutral": "#51cf66", "Positive": "#339af0"})
            st.plotly_chart(fig)
            
            # Generate LIME explanation
            with st.spinner("Generating explanation..."):
                exp = explainer.explain_instance(user_input, predict_proba, num_features=10, num_samples=500)
                lime_html = exp.as_html()
                st.write("### Interactive LIME Explanation:")
                st.components.v1.html(lime_html, height=500, scrolling=True)
                
                st.write("### Feature Contribution Bar Chart:")
                word_weights = exp.as_list()
                if word_weights:
                    words, weights = zip(*word_weights)
                    fig, ax = plt.subplots(figsize=(10, 6))
                    colors = ["red" if w < 0 else "blue" for w in weights]
                    ax.barh(words, weights, color=colors)
                    ax.set_xlabel("Contribution Weight")
                    ax.set_title("LIME Feature Contributions")
                    st.pyplot(fig)
                else:
                    st.write("No feature contributions available.")
        
        # Set a flag indicating a prediction has been made
        st.session_state.prediction_done = True
    
    # Render the feedback UI if a prediction was made
    if st.session_state.get("prediction_done", False):
        st.markdown("---")
        st.write("### Feedback")
        st.write("Please let us know if the prediction was correct:")
        
        # Initialize default feedback values if not already set
        if 'is_correct' not in st.session_state:
            st.session_state.is_correct = "Yes"
        if 'corrected_sentiment' not in st.session_state:
            st.session_state.corrected_sentiment = st.session_state.current_prediction
        
        feedback_col1, feedback_col2 = st.columns([1, 2])
        with feedback_col1:
            is_correct = st.radio("Was the prediction correct?",
                                  ["Yes", "No"],
                                  horizontal=True,
                                  key="feedback_radio")
            st.session_state.is_correct = is_correct
        if is_correct == "No":
            with feedback_col2:
                corrected_sentiment = st.selectbox(
                    "Select the correct sentiment:",
                    ["Negative", "Neutral", "Positive"],
                    index=class_names.index(st.session_state.current_prediction),
                    key="corrected_sentiment_select"
                )
                st.session_state.corrected_sentiment = corrected_sentiment
        else:
            st.session_state.corrected_sentiment = st.session_state.current_prediction
    
        feedback_submit = st.button("Submit Feedback")
        if feedback_submit:
            try:
                input_text = st.session_state.current_input
                model_prediction = st.session_state.current_prediction
                user_correction = st.session_state.corrected_sentiment
                timestamp = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
                
                feedback_data = {
                    "text": [input_text],
                    "predicted_sentiment": [model_prediction],
                    "corrected_sentiment": [user_correction],
                    "timestamp": [timestamp]
                }
                feedback_df = pd.DataFrame(feedback_data)
                feedback_file = "feedback.csv"
                if os.path.exists(feedback_file):
                    feedback_df.to_csv(feedback_file, mode='a', header=False, index=False)
                else:
                    feedback_df.to_csv(feedback_file, index=False)
                
                st.success("Thank you for your feedback! It helps us improve our model.")
            except Exception as e:
                st.error(f"Failed to save feedback: {str(e)}")


with tab2:
    st.title("Real-Time Sentiment Analysis Dashboard")
    st.write("This dashboard simulates live sentiment analysis using Hinglish tweets.")
    
    # Display word cloud in the main dashboard if button is clicked
    wordcloud_container = st.container()
    with wordcloud_container:
        if show_wc_button:
            with st.spinner("Generating word cloud..."):
                # Add extra debug information
                st.info(f"Searching for tweets containing '{query_wc}'...")
                            # Call the fixed word cloud function
                wordcloud_fig = generate_wordcloud(tweet_df, query_wc, sentiment_filter=sentiment_option)
            
                if wordcloud_fig:
                    st.pyplot(wordcloud_fig)
                else:
                    st.error("Could not generate word cloud. Please try a different keyword or sentiment filter.")

# Add a separator
st.markdown("---")

# Tweet search and live analysis
st.subheader("Live Tweet Analysis")
col1, col2 = st.columns([3, 1])
with col1:
    query = st.text_input("Enter a keyword or hashtag (e.g., War)", "War")
with col2:
    refresh_interval = st.number_input("Refresh interval (seconds)", min_value=1, value=5, max_value=30, step=1)

if st.button("Start Live Analysis"):
    st.write("Starting live analysis simulation...")
    sentiment_data = []  # List to accumulate tweet sentiment data
    
    progress_bar = st.progress(0)
    status_text = st.empty()
    chart_placeholder = st.empty()
    tweet_placeholder = st.empty()
    
    # Get tweets containing the query
    df_tweets = fetch_tweets_from_df(query, tweet_df, max_tweets=20)
    total_tweets = len(df_tweets)
    
    if total_tweets == 0:
        st.warning(f"No tweets found containing '{query}'. Try a different keyword.")
    else:
        # Process tweets one by one with simulated delay
        for i, (idx, row) in enumerate(df_tweets.iterrows()):
            # Update progress
            progress = int((i + 1) / total_tweets * 100)
            progress_bar.progress(progress)
            status_text.text(f"Processing tweet {i+1}/{total_tweets}")
            
            # Predict sentiment
            sentiment, probs = predict_sentiment(row["text"])
            
            # Add to results
            sentiment_data.append({
                "text": row["text"],
                "created_at": row["created_at"],
                "Sentiment": sentiment,
                "Negative": float(probs[0]),
                "Neutral": float(probs[1]),
                "Positive": float(probs[2])
            })
            
            # Show the latest tweet and its sentiment
            tweet_placeholder.markdown(f"""
            **Latest Tweet:** "{row['text']}"  
            **Sentiment:** {sentiment} (Confidence: {max(probs):.2%})
            """)
            
            # Update chart every few tweets or at the end
            if (i + 1) % 5 == 0 or i == total_tweets - 1:
                df_sentiments = pd.DataFrame(sentiment_data)
                df_sentiments["created_at"] = pd.to_datetime(df_sentiments["created_at"])
                df_sentiments["minute"] = df_sentiments["created_at"].dt.floor("T")
                
                # Group by minute and sentiment
                timeline = df_sentiments.groupby(["minute", "Sentiment"]).size().unstack(fill_value=0).reset_index()
                
                # Create a grouped bar chart
                fig = px.bar(
                    timeline, 
                    x="minute", 
                    y=timeline.columns[1:], 
                    title="Sentiment Timeline",
                    labels={"value": "Count", "minute": "Time"},
                    barmode="group",
                    color_discrete_map={"Negative": "#ff6b6b", "Neutral": "#51cf66", "Positive": "#339af0"}
                )
                
                chart_placeholder.plotly_chart(fig, use_container_width=True)
            
            # Simulate processing delay
            time.sleep(refresh_interval / total_tweets)
        
        # Show final results
        progress_bar.progress(100)
        status_text.text("Analysis complete! All tweets processed.")
        
        # Display summary statistics
        st.write("### Sentiment Summary")
        sentiment_counts = pd.DataFrame(sentiment_data)["Sentiment"].value_counts().reset_index()
        sentiment_counts.columns = ["Sentiment", "Count"]
        
        fig = px.pie(
            sentiment_counts, 
            values="Count", 
            names="Sentiment",
            title="Sentiment Distribution",
            color="Sentiment",
            color_discrete_map={"Negative": "#ff6b6b", "Neutral": "#51cf66", "Positive": "#339af0"}
        )
        st.plotly_chart(fig, use_container_width=True)
        
        # Display the analyzed tweets in a table
        st.write("### Analyzed Tweets")
        tweet_table = pd.DataFrame(sentiment_data)[["text", "Sentiment", "created_at"]]
        tweet_table.columns = ["Tweet", "Sentiment", "Timestamp"]
        st.dataframe(tweet_table, use_container_width=True)

#Add new Feedback Summary tab
with tab3:
    st.title("Feedback Summary")
    st.write("View and analyze user feedback to improve model performance.")
    # Load feedback data if available
    feedback_file = "feedback.csv"
    if os.path.exists(feedback_file):
        feedback_df = pd.read_csv(feedback_file)

        # Display feedback metrics
        st.write("### Feedback Statistics")
        col1, col2, col3 = st.columns(3)

        with col1:
            total_feedback = len(feedback_df)
            st.metric("Total Feedback", total_feedback)

        with col2:
            correct_predictions = (feedback_df["predicted_sentiment"] == feedback_df["corrected_sentiment"]).sum()
            accuracy = correct_predictions / total_feedback * 100 if total_feedback > 0 else 0
            st.metric("Model Accuracy", f"{accuracy:.1f}%")

        with col3:
            misclassifications = total_feedback - correct_predictions
            st.metric("Misclassifications", misclassifications)

        # Show feedback distribution
        st.write("### Feedback Distribution")

        # Create a confusion matrix
        predicted_sentiments = feedback_df["predicted_sentiment"].value_counts().reset_index()
        predicted_sentiments.columns = ["Sentiment", "Count"]

        corrected_sentiments = feedback_df["corrected_sentiment"].value_counts().reset_index()
        corrected_sentiments.columns = ["Sentiment", "Count"]

        # Show side-by-side charts
        col1, col2 = st.columns(2)

        with col1:
            fig1 = px.pie(
                predicted_sentiments, 
                values="Count", 
                names="Sentiment",
                title="Model Predictions",
                color="Sentiment",
                color_discrete_map={"Negative": "#ff6b6b", "Neutral": "#51cf66", "Positive": "#339af0"}
            )
            st.plotly_chart(fig1)

        with col2:
            fig2 = px.pie(
                corrected_sentiments, 
                values="Count", 
                names="Sentiment",
                title="User Corrections",
                color="Sentiment",
                color_discrete_map={"Negative": "#ff6b6b", "Neutral": "#51cf66", "Positive": "#339af0"}
            )
            st.plotly_chart(fig2)

        # Create a confusion matrix
        if len(feedback_df) > 0:
            st.write("### Confusion Matrix")

            # Create a confusion matrix dataframe
            confusion_data = pd.crosstab(
                feedback_df["predicted_sentiment"], 
                feedback_df["corrected_sentiment"],
                rownames=["Predicted"],
                colnames=["Actual"]
            )

            # Display as a heatmap
            fig = px.imshow(
                confusion_data,
                labels=dict(x="Actual Sentiment", y="Predicted Sentiment", color="Count"),
                x=confusion_data.columns,
                y=confusion_data.index,
                color_continuous_scale="Blues",
                title="Confusion Matrix"
            )
            fig.update_layout(height=400)
            st.plotly_chart(fig, use_container_width=True)

        # Show feedback data table
        st.write("### Feedback Details")
        st.dataframe(feedback_df, use_container_width=True)

        # Add download button for feedback data
        csv = feedback_df.to_csv(index=False)
        st.download_button(
            label="Download Feedback Data",
            data=csv,
            file_name="sentiment_feedback.csv",
            mime="text/csv",
        )
    else:
        st.info("No feedback data available yet. Use the Single Text Analysis tab to provide feedback.")

# Footer
st.markdown("---")
st.markdown(
    """
    <div style="text-align: center; color: #666;">
        <p>Sentiment Analysis Dashboard • Last updated: March 7, 2025</p>
    </div>
    """, 
    unsafe_allow_html=True
)