# Sentiment Analysis Dashboard for Hinglish

This project is a comprehensive Sentiment Analysis Dashboard designed for code-mixed Hinglish (Hindi-English) text. It leverages a fine-tuned XLM-RoBERTa model to classify text into three sentiment categories: Negative, Neutral, and Positive. The interactive dashboard, built using Streamlit, allows users to analyze single texts with explainability (via LIME), view simulated live tweet sentiment trends, and provide feedback to continuously improve the model.
 - Have a look of this project -> [Hinglish-Sentiment-Analysis](https://sentimentanalysisofhinglishtext.streamlit.app/)
## Table of Contents

- [Overview](#overview)
- [Features](#features)
- [Methodology](#methodology)
- [Installation](#installation)
- [Usage](#usage)
- [Requirements](#requirements)
- [Future Enhancements](#future-enhancements)
- [License](#license)
- [Acknowledgements](#acknowledgements)

## Overview

This project focuses on sentiment analysis for Hinglish tweets and social media text. By leveraging the cross-lingual capabilities of XLM-RoBERTa, the model is fine-tuned to handle the unique challenges of code-mixed text. The dashboard provides an interactive experience with multiple components, including single text analysis with LIME explanations, a simulated live sentiment dashboard, and a feedback mechanism for continuous model improvement.

## Features

- **Sentiment Classification:**  
  Classifies Hinglish text into Negative, Neutral, and Positive sentiments using a fine-tuned XLM-RoBERTa model.
  
- **Interactive Dashboard:**  
  Built with Streamlit, the dashboard includes:
  - **Single Text Analysis Tab:**  
    Input a piece of text and view its predicted sentiment along with LIME-based feature explanations.
  - **Tweet Dashboard Tab:**  
    Simulate live sentiment analysis using a pre-collected dataset of Hinglish tweets. Visualizations (line/bar charts) show sentiment trends over time.
  - **Feedback Summary Tab:**  
    View feedback statistics, confusion matrices, and download feedback data for further analysis.
    
- **Feedback Mechanism:**  
  After receiving a prediction, users can provide feedback if the prediction was incorrect. This feedback is stored in `feedback.csv` for future model improvement.

## Methodology

The project uses XLM-RoBERTa, a powerful cross-lingual transformer model, fine-tuned on a dataset of Hinglish tweets. The model is trained to understand the unique linguistic patterns of code-mixed Hinglish text, providing accurate sentiment classification while handling the challenges of mixed language syntax and semantics.

## Installation

### Prerequisites
- Python 3.10 or higher
- Git

### Clone the Repository
```bash
git clone https://github.com/AmeenKhan12345/Sentiment_Analysis_of_Code-Mixed-Hinglish-Text.git
```

### Install Dependencies
```bash
pip install -r requirements.txt
```

## Model Setup

The fine-tuned model used in this project is hosted on the Hugging Face Model Hub. It can be automatically downloaded when you run the app.

1. Create a Hugging Face account if you haven't already.
2. Clone this repository.
3. Install the required dependencies:
   ```bash
   pip install -r requirements.txt
4. The model is loaded using the repository ID:
   ```bash
   MODEL_ID = "AmeenKhan/Sentiment-Analysis-Code-Mixed-Hinglish-Text"
   tokenizer = AutoTokenizer.from_pretrained(MODEL_ID)
   model = AutoModelForSequenceClassification.from_pretrained(MODEL_ID)
For more details on hosting models, refer to the Hugging Face documentation.
By following these steps, you ensure your GitHub repository remains lightweight, and users can download the large model from the Hugging Face Model Hub when they run your project. Let me know if you need any further assistance!

## Usage

### Running the Dashboard
Launch the Streamlit app with:
```bash
streamlit run app.py
```

### Dashboard Features

#### Single Text Analysis Tab
- Enter Hinglish text in the input field
- View predicted sentiment (Negative, Neutral, or Positive)
- Explore LIME explanations showing which words influenced the prediction

#### Tweet Dashboard Tab
- View simulated live sentiment analysis of Hinglish tweets
- Analyze sentiment trends over time using interactive visualizations
- Filter data by date range or sentiment category

#### Feedback Summary Tab
- Review model performance statistics
- Examine confusion matrices for prediction accuracy
- Download collected feedback data for further analysis

### Feedback Mechanism
After receiving a prediction, users can:
1. Indicate if the prediction was correct or incorrect
2. If incorrect, select the proper sentiment category
3. Submit feedback, which is stored for future model improvements

## Requirements

```
streamlit==1.24.1
torch==2.0.1
numpy==1.24.2
pandas==1.5.3
matplotlib==3.7.1
plotly==5.13.1
transformers==4.31.0
lime==0.2.0.1
wordcloud==1.8.2.2
```

## Future Enhancements

- **Live Data Integration:**  
  Integrate live Twitter/social media feeds using free scraping methods or APIs.
  
- **Multitask Learning:**  
  Extend the model to jointly predict sentiment and emotion for richer analysis.
  
- **Predictive Analytics:**  
  Incorporate time-series forecasting to predict future sentiment trends.
  
- **Advanced Explainability:**  
  Experiment with additional methods like SHAP or Integrated Gradients for deeper model insights.
  
- **Scalability and Deployment:**  
  Deploy the dashboard on cloud platforms for public access and real-time monitoring.

## License

This project is licensed under the [MIT License](LICENSE).

## Acknowledgements

- Special thanks to contributors and researchers in this project, [Mr. Pawan Hete](https://github.com/PawanHete) and [Pradyumn Waghmare](https://github.com/Xtrmcoder)
- Thanks to the developers of [Transformers](https://github.com/huggingface/transformers), [Streamlit](https://streamlit.io/), and other open-source tools used in this project.

## Contributing

Contributions are welcome! Please feel free to submit a Pull Request.

## Contact

For questions or feedback, please open an issue in the GitHub repository.
