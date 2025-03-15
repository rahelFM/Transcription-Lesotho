## Speech-to-Text Benchmarking on Code-Switched isiZulu-English Dataset
This repository contains a notebook demonstrating the benchmarking of two speech-to-text models: Whisper Small and Wav2Vec2, trained on a code-switched isiZulu-English audio dataset. 
The notebook includes steps followed to do exploratory data analysis, error analysis, model performance evaluation or explorative model analysis with Character Error Rate and Word Error Rate, and incorporates visualization, explainability tools, and human-in-the-loop techniques as we perfomed some of the error analysis by using human intelligence. 
## Table of Contents
1. [Overview](#overview)
2. [Setup](#setup)
3. [How to Run the Notebook](#how-to-run-the-notebook)
4. [Reproducing Results](#reproducing-results)
5. [Model Evaluation & Error Analysis](#model-evaluation--error-analysis)
6. [Insights and Suggestions for Improving Performance](#insights-and-suggestions-for-improving-performance)
7. [License](#license)

## Overview

In this project, we benchmark two popular speech-to-text models, Whisper Small and Wav2Vec2, on a code-switched dataset of isiZulu-English spoken language. We analyze the transcription results and compare model performance using Character Error Rate (CER) and Word Error Rate (WER). In addition, we perform an error analysis on the Whisper Small model, as it demonstrated better performance.
### Steps Involved:
- Exploratory Data Analysis (EDA) on the dataset
- Error Metrics: Computation of CER and WER
- Model Evaluation: Training and evaluation of Whisper Small and Wav2Vec2
- Error Analysis: Focused error analysis on Whisper Small model performance
- Visualizations: Display of model performance and error distributions
- Explainability: Use of explainability tools for model insights
- Human-in-the-loop: Suggestions for further performance improvements

## Setup

### Prerequisites
Before running the notebook, ensure the following libraries are installed in your environment:

- Python >= 3.7
- TensorFlow
- Transformers (for Wav2Vec2 and Whisper)
- matplotlib, seaborn (for visualizations)
- pandas, numpy (for data manipulation)
- scikit-learn (for computing CER and WER)
- torch (for Wav2Vec2)
- librosa (for audio data processing)

  
### Install Dependencies:
!pip install tensorflow transformers torch librosa matplotlib seaborn pandas scikit-learn


### Dataset
This project uses a code-switched isiZulu-English speech dataset. You can upload the dataset in the "/content/dataset_v2cleaned.zip" directory. There is two csv transcription files and 97 corresponding audio files. The first "transcriptions.csv" file holds original ground truth transcription which has columns of idx,user_ids,accent,country,transcript,nchars,audio_ids,audio_paths,duration,origin,domain,gender. We tried to similarize with the expected trnscription csv files generated with the benchmark models from the audio files which has "filename" and "transcription". Depending on that the second transcription dataset is developed "transcriptionscleaned.csv". transcriptionscleaned.csv dataset is used as a ground truth data when training the models and evaluate. 

### Step 1: Clone the 
If you want to clone the repository and get the notebook along with required resources:
git clone https://github.com/your-username/speech-to-text-benchmarking.git
cd speech-to-text-benchmarking

### Step 2: Import Necessary Libraries
Start by importing the necessary libraries and preparing the dataset.
import pandas as pd
import numpy as np
import librosa
import matplotlib.pyplot as plt
from transformers import Wav2Vec2ForCTC, Wav2Vec2Processor
from openai_whisper import WhisperModel
from sklearn.metrics import accuracy_score
### Step 3: Load and Preprocess the Dataset

In this step, load your audio dataset and corresponding transcriptions. The notebook performs basic pre-processing like:

- Audio loading and normalization
- Tokenization of ground truth and predicted transcriptions
# Load your dataset (replace with your path)
dataset = pd.read_csv('data/dataset.csv')

# Preprocess the audio and transcriptions

### Step 4: Transcribe Audio with Whisper Small and Wav2Vec2
- Whisper Small Model:
  
whisper_model = WhisperModel.from_pretrained('whisper-small')
whisper_preds = whisper_model.transcribe(audio_files)

- Wav2Vec2 Model:
processor = Wav2Vec2Processor.from_pretrained("facebook/wav2vec2-large-960h")
model = Wav2Vec2ForCTC.from_pretrained("facebook/wav2vec2-large-960h")
inputs = processor(audio_files, return_tensors="pt", padding=True)
logits = model(input_values).logits
### Step 5: Evaluate Performance Using CER and WER
from jiwer import wer, cer

# Evaluate Whisper Small Model
whisper_cer = cer(ground_truth_transcriptions, whisper_preds)
whisper_wer = wer(ground_truth_transcriptions, whisper_preds)

# Evaluate Wav2Vec2 Model
wav2vec_cer = cer(ground_truth_transcriptions, wav2vec_preds)
wav2vec_wer = wer(ground_truth_transcriptions, wav2vec_preds)

### Step 6: Visualize the Results
plt.bar(['Whisper Small', 'Wav2Vec2'], [whisper_wer, wav2vec_wer])
plt.ylabel('WER')
plt.title('Model Comparison - WER')
plt.show()

-Since the ## Whisper small model is better in performance than the Wav2Vec2 model to trancribe, on the next step we focused on the better model model for error analysis.
### Step 7: Conduct Error Analysis

Perform error analysis on the Whisper Small model to gain insights into where it makes the most errors. The notebook provides detailed visualizations for different error types such as substitution, insertion, and deletion.
## Reproducing Results

To reproduce the results from this notebook:

1. Clone the repository.
2. Ensure your environment has the necessary dependencies installed.
3. Upload the dataset and ensure it is correctly formatted.
4. Run the notebook cells in sequence to load the dataset, process the audio, evaluate the models, and visualize results.
## Model Evaluation & Error Analysis

### Whisper Small vs Wav2Vec2:

- Whisper Small showed slightly better performance in WER and CER compared to Wav2Vec2, especially on the Zulu-English code-switched dataset.
- Key error types for Whisper Small:
  - High substitution errors for words with heavy code-switching.
  - Moderate deletion and insertion errors.

### Error Analysis:
The error analysis revealed that the model struggles with certain phonetic patterns and sentence structures when switching between Zulu and English in the same sentence. Additionally, noise in the recordings and accent variability posed challenges for both models.
## Insights and Suggestions for Improving Performance

1. Data Augmentation: Increase the amount of training data with code-switched samples, including more diverse accent variations.
2. Fine-tuning on Code-Switched Data: Fine-tune the models specifically on code-switched Zulu-English datasets.
3. Speech Enhancement Techniques: Use noise reduction methods to clean the audio input for better transcriptions.
4. Use of Hybrid Models: Combine Whisper and Wav2Vec2 for a hybrid model that benefits from the strengths of both.
5. Hyperparameter Tuning: Experiment with hyperparameters, especially for Wav2Vec2, to improve performance on non-English languages.
