## Speech-to-Text Benchmarking on Code-Switched Zulu-English Dataset
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
### Step 1: Clone the Repository

If you want to clone the repository and get the notebook along with required resources:
git clone https://github.com/your-username/speech-to-text-benchmarking.git
cd speech-to-text-benchmarking


