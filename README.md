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

### Steps Involved:
- Exploratory Data Analysis (EDA) on the dataset
- Error Metrics: Computation of CER and WER
- Model Evaluation: Training and evaluation of Whisper Small and Wav2Vec2
- Error Analysis: Focused error analysis on Whisper Small model performance
- Explainability: Used substitution, insertion and deletion to have insight
- Human-in-the-loop: Suggestions for further performance improvements

## Setup

### Prerequisites
Before running the notebook, ensure the following libraries are installed in your environment:

| Library                           | Purpose & Use Case                                                                 |
|------------------------------------|------------------------------------------------------------------------------------|
| `os`                               | Handling file paths and directories, and interacting with the operating system.   |
| `librosa`                          | Audio analysis library for feature extraction, signal processing, and transformations.|
| `librosa.display`                  | For visualizing audio signals and spectrograms.                                   |
| `matplotlib.pyplot`                | Plotting and visualizing graphs and audio features.                               |
| `pandas`                           | Data manipulation and analysis, especially for structured data (DataFrames).      |
| `numpy`                            | Numerical computing and matrix operations.                                        |
| `seaborn`                          | Statistical data visualization built on `matplotlib`.                             |
| `ipywidgets`                       | Creating interactive widgets for Jupyter Notebooks.                               |
| `IPython.display`                  | Displaying rich media (e.g., audio, images) in Jupyter Notebooks.                 |
| `wordcloud`                        | Generating word clouds for text data visualization.                               |
| `string`                           | Provides constants and functions for string manipulation.                         |
| `soundfile`                        | Reading and writing sound files in different formats (e.g., WAV, FLAC).           |
| `csv`                              | Handling CSV file reading and writing.                                            |
| `torch`                            | Deep learning framework for model building, tensor operations, and GPU support.  |
| `transformers`                     | Pre-trained models for NLP, ASR, and more. Includes Wav2Vec2 for speech recognition.
|  `whisper`                         | Developed by OpenAI, and it is used for ASR to transcribe and translate spoken language into text across multiple languages.|

### Install Dependencies:
!pip install tensorflow transformers torch librosa matplotlib seaborn pandas scikit-learn
!pip install git+https://github.com/openai/whisper.git
!pip install -U openai-whisper
!pip install jiwer etc. 

### Dataset
This project uses a code-switched isiZulu-English speech dataset. You can upload the dataset in the "/content/dataset_v2cleaned.zip" directory. There is two csv transcription files and 97 corresponding audio files. The first "transcriptions.csv" file holds original ground truth transcription which has columns of idx,user_ids,accent,country,transcript,nchars,audio_ids,audio_paths,duration,origin,domain,gender. We tried to similarize with the expected trnscription csv files generated with the benchmark models from the audio files which has "filename" and "transcription". Depending on that the second transcription dataset is developed "transcriptionscleaned.csv". transcriptionscleaned.csv dataset is used as a ground truth data when training the models and evaluate. 

### Step 1: Clone the 
If you want to clone the repository and get the notebook along with required resources:
git clone https://github.com/rahelFM/Transcription-Lesotho.git
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
-Load your dataset 
- Audio loading and normalization
- Tokenization of ground truth and predicted transcriptions


### Step 4: Explorative data analysis
In this section we have tried to understand the datset's characterstics, including common trends, challenges, and potential biases.
There are 97 audio data samlples in the dataset and the maximum duration of the audio file is 6.34 seconds with average of 2.08 seconds.

We used an automated language identification tool to classify the language of each transcription. The model outputs language codes according to the ISO 639-1 standard (e.g., "en" for English, "zu" for Zulu). 
Detected Languages: The languages identified in the dataset included:
-English (en)
-isiZulu (zu)
-Xhosa (xh)
and other languages such as Portuguese, Spanish, etc.
# challenges faced in language identification
Misclassification: A few cases were found where isiZulu transcriptions were identified as other languages, like English or Portuguese.
Code-Switching: Many transcriptions contained a mix of languages, a phenomenon known as code-switching. The language model could have struggled with identifying these mixed-language transcriptions accurately.
# Results of language identification
Distribution of Languages: The language distribution revealed that a large portion of the dataset was identified as English, while isiZulu and Sesotho transcriptions were underrepresented.
Misclassification Example: Some isiZulu transcriptions were misclassified as English due to linguistic similarities between the two languages.

| Language  | Percentage |
|-----------|------------|
| English   | 51.5%      |
| Xhosa     | 9.3%       |
| isiZulu   | 1.0%       |
| Other     | 38.2%      |

The speakers accent is similar which is eng-sot all over the datset and their speech is clear with no background noise. There are 19 speakers in teh dataset, and CALEB is the most frequent speaker. The highest word count is 31 and highest character lenght is 153. There is a silent part at the begining and the end of the audio on seven audio data.  
In the initial version of the dataset, /content/lelapa_extracted/dataset_v2/dataset_v2cleaned/dataset_v2/transcriptions.csv, which the columns included several attributes such as idx, user_ids, accent, country, transcript, nchars, audio_ids, audio_paths, duration, origin, domain, gender   . While these details could be useful in certain contexts, they were not directly relevant to the transcription task we intended to and not correspondent with the attributes in the audio file. To streamline the dataset and make it more focused on the core objective, I simplified the structure to only include two key columns: Filename and Transcription. /content/lelapa_extracted/dataset_v2/dataset_v2cleaned/dataset_v2
Filename:This column contains the unique identifier for each audio file, ensuring that each transcription corresponds to the correct audio input.
Transcription: This column holds the textual output generated from the speech in the corresponding audio file. It reflects the model's transcription of the audio content.
By focusing on these two essential columns, the dataset is more straightforward to work with, particularly for training, evaluating, and comparing models such as Whisper-Small and Wav2Vec2. This simplification also enhances the clarity and relevance of the dataset for the transcription task, allowing for better evaluation of model performance based on transcription accuracy alone, without the distraction of unrelated metadata. 
### Step 5: Transcribe Audio with Whisper Small and Wav2Vec2
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

-Word Error Rate (WER)
WER measures the difference between the ground truth and the predicted transcription at the word level. A lower WER indicates better performance.
whisper has got 0.9542 average of WER, whereas, the wav2vec2 model has got 1.0129 average of WER. This suggests that the overall word-level accuracy of Whisper small model is better than the Wav2Vec2. 

-Character Error Rate (CER):
CER compares the ground truth and ASR output at the character level. It gives a more detailed analysis than WER by measuring character-level accuracy. On CER the whisper has got 0.6983 average and wav2vec2 has got 0.9062. The analysis of this result is that Whisper performs better in terms of character-level accuracy, meaning it transcribes individual characters more accurately than Wav2Vec2. A lower CER is preferred since it shows fewer errors at the character level. Since whisper has performed better than the Wav2Vec2, in both character level and word level, we chose whisper small for further analysis. 

### Step 6: Conduct Error Analysis

Next we performed error analysis on the Whisper Small model to gain insights into where it makes the most errors. The notebook provides detailed visualizations for different error types such as substitution, insertion, and deletion.
| Model         | CER (%) | WER (%) | Substitutions | Insertions | Deletions |
|--------------|--------|--------|--------------|------------|-----------|
| Whisper-Small | 69    | 95    | 10           | 0          | 2         |
| Wav2Vec2      | 90     | 100     | 6            | 0          | 6         |

## Model Evaluation & Error Analysis

### Whisper Small vs Wav2Vec2:

- Whisper Small showed slightly better performance in WER and CER compared to Wav2Vec2, especially on the Zulu-English code-switched dataset.
- Key error types for Whisper Small:
  - High substitution errors for words with heavy code-switching.
  - Moderate deletion and insertion errors.

### Error Analysis:
The error analysis revealed that the model struggles with certain phonetic patterns and sentence structures when switching between languages in the same sentence. Another issue we have seen on whisper small is that it tends to transcribe some of the audio into another character unlike the ground truth which is transcribed in Latin. The SHAP summary plot showing how individual words contribute to the model’s prediction of Word Error Rate in the Whisper small evaluation on the dataset is shown. The words listed in y-axis (e.g., "que," "so," "the," "nicht," etc.) are the features (individual words from transcriptions). These are the most influential words affecting WER. X-Axis (SHAP Value - Impact on Model Output), indicate how much a word increases or decreases the WER prediction. A positive SHAP value means that the word increases WER, leading to a higher error rate and a value closer to zero means little to no impact. The spread of dots shows variability in how that word impacts different samples.

- Words like "que," "nicht," "para," "gonna" have high SHAP values, meaning they significantly increase WER.
- Words like "the," "so," "on," "you" have low SHAP values, meaning their influence on WER is minimal.
- The model is showing words like "que" and "nicht" have higher impacts on WER, possibly indicating errors with multilingual speech recognition.
- 
## Insights and Suggestions for Improving Performance
1. Data: Woek on the data preprocessing more and increase the amount of training data with code-switched samples, including more diverse accent variations.
2. Fine-tuning: Fine-tune the models specifically on code-switched isiZulu-English datasets.
3. Speech Enhancement Techniques: Use noise reduction methods to clean the audio input for better transcriptions and remove the silent part.
4. Models: Using Whisper base, medium or large or even combining Whisper and Wav2Vec2 for a hybrid model that benefits from the strengths of both can help to overcome, the drawbacks of whisper small model.
5. Experiment with hyperparameters, especially for Wav2Vec2, to improve performance on non-English languages.

To reproduce the results from this notebook:

Download the notebook file you received and open it in Google Colab.
Install any necessary dependencies by running the installation commands in the first few cells (if applicable). You may need to install packages like TensorFlow, PyTorch, etc.
Upload the dataset to Colab using the file upload feature, or provide a link if the dataset is hosted online. Make sure it is properly formatted.
Run the notebook cells in sequence to load the dataset, process the audio, evaluate the models, and visualize results.

