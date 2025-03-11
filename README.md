## Transcription-Lesotho

# Overview

Transcription-Lesotho is a project focused on processing and analyzing audio data from Lesotho. The goal is to develop a structured approach for handling, cleaning, and utilizing transcription data for linguistic and AI applications.

# Data Structure

The dataset consists of the following components:

Audio Files: Stored in audio/ directory in .wav format.

Transcriptions: Corresponding text files in transcripts/ directory with .txt format.

Metadata: CSV file (metadata.csv) containing details such as speaker ID, duration, and language labels.

Annotations: If applicable, labeled dataset in annotations/ for linguistic processing.

Metadata Schema:

Column Name

Description

file_name

Name of the audio file

speaker_id

Unique identifier for the speaker

duration

Duration of the audio file in seconds

transcript

Corresponding transcript of the audio

language

Language or dialect used in the audio

quality

Quality assessment score of the transcription

# Methodology

1. Data Collection

Audio data is collected from various sources, including interviews, radio broadcasts, and spontaneous speech recordings.

Each audio file is paired with a manually verified transcription.

2. Preprocessing

Convert all audio files to a uniform format (16kHz, mono, .wav).

Normalize transcriptions (lowercase conversion, punctuation removal if necessary).

Align audio files with their transcriptions.

3. Data Cleaning

To ensure data quality, we perform:

Noise Removal: Filtering out background noise.

Silence Trimming: Removing long pauses at the beginning or end of recordings.

Spelling Correction: Using NLP-based spell-checking tools.

Speaker Normalization: Ensuring consistent speaker labeling across datasets.

Duplicate Removal: Identifying and eliminating duplicate records.

4. Post-Processing

Tokenization and text normalization for NLP tasks.

Data splitting into train, validation, and test sets.

Formatting for use in speech recognition models (e.g., Kaldi, ESPnet, Whisper).

How to Clean the Data

Follow these steps to clean the dataset:

Install Dependencies:

pip install pandas numpy librosa

Run the Cleaning Script:

python clean_data.py

The script will:

Remove background noise.

Trim silence from audio files.

Normalize text transcriptions.

Validate Data Integrity:

python validate_data.py

This ensures all audio files have corresponding transcriptions and metadata is correctly structured.

Contribution

Feel free to contribute by submitting pull requests, reporting issues, or adding new features to the cleaning pipeline.

License

This project is licensed under the MIT License.