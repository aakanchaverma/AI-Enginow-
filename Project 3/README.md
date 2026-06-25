# AI-Powered Resume Screening & Candidate Ranking System

## Project Overview

The AI-Powered Resume Screening & Candidate Ranking System is an Applicant Tracking System (ATS)-inspired application that automates resume evaluation and candidate ranking.

The system accepts resumes in multiple formats, extracts relevant information using Natural Language Processing (NLP), compares candidate profiles against a job description, and generates ranking scores based on relevance.

This project aims to reduce manual effort in recruitment and provide recruiters with a faster and more consistent candidate screening process.

---

## Problem Statement

Recruiters often receive hundreds of resumes for a single job opening. Manually reviewing every resume is time-consuming, repetitive, and susceptible to human bias.

This project automates the screening process by:

* Parsing resumes
* Extracting skills and experience
* Matching resumes against job requirements
* Generating candidate relevance scores
* Ranking applicants automatically

---

## Features

### Resume Input Support

* CSV Resume Dataset
* PDF Resume Upload
* Direct Resume Text Input

### NLP Pipeline

* Text Cleaning
* Lowercasing
* Stopword Removal
* Lemmatization
* Skill Extraction
* Experience Extraction

### Candidate Evaluation

* TF-IDF Vectorization
* Cosine Similarity Matching
* Skill Match Scoring
* Experience Scoring
* Final Candidate Ranking

### Dashboard Features

* Interactive Streamlit Interface
* Resume Ranking Table
* Skill Gap Analysis
* Download Ranked Results

---

## Technology Stack

### Programming Language

* Python

### Libraries

* Pandas
* NumPy
* Scikit-Learn
* NLTK
* PDFPlumber
* Streamlit
* Matplotlib

### NLP Techniques

* Tokenization
* Stopword Removal
* Lemmatization
* TF-IDF Vectorization
* Cosine Similarity

---

## Dataset

Dataset Used:

Resume Dataset from Kaggle

Dataset Characteristics:

* 586 Resume Records
* Multiple Professional Categories
* Resume Text Stored in CSV Format

Example Categories:

* Information Technology
* Engineering
* Finance
* Banking
* Consultant
* Healthcare
* HR
* Agriculture
* Sales

Additionally, PDF-based resumes were tested using custom resumes.

---

## System Architecture

Resume Input (CSV / PDF / Text)
↓
Text Extraction
↓
Text Preprocessing
↓
Skill Extraction
↓
Experience Extraction
↓
TF-IDF Vectorization
↓
Cosine Similarity Computation
↓
Candidate Scoring
↓
Candidate Ranking
↓
Dashboard Output

---

## Methodology

### Step 1: Resume Ingestion

The system accepts resumes through:

* CSV files
* PDF files
* Direct text input

### Step 2: Text Preprocessing

The resume content is cleaned using:

* Lowercasing
* Removal of special characters
* Stopword removal
* Lemmatization

### Step 3: Skill Extraction

A predefined technical skill dictionary is used to identify skills present in resumes and job descriptions.

### Step 4: Experience Extraction

Regular expressions are used to detect years of experience from resume text.

### Step 5: Similarity Computation

TF-IDF Vectorization converts text into numerical vectors.

Cosine Similarity is then used to measure similarity between resumes and job descriptions.

### Step 6: Candidate Ranking

Final candidate scores are calculated using a weighted scoring approach combining:

* Similarity Score
* Skill Match Score
* Experience Score

Candidates are ranked from highest to lowest score.

---

## Project Structure

AI_Resume_Screener/

├── Resume_Screening_System.ipynb

├── app.py

├── utils.py

├── requirements.txt

├── README.md

├── data/

│ ├── Resume.csv

│ └── My_resume.pdf

└── outputs/

---

## Running the Project

Install dependencies:

pip install -r requirements.txt

Run Streamlit:

streamlit run app.py

---

## Future Enhancements

* Sentence Transformer Embeddings
* BERT-Based Resume Matching
* Named Entity Recognition (NER)
* LLM-Based Resume Summarization
* Bias Detection and Fairness Scoring
* Recruiter Feedback Loop
* Multi-Job Comparison

---

## Conclusion

This project demonstrates the implementation of an ATS-style resume screening system using NLP and Machine Learning techniques. The solution automates candidate evaluation, improves recruitment efficiency, and provides a scalable approach for resume screening.
