# utils.py

import re
import pdfplumber

from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer

stop_words = set(stopwords.words("english"))
lemmatizer = WordNetLemmatizer()

skills_list = [
    "python","java","sql","mysql","postgresql",
    "machine learning","deep learning",
    "tensorflow","pytorch","nlp",
    "streamlit","flask","django",
    "git","github",
    "tableau","power bi","excel",
    "aws","azure","gcp",
    "docker","kubernetes",
    "data analysis","data visualization",
    "statistics","data science",
    "c++","c","html","css",
    "javascript","react","nodejs",
    "mongodb","linux"
]


def extract_pdf_text(pdf_file):

    text = ""

    with pdfplumber.open(pdf_file) as pdf:

        for page in pdf.pages:

            page_text = page.extract_text()

            if page_text:
                text += page_text + "\n"

    return text


def preprocess_text(text):

    text = str(text).lower()

    text = re.sub(r'[^a-zA-Z\s]', ' ', text)

    tokens = text.split()

    tokens = [
        lemmatizer.lemmatize(word)
        for word in tokens
        if word not in stop_words
    ]

    return " ".join(tokens)


def extract_skills(text):

    text = str(text).lower()

    found_skills = []

    for skill in skills_list:

        pattern = r'\b' + re.escape(skill.lower()) + r'\b'

        if re.search(pattern, text):

            found_skills.append(skill)

    return list(set(found_skills))


def calculate_skill_match(
        candidate_skills,
        jd_skills):

    if len(jd_skills) == 0:
        return 0

    matched = set(candidate_skills).intersection(
        set(jd_skills)
    )

    return (
        len(matched) /
        len(jd_skills)
    ) * 100


def extract_experience(text):

    text = str(text).lower()

    patterns = [
        r'(\d+)\+?\s*years',
        r'(\d+)\+?\s*yrs',
        r'(\d+)\+?\s*year',
        r'experience\s*:?\s*(\d+)'
    ]

    for pattern in patterns:

        match = re.search(pattern, text)

        if match:
            return int(match.group(1))

    return 0


def experience_score(candidate_exp, required_exp=2):
    if candidate_exp <= 0:
        return 0
    score = (candidate_exp / required_exp) * 100
    return min(score, 100)


def skill_gap_analysis(
        candidate_skills,
        jd_skills):

    matched = list(
        set(candidate_skills)
        &
        set(jd_skills)
    )

    missing = list(
        set(jd_skills)
        -
        set(candidate_skills)
    )

    return matched, missing