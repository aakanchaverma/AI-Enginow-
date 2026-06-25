import streamlit as st
import pandas as pd

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

from utils import (
    extract_pdf_text,
    preprocess_text,
    extract_skills,
    calculate_skill_match,
    extract_experience,
    experience_score,
    skill_gap_analysis
)

# =====================================================
# PAGE CONFIG
# =====================================================

st.set_page_config(
    page_title="AI Resume Screening System",
    layout="wide"
)

st.title("AI-Powered Resume Screening & Candidate Ranking System")

st.markdown("""
This system automatically analyzes resumes,
matches them against a Job Description,
and ranks candidates based on relevance.
""")

# =====================================================
# INPUT TYPE
# =====================================================

input_type = st.radio(
    "Select Input Type",
    [
        "CSV Dataset",
        "PDF Resume",
        "Text Resume"
    ]
)

# =====================================================
# INPUT SECTION
# =====================================================

uploaded_csv = None
uploaded_pdf = None
resume_text = ""

if input_type == "CSV Dataset":

    uploaded_csv = st.file_uploader(
        "Upload Resume Dataset CSV",
        type=["csv"]
    )

elif input_type == "PDF Resume":

    uploaded_pdf = st.file_uploader(
        "Upload Resume PDF",
        type=["pdf"]
    )

elif input_type == "Text Resume":

    resume_text = st.text_area(
        "Paste Resume Text",
        height=250
    )

# =====================================================
# JOB DESCRIPTION
# =====================================================

job_description = st.text_area(
    "Paste Job Description",
    height=250
)

# =====================================================
# BUTTON
# =====================================================

run_button = st.button("Run Screening")

# =====================================================
# PROCESSING
# =====================================================

if run_button:

    if not job_description.strip():

        st.error(
            "Please enter a Job Description."
        )

        st.stop()

    jd_skills = extract_skills(
        job_description
    )

    st.write("JD Skills:", jd_skills)

    cleaned_jd = preprocess_text(
        job_description
    )

    # =================================================
    # CSV MODE
    # =================================================

    if input_type == "CSV Dataset":

        if uploaded_csv is None:

            st.error(
                "Please upload a CSV file."
            )

            st.stop()

        df = pd.read_csv(
            uploaded_csv
        )

        if "Resume_str" not in df.columns:

            st.error(
                "CSV must contain Resume_str column."
            )

            st.stop()

        with st.spinner(
            "Processing resumes..."
        ):

            df["cleaned_resume"] = (
                df["Resume_str"]
                .apply(preprocess_text)
            )

            df["skills"] = (
                df["Resume_str"]
                .apply(extract_skills)
            )

            df["skill_match_score"] = (
                df["skills"]
                .apply(
                    lambda x:
                    calculate_skill_match(
                        x,
                        jd_skills
                    )
                )
            )

            df["experience_years"] = (
                df["Resume_str"]
                .apply(
                    extract_experience
                )
            )

            df["experience_score"] = (
                df["experience_years"]
                .apply(
                    experience_score
                )
            )

            corpus = (
                df["cleaned_resume"]
                .tolist()
                +
                [cleaned_jd]
            )

            vectorizer = (
                TfidfVectorizer()
            )

            tfidf_matrix = (
                vectorizer.fit_transform(
                    corpus
                )
            )

            resume_vectors = (
                tfidf_matrix[:-1]
            )

            jd_vector = (
                tfidf_matrix[-1]
            )

            similarity_scores = (
                cosine_similarity(
                    resume_vectors,
                    jd_vector
                )
            )

            df["similarity_score"] = (
                similarity_scores
                .flatten()
                * 100
            )

            if (
                df["similarity_score"]
                .max()
                > 0
            ):

                df["similarity_score"] = (
                    df["similarity_score"]
                    /
                    df["similarity_score"]
                    .max()
                ) * 100
            
            tech_categories = ["INFORMATION-TECHNOLOGY","ENGINEERING"]
            df["category_bonus"] = df["Category"].apply(
                lambda x: 10
                if x in tech_categories
                else 0
            )

            df["final_score"] = (
                0.30 *
                df["similarity_score"]
                +
                0.50 *
                df["skill_match_score"]
                +
                0.20 *
                df["experience_score"]
            )

            ranked_df = (
                df.sort_values(
                    "final_score",
                    ascending=False
                )
                .reset_index(
                    drop=True
                )
            )

            ranked_df["Rank"] = (
                ranked_df.index + 1
            )

        st.success(
            "Ranking Completed Successfully!"
        )

        st.subheader(
            "Top Ranked Candidates"
        )

        display_cols = [
            col
            for col in [
                "Rank",
                "Category",
                "similarity_score",
                "skill_match_score",
                "experience_score",
                "final_score"
            ]
            if col in ranked_df.columns
        ]

        st.subheader("Top Candidate Skills")

        st.dataframe(
            ranked_df[
                [
                    "Category",
                    "skills",
                    "skill_match_score",
                    "similarity_score",
                    "final_score"
                ]
            ].head(20)
        )

        st.dataframe(
            ranked_df[
                display_cols
            ]
        )

        csv = ranked_df.to_csv(
            index=False
        )

        st.download_button(
            label="Download Results CSV",
            data=csv,
            file_name="ranking_results.csv",
            mime="text/csv"
        )

    # =================================================
    # PDF MODE
    # =================================================

    elif input_type == "PDF Resume":

        if uploaded_pdf is None:

            st.error(
                "Please upload a PDF file."
            )

            st.stop()

        resume_text = (
            extract_pdf_text(
                uploaded_pdf
            )
        )

        cleaned_resume = (
            preprocess_text(
                resume_text
            )
        )

        candidate_skills = (
            extract_skills(
                resume_text
            )
        )

        skill_match = (
            calculate_skill_match(
                candidate_skills,
                jd_skills
            )
        )

        exp_years = (
            extract_experience(
                resume_text
            )
        )

        exp_score = (
            experience_score(
                exp_years
            )
        )

        corpus = [
            cleaned_resume,
            cleaned_jd
        ]

        vectorizer = (
            TfidfVectorizer()
        )

        tfidf_matrix = (
            vectorizer.fit_transform(
                corpus
            )
        )

        similarity = (
            cosine_similarity(
                tfidf_matrix[0:1],
                tfidf_matrix[1:2]
            )[0][0]
        ) * 100

        final_score = (
            0.30 * similarity
            +
            0.50 * skill_match
            +
            0.20 * exp_score
        )

        matched, missing = (
            skill_gap_analysis(
                candidate_skills,
                jd_skills
            )
        )

        st.success(
            "Resume Evaluated Successfully!"
        )

        c1, c2 = st.columns(2)

        with c1:

            st.metric(
                "Similarity Score",
                round(
                    similarity,
                    2
                )
            )

            st.metric(
                "Skill Match %",
                round(
                    skill_match,
                    2
                )
            )

        with c2:

            st.metric(
                "Experience Score",
                round(
                    exp_score,
                    2
                )
            )

            st.metric(
                "Final Score",
                round(
                    final_score,
                    2
                )
            )

        st.subheader(
            "Skill Gap Analysis"
        )

        st.write(
            "Matched Skills:",
            matched
        )

        st.write(
            "Missing Skills:",
            missing
        )

    # =================================================
    # TEXT MODE
    # =================================================

    elif input_type == "Text Resume":

        if not resume_text.strip():

            st.error(
                "Please paste resume text."
            )

            st.stop()

        cleaned_resume = (
            preprocess_text(
                resume_text
            )
        )

        candidate_skills = (
            extract_skills(
                resume_text
            )
        )

        skill_match = (
            calculate_skill_match(
                candidate_skills,
                jd_skills
            )
        )

        exp_years = (
            extract_experience(
                resume_text
            )
        )

        exp_score = (
            experience_score(
                exp_years
            )
        )

        corpus = [
            cleaned_resume,
            cleaned_jd
        ]

        vectorizer = (
            TfidfVectorizer()
        )

        tfidf_matrix = (
            vectorizer.fit_transform(
                corpus
            )
        )

        similarity = (
            cosine_similarity(
                tfidf_matrix[0:1],
                tfidf_matrix[1:2]
            )[0][0]
        ) * 100

        final_score = (
            0.30 * similarity
            +
            0.50 * skill_match
            +
            0.20 * exp_score
        )

        matched, missing = (
            skill_gap_analysis(
                candidate_skills,
                jd_skills
            )
        )

        st.success(
            "Resume Evaluated Successfully!"
        )

        c1, c2 = st.columns(2)

        with c1:

            st.metric(
                "Similarity Score",
                round(
                    similarity,
                    2
                )
            )

            st.metric(
                "Skill Match %",
                round(
                    skill_match,
                    2
                )
            )

        with c2:

            st.metric(
                "Experience Score",
                round(
                    exp_score,
                    2
                )
            )

            st.metric(
                "Final Score",
                round(
                    final_score,
                    2
                )
            )

        st.subheader(
            "Skill Gap Analysis"
        )

        st.write(
            "Matched Skills:",
            matched
        )

        st.write(
            "Missing Skills:",
            missing
        )