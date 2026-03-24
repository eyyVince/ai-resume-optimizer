import streamlit as st
import pdfplumber
import re
import os
from dotenv import load_dotenv
from openai import OpenAI
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

load_dotenv()

client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

def extract_text_from_pdf(file):
    text = ""
    with pdfplumber.open(file) as pdf:
        for page in pdf.pages:
            text += page.extract_text() + " "
    return re.sub(r'\s+', ' ', text.strip())

def calculate_similarity(resume_text, job_text):
    vectorizer = TfidfVectorizer(stop_words='english')
    tfidf = vectorizer.fit_transform([resume_text, job_text])
    similarity = cosine_similarity(tfidf[0:1], tfidf[1:2])
    return round(similarity[0][0] * 100, 2)

def generate_feedback(resume_text, job_desc):
    prompt = f"""
    You are a resume optimization assistant.
    Compare this resume and job description, then suggest how to improve the resume to better match the job.

    Resume:
    {resume_text}

    Job Description:
    {job_desc}

    Provide bullet-point feedback with missing keywords or phrasing tips.
    """
    response = client.chat.completions.create(
        model="gpt-4o-mini",
        messages=[{"role": "user", "content": prompt}]
    )
    return response.choices[0].message.content.strip()

st.title("🧠 AI-Powered Resume Optimizer")

resume_file = st.file_uploader("📄 Upload your resume (PDF)", type=["pdf"])
job_desc = st.text_area("📝 Paste the job description here")

if st.button("Analyze"):
    if resume_file and job_desc:
        with st.spinner("Analyzing your resume..."):
            resume_text = extract_text_from_pdf(resume_file)
            score = calculate_similarity(resume_text, job_desc)
            feedback = generate_feedback(resume_text, job_desc)

        st.success(f"✅ Resume Match Score: **{score}%**")
        st.subheader("💡 Recommendations:")
        st.write(feedback)
    else:
        st.warning("Please upload a resume and paste a job description.")
