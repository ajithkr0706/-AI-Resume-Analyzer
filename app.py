from flask import Flask, render_template, request
import fitz  # PyMuPDF
import spacy
import re
import os
import pickle
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

app = Flask(__name__)

# Lightweight spaCy tokenizer.
# Does NOT require en_core_web_sm.
nlp = spacy.blank("en")

# Load classification model, vectorizer and category skills mapping
model_path = os.path.join('model', 'resume_classifier.pkl')
vectorizer_path = os.path.join('model', 'tfidf_vectorizer.pkl')
skills_path = os.path.join('model', 'category_skills.pkl')

classifier = None
vectorizer = None
category_skills = {}

if os.path.exists(model_path) and os.path.exists(vectorizer_path) and os.path.exists(skills_path):
    with open(model_path, 'rb') as f:
        classifier = pickle.load(f)

    with open(vectorizer_path, 'rb') as f:
        vectorizer = pickle.load(f)

    with open(skills_path, 'rb') as f:
        category_skills = pickle.load(f)


def preprocess_text(text):
    doc = nlp(text.lower())

    tokens = [
        token.text
        for token in doc
        if not token.is_stop and token.is_alpha
    ]

    return " ".join(tokens)


UPLOAD_FOLDER = '/tmp/uploads'
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER
os.makedirs(UPLOAD_FOLDER, exist_ok=True)


def extract_text_from_pdf(pdf_path):
    text = ""

    with fitz.open(pdf_path) as doc:
        for page in doc:
            text += page.get_text()

    return text


def analyze_resume(text, job_description=""):

    doc = nlp(text)

    # Information Extraction
    email_match = re.search(
        r'\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Z|a-z]{2,}\b',
        text
    )

    phone_match = re.search(
        r'\(?\b\d{3}\)?[-.\s]?\d{3}[-.\s]?\d{4}\b',
        text
    )

    email = email_match.group(0) if email_match else "Not Found"
    phone = phone_match.group(0) if phone_match else "Not Found"

    # CATEGORY PREDICTION
    predicted_category = "General / Unclassified"

    if classifier and vectorizer:
        cleaned_text = preprocess_text(text)
        features = vectorizer.transform([cleaned_text])
        predicted_category = classifier.predict(features)[0]

    # Define Skills Keywords and ML Reference Text
    if job_description.strip():

        reference_text = job_description

        jd_doc = nlp(job_description)

        # Extract useful words from job description
        keywords = list(set([
            token.text.lower()
            for token in jd_doc
            if not token.is_stop
            and token.is_alpha
            and len(token.text) > 2
        ]))

    else:

        # Use standard skills for predicted category
        if predicted_category in category_skills:
            keywords = category_skills[predicted_category]
        else:
            keywords = [
                "python",
                "machine learning",
                "data",
                "flask",
                "ai",
                "sql",
                "html",
                "css",
                "javascript"
            ]

        reference_text = " ".join(keywords)

    skills = []

    for token in doc:
        if token.text.lower() in keywords:
            skills.append(token.text.lower())

    # ML SCORE CALCULATION
    temp_vectorizer = TfidfVectorizer(stop_words='english')

    try:

        tfidf_matrix = temp_vectorizer.fit_transform(
            [text, reference_text]
        )

        ml_score = cosine_similarity(
            tfidf_matrix[0:1],
            tfidf_matrix[1:2]
        )[0][0]

        match_percent = round(ml_score * 100, 2)

    except ValueError:

        match_percent = 0.0

    return {
        "email": email,
        "phone": phone,
        "skills_found": list(set(skills)),
        "match_percent": match_percent,
        "missing_skills": list(set(keywords) - set(skills)),
        "predicted_category": predicted_category
    }


@app.route('/')
def index():
    return render_template('index.html')


@app.route('/analyze', methods=['POST'])
def analyze():

    if 'resume' not in request.files:
        return "No file uploaded", 400

    file = request.files['resume']

    if file.filename == '':
        return "Empty file", 400

    job_description = request.form.get(
        'job_description',
        ''
    )

    filepath = os.path.join(
        app.config['UPLOAD_FOLDER'],
        file.filename
    )

    file.save(filepath)

    text = extract_text_from_pdf(filepath)

    result = analyze_resume(
        text,
        job_description
    )

    return render_template(
        'result.html',
        result=result,
        filename=file.filename
    )


if __name__ == '__main__':
    app.run(debug=True)