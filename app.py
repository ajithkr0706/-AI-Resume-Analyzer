from flask import Flask, render_template, request
import fitz  # PyMuPDF
import spacy
import re
import os

app = Flask(__name__)
nlp = spacy.load("en_core_web_sm")

UPLOAD_FOLDER = 'uploads'
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER
os.makedirs(UPLOAD_FOLDER, exist_ok=True)

def extract_text_from_pdf(pdf_path):
    text = ""
    with fitz.open(pdf_path) as doc:
        for page in doc:
            text += page.get_text()
    return text

def analyze_resume(text):
    doc = nlp(text)
    skills = []
    keywords = ["python", "machine learning", "data", "flask", "ai", "sql", "html", "css", "javascript"]
    
    for token in doc:
        if token.text.lower() in keywords:
            skills.append(token.text.lower())

    total_keywords = len(keywords)
    matched = len(set(skills))
    match_percent = round((matched / total_keywords) * 100, 2)

    return {
        "skills_found": list(set(skills)),
        "match_percent": match_percent,
        "missing_skills": list(set(keywords) - set(skills))
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

    filepath = os.path.join(app.config['UPLOAD_FOLDER'], file.filename)
    file.save(filepath)

    text = extract_text_from_pdf(filepath)
    result = analyze_resume(text)

    return render_template('result.html', result=result, filename=file.filename)

if __name__ == '__main__':
    app.run(debug=True)
