import os
import pickle
import spacy
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression

# Ensure the model directory exists
os.makedirs("model", exist_ok=True)

print("Loading spaCy model...")
nlp = spacy.load("en_core_web_sm")

# Define target skills per category
CATEGORY_SKILLS = {
    "Data Science & Machine Learning": [
        "python", "machine learning", "deep learning", "pandas", "numpy", 
        "scikit-learn", "tensorflow", "pytorch", "sql", "tableau", "data analysis"
    ],
    "Web Development": [
        "html", "css", "javascript", "react", "node.js", "flask", "django", 
        "sql", "mongodb", "bootstrap", "typescript"
    ],
    "Mobile App Development": [
        "android", "ios", "swift", "kotlin", "java", "flutter", 
        "react native", "dart", "xcode", "mobile ui"
    ],
    "DevOps & Cloud Engineering": [
        "aws", "azure", "docker", "kubernetes", "jenkins", "cicd", 
        "terraform", "linux", "ansible", "cloud architecture", "git"
    ],
    "HR & Talent Acquisition": [
        "recruiting", "sourcing", "onboarding", "talent acquisition", "hr compliance", 
        "interview scheduling", "payroll management", "hiring pipelines"
    ],
    "Finance & Business Analysis": [
        "financial modeling", "budgeting", "excel vba", "forecasting", 
        "business analysis", "powerbi", "tableau", "consulting", "market research"
    ]
}

# Labeled training dataset
data = [
    # Data Science & Machine Learning
    ("Experienced Data Scientist with expertise in machine learning, deep learning, python, pandas, numpy, and scikit-learn. Strong SQL skills, data modeling, regression, and building predictive models.", "Data Science & Machine Learning"),
    ("Machine Learning Engineer specialized in neural networks, TensorFlow, PyTorch, computer vision, and NLP. Developed recommendation systems and natural language processing pipelines using python.", "Data Science & Machine Learning"),
    ("Data Analyst proficient in SQL, Python, Tableau, data visualization, statistical analysis, and predictive analysis. Experienced in cleaning complex datasets and extracting insights.", "Data Science & Machine Learning"),
    ("AI Engineer with hands-on experience in training deep neural networks, reinforcement learning, ML pipelines, scikit-learn, pandas, numpy, data engineering, and deploying models to production.", "Data Science & Machine Learning"),
    ("Graduate in Statistics and Computer Science. Skilled in data science, predictive modeling, R, Python, machine learning algorithms, clustering, and classification.", "Data Science & Machine Learning"),
    ("Senior Data Science Specialist. Expert in statistical modeling, neural networks, predictive forecasting, spark, data analytics, regression, scikit-learn, pandas, numpy, Python, and SQL.", "Data Science & Machine Learning"),
    ("Business intelligence and data analysis expert. Building custom machine learning solutions, database queries in SQL, visualizations in Tableau, and predictive analytics scripts in python.", "Data Science & Machine Learning"),

    # Web Development
    ("Full Stack Web Developer with 3 years of experience in HTML, CSS, JavaScript, React, and Node.js. Building responsive web applications with Flask, Django, and SQL databases.", "Web Development"),
    ("Frontend Developer specialized in HTML5, CSS3, modern JavaScript, React, Vue, Tailwind CSS, Bootstrap. Passionate about user interface design, UX, and cross-browser compatibility.", "Web Development"),
    ("Backend Engineer skilled in Python, Flask, Node.js, Express, PostgreSQL, MongoDB, RESTful APIs, and microservices architecture. Focus on database optimization and system scalability.", "Web Development"),
    ("Software Developer building modern web applications. Proficient in HTML, CSS, JS, React, Node, Webpack, Git, and web standards. Experienced in responsive and mobile-first design.", "Web Development"),
    ("Web Designer and Developer. UI/UX design, wireframing, Figma, HTML, CSS, JavaScript, WordPress, frontend development, creating interactive websites.", "Web Development"),
    ("Junior Web Developer with strong foundations in HTML, CSS, JavaScript, React, and REST API consumption. Enthusiastic about clean code, responsive design, and CSS transitions.", "Web Development"),
    ("Backend Web Developer focused on API design, database schemas, authentication systems, NodeJS, Express, Flask, database optimization, MongoDB, PostgreSQL, SQL, and Git.", "Web Development"),

    # Mobile App Development
    ("Mobile Application Developer with expertise in iOS development, Swift, SwiftUI, Objective-C, Xcode. Published multiple apps to the App Store. CoreData, CocoaPods, and REST API integration.", "Mobile App Development"),
    ("Android Developer with 4 years of experience building native apps in Kotlin and Java. Strong knowledge of Android SDK, Jetpack Compose, Retrofit, MVVM architecture, and Play Store publishing.", "Mobile App Development"),
    ("Flutter Developer specializing in cross-platform mobile apps for Android and iOS. Proficient in Dart, state management (Provider, Bloc), Firebase integration, and custom animations.", "Mobile App Development"),
    ("Mobile App Engineer with experience in React Native, JavaScript, Redux, mobile UI/UX, push notifications, offline storage, and deploying apps to Apple App Store and Google Play Store.", "Mobile App Development"),
    ("Native mobile developer building high performance iOS and Android apps using Swift, Kotlin, Java, and native APIs. Integration with third-party libraries and REST APIs.", "Mobile App Development"),
    ("Android App Architect with strong experience in Kotlin, Java, Gradle build system, Android Studio, material design guidelines, local SQLite database storage, and push notifications.", "Mobile App Development"),
    ("iOS Developer building modern applications using Swift, SwiftUI, Apple guidelines, Xcode profiling, test-driven development, and publishing apps to the App Store.", "Mobile App Development"),

    # DevOps & Cloud Engineering
    ("DevOps Engineer with experience in AWS, Docker, Kubernetes, Jenkins CI/CD pipelines, Terraform for infrastructure as code, Ansible, Linux administration, and bash scripting.", "DevOps & Cloud Engineering"),
    ("Cloud Solutions Architect certified in Azure and AWS. Specializing in cloud migrations, serverless computing, infrastructure automation, Docker containerization, and monitoring tools.", "DevOps & Cloud Engineering"),
    ("System Administrator and DevOps Engineer. Experienced in managing Linux servers, network security, bash scripting, Docker containers, CI/CD pipelines, Git, and system monitoring.", "DevOps & Cloud Engineering"),
    ("Site Reliability Engineer (SRE) focused on automation, infrastructure orchestration with Kubernetes and Terraform, CI/CD using GitLab, scripting in Python and Go, and cloud platforms.", "DevOps & Cloud Engineering"),
    ("Infrastructure Engineer with deep knowledge of Docker container security, Kubernetes clustering, AWS CloudFormation, Jenkins, Linux systems, network configuration, and firewalls.", "DevOps & Cloud Engineering"),
    ("Cloud Engineer experienced in Kubernetes deployments, Docker container orchestration, Ansible playbooks, AWS IAM, VPC setups, Terraform provisioning, and Git version control.", "DevOps & Cloud Engineering"),
    ("System Admin specializing in Linux system maintenance, security patch updates, automation via bash scripts, Docker configuration, AWS administration, and infrastructure scaling.", "DevOps & Cloud Engineering"),

    # HR & Talent Acquisition
    ("HR Manager with expertise in talent acquisition, sourcing, recruitment, employee relations, onboarding, performance management, and developing company policies.", "HR & Talent Acquisition"),
    ("Technical Recruiter finding top software talent. Experienced in sourcing candidates via LinkedIn, resume screening, conducting interviews, applicant tracking systems (ATS), and salary negotiation.", "HR & Talent Acquisition"),
    ("Human Resources Specialist focusing on employee engagement, training and development, conflict resolution, compliance, benefits administration, and talent acquisition.", "HR & Talent Acquisition"),
    ("Talent Acquisition Lead managing end-to-end recruitment pipelines, employer branding, coordinating interview loops, hiring managers collaboration, and HR operations.", "HR & Talent Acquisition"),
    ("HR Generalist with hands-on experience in recruiting, onboarding, payroll processing, employee handbook preparation, employee grievances, and performance reviews.", "HR & Talent Acquisition"),
    ("Corporate Recruiter with deep understanding of sourcing strategies, screening resumes, interview coordination, onboarding procedures, HR administration, and HR policies.", "HR & Talent Acquisition"),
    ("Human Resources Coordinator specialized in recruitment operations, HR databases, employee onboarding, benefits management, payroll coordination, and compliance training.", "HR & Talent Acquisition"),

    # Finance & Business Analysis
    ("Financial Analyst with experience in financial modeling, budgeting, forecasting, variance analysis, advanced Excel (VBA, pivot tables), and corporate finance.", "Finance & Business Analysis"),
    ("Business Analyst bridging the gap between business and IT. Gathering requirements, writing user stories, process mapping, SQL, Jira, Agile methodologies, and documentation.", "Finance & Business Analysis"),
    ("Data and Business Analyst skilled in Excel, Power BI, Tableau, SQL, data gathering, market research, business intelligence, dashboard creation, and presenting insights to stakeholders.", "Finance & Business Analysis"),
    ("Investment Analyst specializing in market analysis, portfolio management, financial valuation, DCF models, equity research, risk assessment, and presenting reports.", "Finance & Business Analysis"),
    ("Business Consultant with experience in process improvement, strategic planning, project management, financial analysis, reporting, and stakeholder management.", "Finance & Business Analysis"),
    ("Senior Business Analyst specialized in requirements gathering, process flows, SWOT analysis, Excel forecasting models, PowerPoint presentations, and Agile product backlogs.", "Finance & Business Analysis"),
    ("Financial Consultant providing budgeting advice, financial statement analysis, investment analysis, excel automation, forecasting revenue, and stakeholder presentations.", "Finance & Business Analysis")
]

# Extract texts and labels
texts = [item[0] for item in data]
labels = [item[1] for item in data]

# Preprocessing function using spaCy
def preprocess_text(text):
    doc = nlp(text.lower())
    # Keep only letters, remove stop words and punctuation, apply lemmatization
    tokens = [token.lemma_ for token in doc if not token.is_stop and token.is_alpha]
    return " ".join(tokens)

print("Preprocessing training texts...")
preprocessed_texts = [preprocess_text(text) for text in texts]

print("Vectorizing training texts using TF-IDF...")
vectorizer = TfidfVectorizer(ngram_range=(1, 2))
X_train = vectorizer.fit_transform(preprocessed_texts)

print("Training Logistic Regression classifier...")
classifier = LogisticRegression(max_iter=1000)
classifier.fit(X_train, labels)

# Save models to file
print("Saving models to 'model/' directory...")
with open("model/resume_classifier.pkl", "wb") as f:
    pickle.dump(classifier, f)

with open("model/tfidf_vectorizer.pkl", "wb") as f:
    pickle.dump(vectorizer, f)

with open("model/category_skills.pkl", "wb") as f:
    pickle.dump(CATEGORY_SKILLS, f)

print("Training complete! Model files saved:")
print("- model/resume_classifier.pkl")
print("- model/tfidf_vectorizer.pkl")
print("- model/category_skills.pkl")
