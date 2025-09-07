import os
import pickle
from flask import Flask, request, render_template, redirect, url_for, session, jsonify
from werkzeug.utils import secure_filename

try:
    import PyPDF2
except Exception:
    PyPDF2 = None

try:
    import docx
except Exception:
    docx = None


UPLOAD_FOLDER = "uploads"
ALLOWED_EXTENSIONS = {"txt", "pdf", "docx"}
MODEL_PATH = os.path.join("models", "career_model.pkl")

os.makedirs(UPLOAD_FOLDER, exist_ok=True)

app = Flask(__name__)
app.config["UPLOAD_FOLDER"] = UPLOAD_FOLDER
app.secret_key = "change-this-secret"


career_roadmaps = {
    "Web Developer": ["HTML", "CSS", "JavaScript", "React", "Node.js", "Databases"],
    "Data Scientist": ["Python", "Pandas", "SQL", "Machine Learning", "Deep Learning"],
    "Game Developer": ["C++", "Unity", "C#", "3D Modeling (Blender)"],
    "AR/VR Developer": ["Unity", "Blender", "C#"],
    "Backend Developer": ["Python", "Flask", "Django", "SQL"],
    "Data Analyst": ["SQL", "Python", "Visualization (Tableau/Power BI)"],

    # Sub-roadmaps
    "HTML": ["Basics of HTML", "Forms and Input", "Tables", "Semantic HTML", "Best Practices"],
    "CSS": ["Selectors", "Box Model", "Flexbox", "Grid Layout", "Animations"],
    "JavaScript": ["Variables & Functions", "DOM Manipulation", "ES6+", "Async/Await", "APIs"],
    "React": ["JSX & Components", "State & Props", "Hooks", "Routing", "State Management (Redux)"],
    "Node.js": ["NPM & Packages", "Express.js", "Middleware", "APIs", "Authentication"],

    "Python": ["Syntax & Basics", "OOP", "Libraries (NumPy, Pandas)", "APIs", "Scripting"],
    "Pandas": ["Series & DataFrames", "Data Cleaning", "Merging & Grouping", "Visualization"],
    "SQL": ["DDL/DML", "Joins", "Subqueries", "Indexes", "Optimization"],

    "Machine Learning": ["Supervised Learning", "Unsupervised Learning", "Model Evaluation", "Scikit-learn"],
    "Deep Learning": ["Neural Networks Basics", "CNNs", "RNNs", "Transfer Learning", "TensorFlow/PyTorch"],

    "Unity": ["Basics & Interface", "C# Scripting", "Physics", "Animations", "Build & Deploy"],
    "Blender": ["Modeling Basics", "Texturing", "Lighting", "Rigging", "Animation"],
    "C#": ["Syntax", "OOP", "Collections", "Async Programming", "Unity Scripting"],

    "Flask": ["Routing", "Templates (Jinja)", "Forms & Validation", "Database Integration", "APIs"],
    "Django": ["Models & ORM", "Templates", "Authentication", "REST Framework", "Deployment"],

    "Tableau/Power BI": ["Data Import", "Charts & Graphs", "Dashboards", "Calculated Fields", "Storytelling"]
}


SKILLS_VOCAB = [s.lower() for s in [
    "Python", "Pandas", "NumPy", "SQL", "Java", "C++", "C#", "Unity", "Blender",
    "Flask", "Django", "HTML", "CSS", "JavaScript", "React", "Node.js", "Machine Learning",
    "Deep Learning", "TensorFlow", "PyTorch", "Tableau", "Power BI"
]]

model_pipeline = None
model_loaded = False
if os.path.exists(MODEL_PATH):
    try:
        with open(MODEL_PATH, "rb") as f:
            model_pipeline = pickle.load(f)
        model_loaded = True
        print("Loaded model from", MODEL_PATH)
    except Exception as e:
        print("Error loading model:", e)


def allowed_file(filename):
    return "." in filename and filename.rsplit(".", 1)[1].lower() in ALLOWED_EXTENSIONS


def extract_text_from_file(filepath):
    ext = filepath.rsplit(".", 1)[1].lower()
    try:
        if ext == "txt":
            with open(filepath, "r", encoding="utf-8", errors="ignore") as f:
                return f.read()
        elif ext == "pdf" and PyPDF2:
            with open(filepath, "rb") as f:
                reader = PyPDF2.PdfReader(f)
                return "\n".join(page.extract_text() or "" for page in reader.pages)
        elif ext in ["docx", "doc"] and docx:
            doc = docx.Document(filepath)
            return "\n".join([p.text for p in doc.paragraphs])
        else:
            with open(filepath, "r", encoding="utf-8", errors="ignore") as f:
                return f.read()
    except Exception as e:
        print("Error extracting text:", e)
        return ""


def extract_skills_rule_based(text):
    text_lower = text.lower()
    found = []
    for skill in SKILLS_VOCAB:
        if skill in text_lower and skill not in found:
            found.append(skill)
    return found


def recommend_careers_from_skills(skills):
    recommendations = []
    for career, req in career_roadmaps.items():
        score = len(set(s.lower() for s in skills) & set(r.lower() for r in req))
        if score > 0:
            recommendations.append({"career": career, "score": score})
    return sorted(recommendations, key=lambda x: x["score"], reverse=True)


def predict_with_model(text):
    if not model_loaded or model_pipeline is None:
        return None
    try:
        probs = model_pipeline.predict_proba([text])[0]
        classes = model_pipeline.classes_
        class_probs = sorted(zip(classes, probs), key=lambda x: x[1], reverse=True)
        predicted = class_probs[0][0]
        topk = [{"career": c, "prob": float(p)} for c, p in class_probs[:5]]
        return {"predicted": predicted, "predicted_prob": float(class_probs[0][1]), "topk": topk}
    except Exception as e:
        print("Model prediction error:", e)
        return None


@app.route("/")
def index():
    return render_template("index.html", model_loaded=model_loaded, careers=list(career_roadmaps.keys()))


@app.route("/upload_resume", methods=["GET"])
def upload_resume_page():
    return render_template("index.html", model_loaded=model_loaded, careers=list(career_roadmaps.keys()))


@app.route("/upload_resume", methods=["POST"])
def handle_upload_resume():
    if "file" not in request.files:
        return render_template("results.html", error="No file uploaded", model_loaded=model_loaded)

    file = request.files["file"]
    if file.filename == "":
        return render_template("results.html", error="No file selected", model_loaded=model_loaded)

    if not allowed_file(file.filename):
        return render_template("results.html", error="Unsupported file type", model_loaded=model_loaded)

    filename = secure_filename(file.filename)
    path = os.path.join(app.config["UPLOAD_FOLDER"], filename)
    file.save(path)

    text = extract_text_from_file(path)
    os.remove(path)

    if not text.strip():
        return render_template("results.html", error="Could not extract text from the file", model_loaded=model_loaded)

    skills = extract_skills_rule_based(text)
    rule_recs = recommend_careers_from_skills(skills)
    ml_result = predict_with_model(text) if model_loaded else None

    return render_template("results.html",
                           text_snippet=text[:2000],
                           skills=skills,
                           rule_recs=rule_recs,
                           ml_result=ml_result,
                           model_loaded=model_loaded)


@app.route("/roadmap/<topic>")
def show_roadmap(topic):
    roadmap = career_roadmaps.get(topic, [])
    completed_steps_list = session.get("progress", {}).get(topic, [])

    parent = None
    for career, steps in career_roadmaps.items():
        if topic in steps:
            parent = career
            break

    return render_template("roadmap.html",
                           topic=topic,
                           roadmap=roadmap,
                           all_topics=career_roadmaps.keys(),
                           parent=parent,  # ✅ Pass parent topic
                           back_link=url_for("show_roadmap", topic=parent) if parent else url_for("index"),
                           completed_steps_list=completed_steps_list)


@app.route("/complete_step/<topic>/<int:step_index>", methods=["POST"])
def complete_step(topic, step_index):
    if "progress" not in session:
        session["progress"] = {}

    if topic not in session["progress"]:
        session["progress"][topic] = []

    action = request.json.get("action")

    if action == "check":
        if step_index not in session["progress"][topic]:
            session["progress"][topic].append(step_index)
    elif action == "uncheck":
        if step_index in session["progress"][topic]:
            session["progress"][topic].remove(step_index)

    session.modified = True

    return jsonify({"status": "success"})


@app.route("/trends/<career>")
def show_trends(career):
    trends = {
        "Web Developer": ["Web3", "PWA", "AI integration"],
        "Data Scientist": ["Generative AI", "LLMs", "MLOps"],
        "Game Developer": ["VR/AR games", "AI NPCs"],
        "AR/VR Developer": ["Immersive AR apps", "XR platforms"],
        "Backend Developer": ["Serverless", "Microservices"],
        "Data Analyst": ["BI automation", "Data storytelling"]
    }

    return render_template("trends.html",
                           career=career,
                           trends=trends.get(career, ["No trends available"]),
                           back_link=url_for("show_roadmap", topic=career))


if __name__ == "__main__":
    app.run(debug=True, host="0.0.0.0", port=5000)
