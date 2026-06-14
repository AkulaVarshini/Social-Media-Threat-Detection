from flask import Flask, render_template, request, redirect, url_for, session, flash, jsonify
from flask_sqlalchemy import SQLAlchemy
from flask_login import LoginManager, UserMixin, login_user, logout_user, login_required, current_user
from werkzeug.security import generate_password_hash, check_password_hash
from werkzeug.utils import secure_filename
import joblib
import os
import re
import pandas as pd
import numpy as np
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image
import cv2

app = Flask(__name__)
app.config['SQLALCHEMY_DATABASE_URI'] = 'sqlite:///users.db'
app.config['SECRET_KEY'] = 'your-secret-key-change-this'  # Change this to a secure key
app.config['MAX_CONTENT_LENGTH'] = 16 * 1024 * 1024  # 16MB max file size
app.config['UPLOAD_FOLDER'] = os.path.join('static', 'uploads')

# Create upload folder if it doesn't exist
os.makedirs(app.config['UPLOAD_FOLDER'], exist_ok=True)

ALLOWED_EXTENSIONS = {'png', 'jpg', 'jpeg', 'gif', 'bmp'}

# Suspicious keyword detection (case-insensitive substring matching)
PREDATOR_KEYWORDS = [
    "private", "secret", "alone", "photos", "send pic", "don't tell", "meet secretly", "video call"
]
HARASSMENT_KEYWORDS = [
    "idiot", "stupid", "dumb", "useless", "loser"
]

def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

db = SQLAlchemy(app)
login_manager = LoginManager()
login_manager.init_app(app)
login_manager.login_view = 'login'

# User Model
class User(UserMixin, db.Model):
    id = db.Column(db.Integer, primary_key=True)
    username = db.Column(db.String(80), unique=True, nullable=False)
    email = db.Column(db.String(120), unique=True, nullable=False)
    password = db.Column(db.String(200), nullable=False)

    def set_password(self, password):
        self.password = generate_password_hash(password)

    def check_password(self, password):
        return check_password_hash(self.password, password)

@login_manager.user_loader
def load_user(user_id):
    return User.query.get(int(user_id))

# Load model safely
model_path = os.path.join("model", "model.pkl")
vectorizer_path = os.path.join("model", "vectorizer.pkl")
image_model_path = os.path.join("model", "image_model.h5")
image_label_encoder_path = os.path.join("model", "image_label_encoder.pkl")

if os.path.exists(model_path) and os.path.exists(vectorizer_path):
    model = joblib.load(model_path)
    vectorizer = joblib.load(vectorizer_path)
else:
    model = None
    vectorizer = None

# Load image model
if os.path.exists(image_model_path):
    try:
        image_model = load_model(image_model_path)
        image_label_encoder = joblib.load(image_label_encoder_path)
    except Exception as e:
        print(f"Note: Image model not yet trained. {e}")
        image_model = None
        image_label_encoder = None
else:
    image_model = None
    image_label_encoder = None

# Dashboard stats
total_messages = 0
harmful_messages = 0


@app.route("/register", methods=["GET", "POST"])
def register():
    if request.method == "POST":
        username = request.form.get("username")
        email = request.form.get("email")
        password = request.form.get("password")
        confirm_password = request.form.get("confirm_password")

        if not username or not email or not password or not confirm_password:
            flash("All fields are required!", "error")
            return redirect(url_for("register"))

        if password != confirm_password:
            flash("Passwords do not match!", "error")
            return redirect(url_for("register"))

        user = User.query.filter_by(username=username).first()
        if user:
            flash("Username already exists!", "error")
            return redirect(url_for("register"))

        user = User.query.filter_by(email=email).first()
        if user:
            flash("Email already registered!", "error")
            return redirect(url_for("register"))

        new_user = User(username=username, email=email)
        new_user.set_password(password)
        db.session.add(new_user)
        db.session.commit()

        flash("Registration successful! Please login.", "success")
        return redirect(url_for("login"))

    return render_template("register.html")


@app.route("/login", methods=["GET", "POST"])
def login():
    if current_user.is_authenticated:
        return redirect(url_for("home"))

    if request.method == "POST":
        username_or_email = request.form.get("username_or_email")
        password = request.form.get("password")

        if not username_or_email or not password:
            flash("Username/Email and password are required!", "error")
            return redirect(url_for("login"))

        user = User.query.filter((User.username == username_or_email) | (User.email == username_or_email)).first()

        if user and user.check_password(password):
            login_user(user)
            flash("Login successful!", "success")
            return redirect(url_for("home"))
        else:
            flash("Invalid username/email or password!", "error")
            return redirect(url_for("login"))

    return render_template("login.html")


@app.route("/logout")
@login_required
def logout():
    logout_user()
    flash("You have been logged out.", "success")
    return redirect(url_for("home"))


@app.route("/")
def home():
    return render_template("home.html")


@app.route("/detect", methods=["GET", "POST"])
def detect():
    global total_messages, harmful_messages

    result = None
    confidence = None
    risk = None
    alert_message = None
    detected_words = []

    if request.method == "POST":
        text = request.form["message"]
        text_lower = text.lower()

        # Detect suspicious keywords (word boundaries for single words, phrase matching where needed)
        def _matches_keyword(keyword: str, text: str) -> bool:
            # Use word boundary matching for single keywords to reduce false positives
            if " " in keyword:
                return re.search(re.escape(keyword), text) is not None
            return re.search(r"\b" + re.escape(keyword) + r"\b", text) is not None

        found_predator = [kw for kw in PREDATOR_KEYWORDS if _matches_keyword(kw, text_lower)]
        found_harassment = [kw for kw in HARASSMENT_KEYWORDS if _matches_keyword(kw, text_lower)]

        # Keep the list of detected keywords unique and maintain order
        detected_words = []
        for kw in found_predator + found_harassment:
            if kw not in detected_words:
                detected_words.append(kw)

        if model:
            vector = vectorizer.transform([text])
            prediction = model.predict(vector)[0]
            proba = max(model.predict_proba(vector)[0])

            total_messages += 1

            if prediction != "safe":
                harmful_messages += 1

            result = prediction.capitalize()
            confidence = round(proba * 100, 2)

            if confidence > 85:
                risk = "High"
            elif confidence >= 60:
                risk = "Medium"
            else:
                risk = "Low"

            # Alert message based on classification
            if result == "Predator":
                alert_message = "🚨 Predator behaviour detected!"
            elif result == "Harassment":
                alert_message = "⚠ Harassment detected!"
            elif result == "Safe":
                alert_message = "✅ Safe message"

    # If caller expects JSON (API usage), return structured JSON response
    if request.is_json or request.accept_mimetypes.accept_json > request.accept_mimetypes.accept_html:
        return jsonify({
            "Prediction": result,
            "Confidence": f"{confidence}%" if confidence is not None else None,
            "Risk Level": risk,
        })

    return render_template("detect.html",
                           result=result,
                           confidence=confidence,
                           risk=risk,
                           alert_message=alert_message,
                           detected_words=detected_words)


@app.route("/detect_image", methods=["GET", "POST"])
def detect_image():
    """Detect image content - predator, harassment, or safe"""
    result = None
    confidence = None
    risk = None
    image_filename = None
    error = None
    alert_message = None

    if request.method == "POST":
        # Check if file is in request
        if 'image' not in request.files:
            error = "No image file provided"
            return render_template("detect_image.html", error=error)

        file = request.files['image']
        
        if file.filename == '':
            error = "No file selected"
            return render_template("detect_image.html", error=error)

        if not allowed_file(file.filename):
            error = "Invalid file type. Allowed: " + ", ".join(ALLOWED_EXTENSIONS)
            return render_template("detect_image.html", error=error)

        if image_model is None:
            error = "Image model not available. Please train the model first."
            return render_template("detect_image.html", error=error)

        try:
            # Save uploaded file
            filename = secure_filename(file.filename)
            filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
            file.save(filepath)
            image_filename = filename

            # Preprocess image
            img = image.load_img(filepath, target_size=(224, 224))
            img_array = image.img_to_array(img)
            img_array = np.expand_dims(img_array, axis=0)
            img_array = img_array / 255.0

            # Make prediction
            predictions = image_model.predict(img_array, verbose=0)
            confidence_val = float(np.max(predictions))
            predicted_class_idx = np.argmax(predictions)
            
            # Map to label
            predicted_label = image_label_encoder.inverse_transform([predicted_class_idx])[0]
            
            result = predicted_label.capitalize()
            confidence = round(confidence_val * 100, 2)

            # Determine risk level
            if result == "Predator":
                risk = "High"
            elif result == "Harassment":
                risk = "Medium"
            else:
                risk = "Low"

            # Alert message based on classification
            if result == "Predator":
                alert_message = "⚠️ ALERT: Possible Child Predator Message Detected"
            elif result == "Harassment":
                alert_message = "⚠️ Warning: Harassment Content Detected"
            elif result == "Safe":
                alert_message = "✅ Message appears safe"

            total_messages += 1
            if predicted_label != "safe":
                harmful_messages += 1

        except Exception as e:
            error = f"Error processing image: {str(e)}"

    return render_template("detect_image.html",
                           result=result,
                           confidence=confidence,
                           risk=risk,
                           image_filename=image_filename,
                           alert_message=alert_message,
                           error=error)


@app.route("/dashboard")
def dashboard():
    # Compute model performance metrics using the existing trained model and vectorizer.
    accuracy = precision = recall = f1 = None
    error_message = None

    if model is None or vectorizer is None:
        error_message = "Model or vectorizer is not available. Train the model and restart the app."
    else:
        try:
            # Load evaluation dataset
            dataset_path = os.path.join("data", "dataset.csv")
            df = pd.read_csv(dataset_path)

            # Normalize labels to match model target values and avoid case mismatches
            label_map = {
                "safe": "safe",
                "predator": "predator",
                "cyber harassment": "harassment",
                "harassment": "harassment",
            }

            def normalize_label(label: str) -> str:
                if not isinstance(label, str):
                    return "safe"
                return label_map.get(label.strip().lower(), "safe")

            y_true = df["label"].astype(str).apply(normalize_label)
            X = vectorizer.transform(df["text"].astype(str))

            # The model may output labels in different casing (e.g. 'Safe', 'Predator',
            # 'Cyber Harassment'). Normalize predictions to match y_true.
            y_pred_raw = model.predict(X)
            y_pred = [normalize_label(pred) for pred in y_pred_raw]

            # Compute metrics
            accuracy = float(accuracy_score(y_true, y_pred))
            precision = float(precision_score(y_true, y_pred, average="weighted", zero_division=0))
            recall = float(recall_score(y_true, y_pred, average="weighted", zero_division=0))
            f1 = float(f1_score(y_true, y_pred, average="weighted", zero_division=0))
        except Exception as e:
            error_message = f"Unable to compute performance metrics: {e}"

    return render_template(
        "dashboard.html",
        accuracy=accuracy,
        precision=precision,
        recall=recall,
        f1=f1,
        error_message=error_message,
    )


@app.route("/about")
def about():
    return render_template("about.html")


@app.route("/admin")
def admin():
    return render_template("admin.html")


if __name__ == "__main__":
    with app.app_context():
        db.create_all()
    app.run(debug=True)
