from flask import Flask, render_template, request, redirect, url_for, session, jsonify
import pandas as pd
import joblib
import warnings
import logging
from datetime import datetime
import json
import os
from pathlib import Path

# Suppress sklearn warnings
warnings.filterwarnings("ignore", category=UserWarning, module="sklearn")

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# User data file path
USERS_FILE = Path(__file__).parent / "users_data.json"

def load_users():
    """Load users from JSON file"""
    if USERS_FILE.exists():
        with open(USERS_FILE, 'r') as f:
            return json.load(f)
    return {}

def save_users(users_dict):
    """Save users to JSON file"""
    with open(USERS_FILE, 'w') as f:
        json.dump(users_dict, f, indent=2)

# ---------------------------------------
# Initialize Flask
# ---------------------------------------
app = Flask(__name__)
app.secret_key = "supersecretkey123"
app.config['JSON_SORT_KEYS'] = False

# ---------------------------------------
# ---------------------------------------
# Load trained model with error handling
# ---------------------------------------
try:
    model = joblib.load("model.pkl")
    logger.info("Model loaded successfully")
except Exception as e:
    logger.error(f"Error loading model: {e}")
    model = None

# ---------------------------------------
# Feature columns (must match training)
# ---------------------------------------
feature_columns = [
"MemoryComplaints",
"SleepQuality",
"CholesterolHDL",
"BehavioralProblems",
"FunctionalAssessment",
"ADL",
"MMSE",
"FamilyHistoryAlzheimers",
"CardiovascularDisease",
"Diabetes",
"Hypertension",
"BMI",
"CholesterolLDL",
"EducationLevel"
]

# Authentication Routes
# ---------------------------------------

# Sign-up route
@app.route("/signup", methods=["GET", "POST"])
def signup():
    if request.method == "POST":
        username = request.form.get("username", "").strip()
        password = request.form.get("password", "").strip()
        confirm_password = request.form.get("confirm_password", "").strip()
        
        # Validation
        if not username or not password or not confirm_password:
            return render_template("signup.html", error="All fields are required")
        
        if len(username) < 2:
            return render_template("signup.html", error="Name must be at least 2 characters")
        
        if password != confirm_password:
            return render_template("signup.html", error="Passwords do not match")
        
        users = load_users()
        if username in users:
            return render_template("signup.html", error="User already exists")
        
        # Store user (in production, use proper hashing)
        users[username] = {"password": password}
        save_users(users)
        
        logger.info(f"New user registered: {username}")
        return redirect(url_for("signin"))
    
    return render_template("signup.html")

# Sign-in route
@app.route("/signin", methods=["GET", "POST"])
def signin():
    if request.method == "POST":
        username = request.form.get("username", "").strip()
        password = request.form.get("password", "").strip()
        
        if not username or not password:
            return render_template("signin.html", error="Username and password required")
        
        users = load_users()
        if username not in users or users[username]["password"] != password:
            return render_template("signin.html", error="Invalid username or password")
        
        # Set session
        session['user'] = username
        logger.info(f"User logged in: {username}")
        return redirect(url_for("home"))
    
    return render_template("signin.html")

# Logout route
@app.route("/logout")
def logout():
    username = session.pop('user', None)
    logger.info(f"User logged out: {username}")
    return redirect(url_for("home"))

# Home page
# ---------------------------------------
@app.route("/")
def home():
    # Check if user is logged in, pass user info to template if they are
    user = session.get('user')
    return render_template("index.html", user=user)


# ---------------------------------------
# Dashboard route
@app.route("/dashboard")
def dashboard():
    if 'user' not in session:
        return redirect(url_for("signin"))

    return render_template(
        "dashboard.html",
        user=session.get('user'),
        prediction_text=session.get('prediction_text'),
        details_text=session.get('details_text'),
        input_data=session.get('input_data', {}),
        assessment_score=session.get('assessment_score'),
        timestamp=session.get('timestamp')
    )


# ---------------------------------------
# Prediction route
# ---------------------------------------
@app.route("/predict")
def predict_page():
    # Check if user is logged in
    if 'user' not in session:
        return redirect(url_for("signin"))

    input_data = session.get('input_data', {})
    return render_template("predict.html", user=session.get('user'), input_data=input_data)



# ---------------------------------------
# Prediction submit route
# ---------------------------------------
@app.route("/predict", methods=["POST"])
def predict():

    input_data = {}

    if model is None:
        logger.error("Model not loaded - prediction request rejected")
        session['prediction_text'] = "Error: Machine learning model not loaded. Please contact administrator."
        session['details_text'] = ""
        session['input_data'] = input_data
        session['assessment_score'] = 0
        session['timestamp'] = datetime.now().strftime("%B %d, %Y at %I:%M %p")
        return redirect(url_for('result_page'))

    try:
        # Collect input values from form
        numeric_ranges = {
            "MMSE": (0, 30),
            "ADL": (0, 10),
            "FunctionalAssessment": (0, 10),
            "CholesterolHDL": (0, 100),
            "CholesterolLDL": (0, 200),
            "BMI": (10, 50),
            "SleepQuality": (0, 10)
        }

        for feature in feature_columns:
            value = request.form.get(feature)
            if value is None or value.strip() == '':
                return render_template(
                    "predict.html",
                    prediction_text=f"Error: Missing value for {feature}"
                )
            try:
                parsed_value = float(value)
            except ValueError:
                return render_template(
                    "predict.html",
                    prediction_text=f"Error: Invalid number format for {feature}: {value}"
                )
            if feature in numeric_ranges:
                low, high = numeric_ranges[feature]
                if not (low <= parsed_value <= high):
                    return render_template(
                        "predict.html",
                        prediction_text=f"Error: {feature} must be between {low} and {high}."
                    )
            input_data[feature] = parsed_value

        # Create DataFrame for prediction (ensure correct column order)
        input_df = pd.DataFrame([input_data], columns=feature_columns)

        # Model prediction with error handling
        try:
            prediction = int(model.predict(input_df)[0])  # Convert to native Python int
        except Exception as e:
            return render_template(
                "predict.html",
                prediction_text=f"Error: Model prediction failed - {str(e)}"
            )

        # Result message
        if prediction == 1:
            result = "✅ No sign of Alzheimer’s"
            suggestions = """
            <br><br><strong>What to do next:</strong>
            <ul>
              <li>Keep moving: walk, stretch, or do light exercise most days.</li>
              <li>Eat more vegetables, fruits, and healthy fats.</li>
              <li>Sleep 7-8 hours every night.
              <li>Keep your mind active with reading or puzzles.
            </ul>
            """
        else:
            result = "⚠️ Possible Alzheimer’s sign"
            suggestions = """
            <br><br><strong>What to do next:</strong>
            <ul>
              <li>Talk with a doctor soon.
              <li>Ask for a brain check and memory test.
              <li>Keep a healthy diet and stay active.
              <li>Write down your symptoms to show your doctor.
            </ul>
            """

        # Probability (if available)
        probability_text = ""
        try:
            if hasattr(model, "predict_proba"):
                prob = model.predict_proba(input_df)[0]
                # Get the probability for the predicted class
                probability_value = float(prob[prediction]) * 100  # Convert to native Python float
                probability_text = f" (Confidence: {probability_value:.1f}%)"
        except Exception as e:
            logger.warning(f"Could not calculate probability: {e}")
            # Continue without probability

        session['prediction_text'] = result + probability_text
        session['details_text'] = suggestions
        # Convert input_data values to ensure JSON serializable (float, not numpy types)
        session['input_data'] = {k: float(v) for k, v in input_data.items()}
        session['assessment_score'] = int(75 if prediction == 0 else 20)  # Ensure native int
        session['prediction_class'] = int(prediction)  # Ensure native int
        session['timestamp'] = datetime.now().strftime("%B %d, %Y at %I:%M %p")
        
        user = session.get('user', 'Unknown')
        logger.info(f"Prediction result for user '{user}': {result} - Assessment Score: {session['assessment_score']}%")
        return redirect(url_for('result_page'))

    except ValueError as e:
        logger.error(f"Value error: {str(e)}")
        session['prediction_text'] = f"Error: Invalid input value - {str(e)}"
        session['details_text'] = ""
        session['input_data'] = input_data
        session['assessment_score'] = 0
        session['timestamp'] = datetime.now().strftime("%B %d, %Y at %I:%M %p")
        return redirect(url_for('result_page'))
    except Exception as e:
        logger.error(f"Unexpected error: {str(e)}")
        session['prediction_text'] = f"Error: {str(e)}"
        session['details_text'] = ""
        session['input_data'] = input_data
        session['assessment_score'] = 0
        session['timestamp'] = datetime.now().strftime("%B %d, %Y at %I:%M %p")
        return redirect(url_for('result_page'))


# ---------------------------------------
# Result route
# ---------------------------------------
@app.route("/result")
def result_page():
    prediction_text = session.get('prediction_text')
    details_text = session.get('details_text')
    input_data = session.get('input_data', {})
    assessment_score = session.get('assessment_score', 0)
    timestamp = session.get('timestamp', '')
    prediction_class = session.get('prediction_class', 0)
    user = session.get('user', 'User')

    if not prediction_text:
        return redirect(url_for('predict_page'))

    return render_template(
        "result.html",
        prediction_text=prediction_text,
        details_text=details_text,
        input_data=input_data,
        assessment_score=assessment_score,
        timestamp=timestamp,
        prediction_class=prediction_class,
        user=user
    )


# ---------------------------------------
# Recommendations route
# ---------------------------------------
@app.route("/recommendations")
def recommendations():
    user = session.get('user', 'User')
    return render_template("recommendations.html", user=user)


# ---------------------------------------

# Health tips API endpoint
@app.route("/api/health-tips")
def health_tips():
    tips = {
        "exercise": "Exercise 30 minutes daily - walking, swimming, or yoga can improve cognitive health.",
        "diet": "Mediterranean diet rich in vegetables, fruits, and fish supports brain health.",
        "sleep": "Maintain consistent sleep schedule - 7-9 hours nightly is crucial for brain health.",
        "cognitive": "Engage in mentally stimulating activities like puzzles, learning languages, or reading.",
        "social": "Stay socially connected - regular social interaction supports brain health.",
        "stress": "Practice meditation or relaxation techniques to reduce stress levels."
    }
    return jsonify(tips)

# Run Flask
# ---------------------------------------
if __name__ == "__main__":
    app.run(debug=True)