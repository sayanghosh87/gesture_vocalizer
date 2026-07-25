from flask import Flask, request, jsonify, send_from_directory
from flask_cors import CORS
import csv
import os
import pickle
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import f1_score

app = Flask(__name__, static_folder='static')
CORS(app)

DATA_FILE = "gesture_data.csv"
MODEL_FILE = "model.pkl"
FEEDBACK_FILE = "feedback_log.csv"

# In-memory list of (true_label, predicted_label) pairs, used to compute a
# running F1 score. Loaded from FEEDBACK_FILE on startup so the score
# survives server restarts, and appended to that file on every new
# feedback entry.
feedback_log = []


def _load_feedback_log():
    """Load previously logged feedback pairs from disk, if any exist."""
    if not os.path.exists(FEEDBACK_FILE):
        return
    try:
        with open(FEEDBACK_FILE, "r", newline="") as f:
            reader = csv.reader(f)
            header = next(reader, None)
            for row in reader:
                if len(row) >= 2:
                    feedback_log.append((row[0], row[1]))
    except (OSError, csv.Error) as exc:
        print(f"Warning: could not load feedback log: {exc}")


def _compute_f1_metrics():
    """
    Compute macro and weighted F1 over every feedback pair logged so far.

    Returns:
        dict with keys macro_f1, weighted_f1, n. The F1 values are None
        when no feedback has been given yet.
    """
    if not feedback_log:
        return {"macro_f1": None, "weighted_f1": None, "n": 0}

    y_true = [t for t, _ in feedback_log]
    y_pred = [p for _, p in feedback_log]
    labels = sorted(set(y_true) | set(y_pred))

    macro = f1_score(y_true, y_pred, labels=labels, average="macro", zero_division=0)
    weighted = f1_score(y_true, y_pred, labels=labels, average="weighted", zero_division=0)

    return {
        "macro_f1": round(float(macro), 3),
        "weighted_f1": round(float(weighted), 3),
        "n": len(feedback_log),
    }


_load_feedback_log()


@app.route('/')
def index():
    return send_from_directory('static', 'index.html')

@app.route('/<path:path>')
def static_proxy(path):
    return send_from_directory('static', path)

# ====== SIGN TO TEXT ROUTES ======
@app.route('/api/record', methods=['POST'])
def record():
    data = request.json
    label = data.get('label')
    landmarks = data.get('landmarks')
    
    if not label or not landmarks:
        return jsonify({"error": "Missing label or landmarks"}), 400
        
    file_exists = os.path.isfile(DATA_FILE)
    
    with open(DATA_FILE, "a", newline="") as f:
        writer = csv.writer(f)
        if not file_exists:
            header = ["label"] + [f"lm_{i}" for i in range(84)]
            writer.writerow(header)
            
        row = [label] + landmarks
        writer.writerow(row)
        
    return jsonify({"success": True, "message": "Saved successfully!"})

@app.route('/api/train', methods=['POST'])
def train():
    if not os.path.exists(DATA_FILE):
        return jsonify({"error": "No data found. Please record first."}), 400
        
    X = []
    y = []
    
    with open(DATA_FILE, "r") as f:
        reader = csv.reader(f)
        header = next(reader, None)
        for row in reader:
            if not row: continue
            label = row[0]
            features = [float(val) for val in row[1:]]
            y.append(label)
            X.append(features)
            
    if not X:
        return jsonify({"error": "No data found."}), 400
        
    model = RandomForestClassifier(n_estimators=150)
    model.fit(X, y)
    
    with open(MODEL_FILE, "wb") as f:
        pickle.dump(model, f)
        
    return jsonify({"success": True, "message": f"Model trained on {len(X)} samples!"})

@app.route('/api/predict', methods=['POST'])
def predict():
    data = request.json
    landmarks = data.get('landmarks')
    
    if not os.path.exists(MODEL_FILE):
        return jsonify({"error": "Model not trained yet."}), 400
        
    with open(MODEL_FILE, "rb") as f:
        model = pickle.load(f)
        
    probs = model.predict_proba([landmarks])[0]
    max_prob = max(probs)
    prediction = model.classes_[probs.argmax()]
    
    if max_prob < 0.65:
        prediction = ""
        
    return jsonify({
        "prediction": prediction,
        "confidence": float(max_prob)
    })


# ====== LIVE F1 SCORE (FEEDBACK) ROUTES ======
@app.route('/api/feedback', methods=['POST'])
def feedback():
    """
    Log a user-confirmed (true_label, predicted_label) pair and return the
    running F1 score. Ground truth for a live prediction stream can only
    come from the person using it, so the browser sends this whenever they
    press "Correct" (true == predicted) or "Wrong" (true = the sign they
    actually meant, entered manually).
    """
    data = request.json or {}
    true_label = data.get('true_label')
    predicted_label = data.get('predicted_label')

    if not true_label or not predicted_label:
        return jsonify({"error": "Missing true_label or predicted_label"}), 400

    feedback_log.append((true_label, predicted_label))

    file_exists = os.path.isfile(FEEDBACK_FILE)
    try:
        with open(FEEDBACK_FILE, "a", newline="") as f:
            writer = csv.writer(f)
            if not file_exists:
                writer.writerow(["true_label", "predicted_label"])
            writer.writerow([true_label, predicted_label])
    except OSError as exc:
        print(f"Warning: could not persist feedback: {exc}")

    metrics = _compute_f1_metrics()
    return jsonify({"success": True, **metrics})


@app.route('/api/feedback-stats', methods=['GET'])
def feedback_stats():
    """Return the current running F1 score without logging new feedback."""
    return jsonify(_compute_f1_metrics())


@app.route('/api/feedback-reset', methods=['POST'])
def feedback_reset():
    """Clear all logged feedback and reset the F1 score to empty."""
    feedback_log.clear()
    try:
        if os.path.exists(FEEDBACK_FILE):
            os.remove(FEEDBACK_FILE)
    except OSError as exc:
        print(f"Warning: could not remove feedback log file: {exc}")
    return jsonify({"success": True, **_compute_f1_metrics()})


# ====== TEXT TO SIGN ROUTE ======
@app.route('/api/text-to-sign', methods=['POST'])
def text_to_sign():
    text = request.json.get('text', '').lower().strip()
    video_dir = os.path.join("static", "videos")
    
    # Load your dataset dynamically
    dataset = {}
    if os.path.exists(video_dir):
        for f in os.listdir(video_dir):
            if f.lower().endswith(".mp4"):
                name = f.lower().replace(".mp4", "").replace("_", " ").strip()
                name = " ".join(name.split())
                dataset[name] = f"static/videos/{f}"
                
    text = " ".join(text.split())
    matched_videos = []
    
    # Your exact order-preserving logic
    while text:
        found = False
        
        # Match longest phrases first
        for key in sorted(dataset.keys(), key=len, reverse=True):
            if text.startswith(key):
                matched_videos.append(dataset[key])
                text = text[len(key):].strip()
                found = True
                break
                
        # Fallback word-by-word
        if not found:
            parts = text.split(" ", 1)
            word = parts[0]
            
            if word in dataset:
                matched_videos.append(dataset[word])
            else:
                # Letter fallback
                for ch in word:
                    if ch.isalpha():
                        letter_path = f"static/videos/{ch}.mp4"
                        if os.path.exists(os.path.join("static", "videos", f"{ch}.mp4")):
                            matched_videos.append(letter_path)
                            
            text = parts[1] if len(parts) > 1 else ""
            
    return jsonify({"videos": matched_videos})


# ====== NEW: UPLOAD VIDEO ROUTE ======
@app.route('/api/upload-video', methods=['POST'])
def upload_video():
    if 'video' not in request.files:
        return jsonify({"error": "No video provided"}), 400
        
    file = request.files['video']
    if file.filename == '':
        return jsonify({"error": "No selected file"}), 400
        
    # Ensure the videos folder exists
    video_dir = os.path.join("static", "videos")
    os.makedirs(video_dir, exist_ok=True)
    
    # Save the file with the name sent by the browser
    filepath = os.path.join(video_dir, file.filename)
    file.save(filepath)
    
    return jsonify({"success": True, "message": f"Successfully added '{file.filename}' to your database!"})

if __name__ == '__main__':
    print("🚀 Server starting at http://127.0.0.1:5000")
    app.run(debug=True, port=5000)