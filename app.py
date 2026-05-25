
from flask import Flask, request, jsonify
import os
import joblib, pandas as pd

app = Flask(__name__)

MODEL_PATH = "model_classification.pkl"
PREP_PATH = "preprocessor_classification.pkl"

clf_model = joblib.load(MODEL_PATH)
clf_prep = joblib.load(PREP_PATH) if os.path.exists(PREP_PATH) else None


def _is_pipeline(model):
    return hasattr(model, "named_steps")


def _get_classes(model, probs):
    if hasattr(model, "named_steps") and "model" in model.named_steps:
        inner = model.named_steps["model"]
        if hasattr(inner, "classes_"):
            return inner.classes_
    if hasattr(model, "classes_"):
        return model.classes_
    return [str(i) for i in range(len(probs))]

@app.route("/predict", methods=["POST"])
def predict():
    data = request.get_json()
    df = pd.DataFrame([data])

    if _is_pipeline(clf_model):
        features = df
    else:
        features = clf_prep.transform(df) if clf_prep is not None else df

    grade = clf_model.predict(features)[0]
    probs = clf_model.predict_proba(features)[0]
    classes = _get_classes(clf_model, probs)

    return jsonify({
        "predicted_grade": grade,
        "grade_probabilities": {c: float(p) for c, p in zip(classes, probs)}
    })

if __name__ == "__main__":
    app.run(debug=True)
