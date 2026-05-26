from __future__ import annotations

import os
from typing import Any, Optional, Tuple

import gradio as gr
import joblib
import pandas as pd


MODEL_PATH = os.environ.get("MODEL_PATH", "model_classification.pkl")
PREP_PATH = os.environ.get("PREP_PATH", "preprocessor_classification.pkl")

_MODEL: Optional[Any] = None
_PREP: Optional[Any] = None


def _is_pipeline(model: Any) -> bool:
    return hasattr(model, "named_steps")


def _load_artifacts() -> Tuple[Any, Optional[Any]]:
    global _MODEL, _PREP
    if _MODEL is None:
        _MODEL = joblib.load(MODEL_PATH)
    # Only load the standalone preprocessor if the model is NOT a pipeline.
    # Pipeline models generally already contain preprocessing.
    if _PREP is None and (not _is_pipeline(_MODEL)) and os.path.exists(PREP_PATH):
        _PREP = joblib.load(PREP_PATH)
    return _MODEL, _PREP


def _get_classes(model: Any, probs: Any) -> list:
    if hasattr(model, "classes_"):
        return list(model.classes_)
    if hasattr(model, "named_steps") and "model" in getattr(model, "named_steps", {}):
        inner = model.named_steps["model"]
        if hasattr(inner, "classes_"):
            return list(inner.classes_)
    return [str(i) for i in range(len(probs))]


def predict_grade(
    age,
    gender,
    hours_studied,
    attendance,
    sleep_hours,
    stress_level,
    screen_time,
    previous_gpa,
    part_time_job,
    study_method,
    diet_quality,
    internet_quality,
    extracurricular,
    tutoring_sessions_per_week,
    family_income_level,
    exam_anxiety_score,
):
    model, prep = _load_artifacts()

    # Keep raw categorical strings by default. If the model is a pipeline, its
    # internal encoders expect the original category labels (e.g., "Yes"/"No").
    df = pd.DataFrame(
        [
            {
                "Age": age,
                "Gender": gender,
                "Hours_Studied": hours_studied,
                "Attendance": attendance,
                "Sleep_Hours": sleep_hours,
                "Stress_Level": stress_level,
                "Screen_Time": screen_time,
                "Previous_GPA": previous_gpa,
                "Part_Time_Job": part_time_job,
                "Study_Method": study_method,
                "Diet_Quality": diet_quality,
                "Internet_Quality": internet_quality,
                "Extracurricular": extracurricular,
                "Tutoring_Sessions_Per_Week": tutoring_sessions_per_week,
                "Family_Income_Level": family_income_level,
                "Exam_Anxiety_Score": exam_anxiety_score,
            }
        ]
    )

    # IMPORTANT:
    # If the saved model is a pipeline (e.g., imblearn Pipeline), it likely already
    # contains preprocessing (ColumnTransformer). In that case, pass the raw df.
    if _is_pipeline(model):
        features = df
    else:
        if prep is None:
            raise gr.Error(
                f"Preprocessor '{PREP_PATH}' is missing, but the model is not a pipeline."
            )
        # When using an external preprocessor, mirror training-time encoding for
        # these Yes/No fields.
        df["Part_Time_Job"] = 1 if part_time_job == "Yes" else 0
        df["Extracurricular"] = 1 if extracurricular == "Yes" else 0
        features = prep.transform(df)

    grade = model.predict(features)[0]
    probs = model.predict_proba(features)[0]
    classes = _get_classes(model, probs)
    prob_dict = {str(c): float(p) for c, p in zip(classes, probs)}
    return str(grade), prob_dict


with gr.Blocks(title="Student Performance Grade Predictor") as demo:
    gr.Markdown(
        """
        # Student Performance Grade Predictor
        Enter student details and get the predicted grade along with class probabilities.
        """
    )

    with gr.Row():
        age = gr.Number(label="Age", value=20)
        gender = gr.Dropdown(["Female", "Male"], label="Gender", value="Female")
        study_method = gr.Dropdown(
            ["Online", "Offline", "Hybrid"],
            label="Study Method",
            value="Online",
        )

    with gr.Row():
        hours_studied = gr.Number(label="Hours Studied", value=6.5)
        attendance = gr.Number(label="Attendance (%)", value=85)
        sleep_hours = gr.Number(label="Sleep Hours", value=7)

    with gr.Row():
        stress_level = gr.Number(label="Stress Level", value=3.5)
        screen_time = gr.Number(label="Screen Time (hrs)", value=2.5)
        previous_gpa = gr.Number(label="Previous GPA", value=3.2)

    with gr.Row():
        part_time_job = gr.Dropdown(["No", "Yes"], label="Part-Time Job", value="No")
        extracurricular = gr.Dropdown(["No", "Yes"], label="Extracurricular", value="Yes")
        tutoring_sessions_per_week = gr.Number(label="Tutoring Sessions/Week", value=2)

    with gr.Row():
        diet_quality = gr.Dropdown(
            ["Poor", "Average", "Good"],
            label="Diet Quality",
            value="Average",
        )
        internet_quality = gr.Dropdown(
            ["Poor", "Average", "Good", "Excellent"],
            label="Internet Quality",
            value="Good",
        )
        family_income_level = gr.Dropdown(
            ["Low", "Middle", "High"],
            label="Family Income",
            value="Middle",
        )

    exam_anxiety_score = gr.Number(label="Exam Anxiety Score", value=4.0)

    predict_btn = gr.Button("Predict Grade")
    predicted_grade = gr.Textbox(label="Predicted Grade")
    grade_probabilities = gr.JSON(label="Grade Probabilities")

    predict_btn.click(
        fn=predict_grade,
        inputs=[
            age,
            gender,
            hours_studied,
            attendance,
            sleep_hours,
            stress_level,
            screen_time,
            previous_gpa,
            part_time_job,
            study_method,
            diet_quality,
            internet_quality,
            extracurricular,
            tutoring_sessions_per_week,
            family_income_level,
            exam_anxiety_score,
        ],
        outputs=[predicted_grade, grade_probabilities],
    )


if __name__ == "__main__":
    # For Hugging Face Spaces: leave launch configuration to Gradio/Spaces.
    # Spaces typically sets `GRADIO_SERVER_NAME=0.0.0.0` and `GRADIO_SERVER_PORT`.
    # Locally, you can override with env vars if needed.
    demo.launch(show_error=True)