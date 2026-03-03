# PMS Mood Compass

PMS Mood Compass is a full-stack portfolio project that helps users track cycle-related symptoms, lifestyle patterns, and day-to-day wellbeing. It combines a clean web interface with a Python machine learning service to estimate symptom load, surface likely pattern drivers, and forecast the next 7 days.

## Overview
Many cycle-tracking tools focus on logging alone. PMS Mood Compass is designed to go one step further: it turns daily entries into interpretable pattern insights.

Users can:
- log symptoms and lifestyle factors day by day
- record recent period start dates to estimate cycle timing
- generate a daily symptom-load score
- compare logged symptom load against model expectations
- see which non-symptom factors are most associated with symptom changes
- view a short-term 7-day outlook based on historical patterns

This project is informational only and is not medical advice.

## What The App Measures
The app works with two related ideas:

### 1. Logged Symptom Score
This is the score calculated from the symptom sliders the user enters for a given day.

It is a **0 to 5 symptom-load score**:
- lower score = lighter symptoms
- higher score = heavier symptoms

A score like `1.83 / 5` means relatively lighter symptom load.
A score like `5.00 / 5` means very high symptom load.

### 2. Model Expectation
The machine learning model estimates what symptom load the day was likely to have based on non-symptom context such as:
- cycle timing
- sleep
- stress
- caffeine
- workout
- previous-day history
- optional sensor inputs

This makes it possible to compare:
- what was logged
- what the model expected
- where the model underestimated or overestimated the day

## Scoring Logic
The logged symptom score is built from a weighted symptom formula.

High-level interpretation:
- `Good`: under `2.0`
- `Moderate`: `2.0` to under `3.5`
- `Poor`: `3.5` and above

Mood is handled differently from the other symptom sliders:
- `0 = very low mood`
- `5 = very good mood`

Because a better mood should reduce symptom burden, the score uses an inverted mood contribution internally.

## Machine Learning Approach
The backend is built as a non-linear regression service in FastAPI.

At a high level, the model:
- learns from time-ordered daily logs
- uses cycle-aware and lifestyle-aware features
- predicts symptom load rather than raw symptom values
- compares its performance against a simple baseline
- returns both global and local interpretability outputs

The project supports an AutoML-first workflow with strong tree-based fallbacks, so the backend can select an appropriate non-linear regressor while keeping inference fast and interpretable.

## Why This Project Is Useful
PMS Mood Compass is useful as a portfolio project because it brings together:
- product thinking
- frontend UX for health-style tracking
- API design with FastAPI
- ML problem framing for user-generated daily data
- feature engineering for cycle-aware forecasting
- interpretable model outputs instead of black-box predictions only

It is also a practical example of how machine learning can be applied to everyday self-tracking data in a way that stays understandable to end users.

## Core Product Areas
### Setup
- add, edit, and remove period start dates
- estimate cycle length from recent entries
- calculate cycle regularity
- view cycle and nutrition tip cards

### Daily Log
- log mood, cramps, fatigue, headache, acne, sleep, stress, caffeine, workout, and optional sensor values
- save, edit, delete, import, and export entries
- keep data available within the current browser session

### Insights
- train or refresh the model on current session data
- compare model error against a simple baseline
- see major pattern signals
- explain a selected logged day
- forecast the next 7 days

## Technical Stack
- Frontend: Next.js (App Router) + TypeScript
- Backend: FastAPI + Python
- ML: AutoML-assisted non-linear regression with boosted-tree fallbacks
- Data handling: Pydantic, pandas, NumPy
- Persistence: session-based frontend storage + backend model artifacts

## Repository Structure
```text
/
  frontend/      Next.js application
  backend/       FastAPI ML service
  sample_data.csv
  README.md
  .gitignore
```

## Notes
- This project is informational only and not intended for diagnosis or treatment.
- Model outputs represent learned associations from logged data, not causation.
- The quality of the insights depends on the consistency and volume of user-entered data.

## Author
Created by **Tr Shwetha**.
