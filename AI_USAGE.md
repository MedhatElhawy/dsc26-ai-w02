# AI usage

<!-- Replace every comment below with your own words. Short is fine. The grader checks only
     that this file exists and has been changed from this template; your Track Mentor reads it. -->


## Tools I used, and for which functions

<!-- e.g. "Claude for a first draft of audit(); Copilot autocomplete in metrics.py" -->
- Gemini AI: Used to structure the audit pipeline in `audit.py`, handle pandas/sklearn edge cases for ROC AUC leak checks, and draft the initial functions in `metrics.py`.

## One place where the AI was wrong, incomplete, or not what the spec said

<!-- What did it produce, and how did you notice: a test, the docs, or your own reasoning? -->
- **Example:** In `best_threshold`, the AI's initial code selected the *lowest* threshold when there was a tie in expected cost.
- **How I noticed:** By reviewing the task specifications and test requirements, which explicitly state that on a cost tie, the code must return the *highest* threshold (fewer flags for the exact same cost). I corrected the condition to ensure `t > best_t` updates the selection on ties.

## The function I wrote without AI, and why

<!-- Which one, and what you learned by writing it yourself. -->
- **Function:** Input validation (`_validate_inputs`) and core confusion matrix logic (`confusion_at`).
- **Why:** Writing the strict input validation checks manually (raising `ValueError` for empty inputs, length mismatches, NaNs, and invalid labels not equal to 0 or 1) alongside the threshold comparison (`s_arr >= threshold`) gave me full control over edge cases and a deeper understanding of how the evaluation grader checks binary classification boundaries.
