# Framing: Rawaj Pay mule detection

<!-- Write a few sentences under each heading about Rawaj Pay's mule detection.
     Keep the four headings exactly as they are. Text inside these comments does not count.
     The grader checks only that each heading has some text under it; your Track Mentor reads it. -->

## Decision
Rawaj Pay needs to detect instant transfer mule accounts effectively without overwhelming human compliance analysts with unmanageable false alarms. The threshold decision balances operational capacity against financial and regulatory risk.
<!-- What decision changes because of the model? Who acts on it, and how many cases a day
     can they handle? What does each kind of error cost? -->

## Metric
We utilize precision, recall, and expected cost metrics tailored to the business costs of false positives (analyst review time and friction for legitimate users) and false negatives (financial losses and regulatory penalties).
<!-- Which metric matches that decision, and at what threshold? Why not accuracy? -->

## Baseline
A robust baseline relies on strict data validation, anomaly checks, and heuristic rules combined with historical transfer patterns before deploying machine learning models.
<!-- What happens today without the model? What must the model beat, on the client's terms? -->

## Classical ML or LLM
For structured instant transfer telemetry and tabular risk scoring, classical machine learning models (such as XGBoost) offer high efficiency, explainability, and speed, whereas LLMs are reserved for unstructured compliance policy auditing.
<!-- Classical model, LLM, a rule, or a mix? Argue from the decision table in the session. -->
