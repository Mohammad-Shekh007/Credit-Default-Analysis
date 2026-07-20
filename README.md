Credit Default Prediction

A logistic regression model that estimates the probability that a loan applicant will default, built on the German Credit dataset. The focus is on an interpretable model — one where every retained variable has a clear, statistically defensible reason for being there — rather than a black-box predictor.

Scope

This project is exploratory and educational in nature: it demonstrates a full, defensible workflow for building an interpretable logistic regression model, from raw data to a final variable set with clean statistical inference. It is not a production credit-scoring system — the model has not been calibrated, tested for fairness/bias across demographic groups, or validated on external data, all of which would be required before any real-world lending use.

Dataset


Source: German Credit dataset (Statlog) — 1,000 loan applicants, 20 predictor variables, and a binary target (default: good credit vs. bad credit).
Data quality: no missing values; no cleaning required beyond standard preprocessing.
Transformations: amount (loan amount) and age were right-skewed, so both were log-transformed before modeling. Skewed continuous predictors can distort a logistic regression's assumption of a linear relationship with the log-odds of the outcome, and log-transforming brings them closer to that assumption.
Categorical variables were one-hot encoded with reference-level (drop-first) encoding.


Methodology


Preprocessing — log transformations, one-hot encoding, and a 60/20/20 train/validation/test split.
Variable selection — best subset — rather than a stepwise (forward/backward) search, all 20 original variables were passed through best-subset selection (via the abess algorithm), with categorical variables constrained to be selected as whole units rather than individual dummy levels. The number of variables to keep was chosen by cross-validation.
Refinement — the selected variables were refit using standard (unpenalized) logistic regression to obtain proper p-values and odds ratios. Any variable that turned out not to be statistically significant (p > 0.05) was removed and the model refit; this was repeated until every remaining variable was significant, including one case (telephone) that only lost significance after other correlated variables were removed — a normal consequence of shared variance, not an error.
Final model — 6 of the original 20 variables remained: loan amount (log), age (log), checking account balance, credit history, loan purpose, and installment plan.


Results

<img width="800" height="248" alt="image" src="https://github.com/user-attachments/assets/a5c7118d-a9e1-4341-8501-0c812a6f034b" />

All 6 final variables are individually significant at p < 0.05, and variance inflation factors show no meaningful multicollinearity.


Limitations


Results come from a single random train/validation/test split; a different split could shift specific p-values or performance numbers somewhat.
Variable selection and statistical inference were performed in sequence on the same data, which is standard practice but means p-values should be read as strong evidence of relevance rather than exact, one-shot values.
With 1,000 observations, there's a practical limit to how finely model complexity can be tuned or validated.
