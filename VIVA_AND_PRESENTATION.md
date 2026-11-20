# Viva questions and short answers

1. **What is Machine Learning?** It is a way for a computer to learn patterns from example data and use them to make estimates or decisions.
2. **What is supervised learning?** It learns from examples that include both input features and a known target value.
3. **What is regression?** Regression predicts a numeric value. Here the value is simulated remaining kilometres until service.
4. **What is a Random Forest?** It trains many decision trees on varied samples/features and averages their numeric predictions.
5. **What is a Decision Tree?** It predicts by applying a sequence of feature-based splits, like a flowchart.
6. **Why choose these algorithms?** They handle numeric and one-hot encoded inputs, learn nonlinear patterns, need little scaling, and are beginner-friendly. Extra Trees was added to meet the report template's additional candidate-model requirement.
7. **What is a dataset?** A structured collection of examples. Our CSV has 8,000 reproducibly generated synthetic passenger-car examples.
8. **What is preprocessing?** Converting raw fields into consistent model inputs. We impute numeric/categorical values and one-hot encode categories inside a Scikit-learn pipeline.
9. **What is train-validation-test splitting?** Train fits models, validation selects a model, and a held-out test set reports final performance.
10. **What is MAE?** Mean Absolute Error is the average absolute difference between a prediction and its target, measured here in kilometres.
11. **What is RMSE?** Root Mean Squared Error is the square root of average squared error. Larger errors count more strongly than in MAE.
12. **What is R²?** It compares model error with a mean-target baseline. One is a perfect fit, zero is similar to predicting the mean, and negative is worse than that baseline.
13. **What is overfitting?** A model memorizes details of its training examples and does worse on unseen examples.
14. **Why Streamlit?** It lets a Python team build an interactive demonstration without a separate JavaScript frontend.
15. **How does this app estimate service?** It builds date, distance and usage features, loads the saved regression pipeline, then caps the model estimate at the earlier configured distance or time limit from the owner's manual.
16. **What are the limitations?** The training labels are synthetic assumptions, not observed car-service outcomes. The test score only measures agreement with those assumptions. The tool is not a mechanical diagnosis.
17. **What can improve it?** Obtain consented, licensed, car-only maintenance records with repeated odometer/date entries; evaluate by vehicle/time split; calibrate across models and climates; and validate recommendations with mechanics and owner manuals.

## Three-member presentation script

- **Member 1 — data and problem (about 1 minute):** “Our project estimates when a passenger car may reach its next scheduled service interval. We reviewed candidate sources, but the accessible data did not provide a verified continuous remaining-kilometre target with clear reuse terms. We therefore generate 8,000 reproducible synthetic rows and document each assumption. The generated data is for a demonstration and is not real vehicle history.”
- **Member 2 — algorithms and results (about 1 minute):** “We compare a median baseline with Decision Tree, Extra Trees, Random Forest and a tuned Random Forest regressor. A preprocessing pipeline imputes missing values and one-hot encodes categories. We use separate training, validation and test partitions. The selected tuned Random Forest achieved the recorded metrics on held-out synthetic data; those scores measure the simulated rule, not real-world accuracy.”
- **Member 3 — application and limits (about 1 minute):** “In the Streamlit app, the driver enters readings and the interval from the vehicle's own manual. The model can suggest an earlier service window, while a rule cap prevents it from extending beyond the manual interval. The dashboard shows metrics and session-only history. This is a student prototype, not a mechanical diagnosis.”
