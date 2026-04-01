# Linear Regression from Scratch with Regularization

This project implements a **Linear Regression** model from scratch using Python and NumPy, without relying on high-level machine learning libraries. It demonstrates a deep understanding of the mathematical foundations of Gradient Descent, Feature Scaling, and L2 Regularization (Ridge).

## 🛠 Features & Highlights
* **Custom Class Implementation**: A robust `LinearRegression` class supporting training (`fit`) and evaluation (`predict`).
* **Feature Scaling (Standardization)**: Handles datasets with extreme scale differences, bringing them into a suitable range for efficient training.
* **L2 Regularization (Ridge)**: Incorporates a penalty term to control model complexity and prevent overfitting.
* **Comprehensive Experimentation**: A detailed exploration of hyperparameters through learning curve analysis and performance comparisons.

## 🧪 Experiment Methodology
The `experiment.ipynb` notebook simulates real-world challenges using synthetic data:

1.  **Data Generation & Sabotage**: Generates 1,000 samples with 10 features using `make_regression`. To test the robustness of the scaling system, specific features are "sabotaged" by multiplying them by 10,000 or 0.001.
2.  **Hyperparameter Tuning (Learning Rate)**: Tests various $LR$ values (from $0.001$ to $1.0$) to identify the optimal rate for fast and stable convergence.
3.  **Model Evaluation**: Performance is measured using **Mean Absolute Error (MAE)** and visualized through **Actual vs. Predicted** plots for both training and validation sets.

## 📈 Key Results & Insights
Based on the experiments conducted in the notebook:

* **Scaling Verification**: Features with a standard deviation as high as 3,200 were successfully normalized to 1.00, ensuring stable Gradient Descent.
* **Optimal Convergence**: A Learning Rate of **0.1** was found to be ideal, reaching a cost plateau in fewer than 100 iterations.
* **Accuracy**: The model achieves high predictive accuracy, with low MAE and data points closely following the "Perfect Prediction" line in visualizations.

## 📁 Project Structure
* `linear_regression.py`: The core model engine, designed for cleanliness and reusability.
* `experiment.ipynb`: The primary lab notebook for data preparation, hyperparameter tuning, and visualization.

## 🚀 How to Run
1.  Clone the repo: `git clone https://github.com/poomreez/linear_regression_from_scratch.git`
2.  Ensure you have `numpy`, `matplotlib`, and `sklearn` installed.
3.  Open and run `experiment.ipynb` to view the analysis or retrain the model.
