# Linear Regression from Scratch

This project implements a **Linear Regression** model from scratch using Python and NumPy. It demonstrates a deep understanding of the mathematical foundations behind Gradient Descent, Feature Scaling, and L2 Regularization (Ridge).

## 🛠 Features & Highlights
* **Custom Class Implementation**: A robust `LinearRegression` class supporting training (`fit`) and evaluation (`predict`).
* **Feature Scaling (Standardization)**: Successfully handles datasets with extreme scale differences (e.g., features multiplied by 10,000).
* **L2 Regularization (Ridge)**: Incorporates a penalty term to control model complexity and prevent overfitting.
* **Benchmarked against Scikit-learn**: Proven accuracy by achieving results identical to industry-standard libraries.

## 🧪 Experiment Methodology
The `experiment.ipynb` notebook simulates real-world challenges using synthetic data:

1.  **Data Generation & Sabotage**: 1,000 samples with 10 features were generated. Specific features were "sabotaged" with extreme scaling to test the robustness of the standardization logic.
2.  **Hyperparameter Tuning**: Analyzed various Learning Rates. While **0.1** offered the fastest convergence, a more conservative **0.03** was selected for the final model to ensure maximum stability.
3.  **Model Evaluation**: Performance is measured using **Mean Absolute Error (MAE)** and visualized through **Actual vs. Predicted** plots.

## 📊 Benchmarking: Scratch vs. Scikit-learn
To validate the implementation, the custom model was compared against Scikit-learn's `LinearRegression`. 

| Metric | Custom Model (Scratch) | Scikit-learn |
| :--- | :--- | :--- |
| **Mean Absolute Error (MAE)** | **~7.19** | **~7.19** |
| **Prediction Consistency** | **Identical** | **Identical** |

The results confirm that the custom Gradient Descent and Scaling logic reach the exact same optimal solution as the industry-standard implementation.

## 📈 Key Results & Insights
* **Scaling Verification**: Sabotaged features (Std Dev ~3,200) were successfully normalized to **1.00**, enabling stable Gradient Descent.
* **Stability First**: Choosing $LR = 0.03$ prevented overshooting and ensured a smooth cost reduction curve.
* **High Fidelity**: Data points on the validation set closely follow the "Perfect Prediction" line, indicating excellent generalization.

## 📁 Project Structure
* `src/linear_regression.py`: The core model engine.
* `notebooks/experiment.ipynb`: Data preparation, tuning, and benchmarking against Scikit-learn.

## 🚀 How to Run
1.  Clone the repo: `git clone https://github.com/poomreez/linear_regression_from_scratch.git`
2.  Ensure you have `numpy`, `matplotlib`, and `sklearn` installed.
3.  Open and run `experiment.ipynb` to view the analysis or retrain the model.
