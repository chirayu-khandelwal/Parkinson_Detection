<h3 align="center">Parkinson's Disease Motor UPDRS Prediction</h3>

  <p align="center">
    Predicting motor UPDRS scores from biomedical voice measurements using machine learning.
    <br />
     <a href="https://github.com/chirayu-khandelwal/parkinson_detection">github.com/chirayu-khandelwal/parkinson_detection</a>
  </p>
</div>

## Table of Contents

<details>
  <summary>Table of Contents</summary>
  <ol>
    <li>
      <a href="#about-the-project">About The Project</a>
      <ul>
        <li><a href="#key-features">Key Features</a></li>
      </ul>
    </li>
    <li><a href="#architecture">Architecture</a></li>
    <li>
      <a href="#getting-started">Getting Started</a>
      <ul>
        <li><a href="#prerequisites">Prerequisites</a></li>
        <li><a href="#installation">Installation</a></li>
      </ul>
    </li>
    <li><a href="#acknowledgments">Acknowledgments</a></li>
  </ol>
</details>

## About The Project

This project focuses on predicting the motor UPDRS (Unified Parkinson's Disease Rating Scale) score of patients with Parkinson's disease using machine learning techniques. The dataset used contains biomedical voice measurements from individuals with early-stage Parkinson's disease. The goal is to accurately estimate the motor UPDRS score based on these voice features, aiding in remote symptom progression monitoring.

### Key Features

- **Data Acquisition:** Utilizes the UC Irvine Machine Learning Repository (UCIMLRepo) to fetch the Parkinsons Telemonitoring dataset.
- **Data Preprocessing:** Includes data scaling using MinMaxScaler to ensure optimal performance of regression models.
- **Model Training:** Implements several regression models, including Linear Regression, Ridge Regression, Lasso Regression, K-Nearest Neighbors Regressor, Decision Tree Regressor, Random Forest Regressor, and Support Vector Regressor (SVR).
- **Hyperparameter Tuning:** Employs GridSearchCV to fine-tune model parameters for enhanced accuracy.
- **Performance Evaluation:** Provides comprehensive evaluation metrics such as Mean Squared Error (MSE), Root Mean Squared Error (RMSE), Mean Absolute Error (MAE), and R-squared (R²) for each model.
- **Feature Importance:** Extracts and visualizes feature importances from tree-based models to identify key predictors of motor UPDRS.
- **Visualization:** Includes various plots such as distribution plots, box plots, correlation heatmaps, and scatter plots to explore the dataset and model performance.


The project is structured around a Jupyter Notebook (`Parkinson_Detection_4_0.ipynb`) that performs data loading, preprocessing, model training, and evaluation.

- **Data Source:** UCIMLRepo (Parkinsons Telemonitoring dataset)
- **Programming Language:** Python
- **Libraries:**
  - pandas
  - numpy
  - seaborn
  - matplotlib
  - scikit-learn (sklearn)
  - ucimlrepo

The notebook follows a typical machine learning workflow:

1.  **Data Loading:** Fetches the dataset using `ucimlrepo`.
2.  **Data Exploration:** Explores the dataset using pandas and visualizes data distributions and correlations using seaborn and matplotlib.
3.  **Data Preprocessing:** Scales the features using `MinMaxScaler` from scikit-learn.
4.  **Model Training and Evaluation:** Trains and evaluates various regression models using scikit-learn, with hyperparameter tuning using `GridSearchCV`.
5.  **Feature Importance:** Extracts and visualizes feature importances from the trained Random Forest model.

## Getting Started

To run this project, you need to have Python installed along with the following libraries.

### Prerequisites

- Python (>=3.6)
- pip

Install the required libraries using pip:
```sh
pip install pandas numpy seaborn matplotlib scikit-learn ucimlrepo
```

### Installation

1. Clone the repository:
   ```sh
   git clone https://github.com/chirayu-khandelwal/parkinson_detection.git
   ```
2. Navigate to the project directory:
   ```sh
   cd parkinson_detection
   ```
3. Open and run the Jupyter Notebook:
   ```sh
   jupyter notebook Parkinson_Detection_4_0.ipynb
   ```
## Results

The following table summarizes the performance metrics for each trained regression model:

```
--- Performance Metrics Summary --- (Lower is better for MAE/MSE/RMSE, Higher is better for R2) ---
```

| Metric                        | Linear Regression | Ridge Regression | Lasso Regression | KNN Regressor (k=5) | Decision Tree Regressor | Random Forest Regressor | SVR (RBF Kernel) | Random Forest Regressor (Tuned) | SVR (Tuned) |
| :---------------------------  | :---------------- | :--------------- | :--------------- | :------------------ | :------------------------ | :---------------------- | :--------------- | :------------------------------ | :---------- |
| R2-Score                      | *None*            | *None*           | *None*           | *None*              | *None*                    | *None*                  | *None*           | *None*                          | *None*      |
| MAE(Mean absolute error)      | *None*            | *None*           | *None*           | *None*              | *None*                    | *None*                  | *None*           | *None*                          | *None*      |
| MSE(Mean squared error)       | *None*            | *None*           | *None*           | *None*              | *None*                    | *None*                  | *None*           | *None*                          | *None*      |
| RMSE (Root Mean Square Error) | *None*            | *None*           | *None*           | *None*              | *None*                    | *None*                  | *None*           | *None*                          | *None*      |


## Acknowledgments

- This README was created using [gitreadme.dev](https://gitreadme.dev) — an AI tool that looks at your entire codebase to instantly generate high-quality README files.
- UC Irvine Machine Learning Repository for providing the Parkinsons Telemonitoring dataset.

**Note:** The `None` values in the table indicate that the results were not properly captured in the original notebook's output.  In a complete run, these would be replaced with the actual metric values.  Based on the notebook output, the Random Forest Regressor and Decision Tree Regressor appear to perform the best.


| Feature       | Importance |
| :------------ | :--------- |
| age           | 0.330499   |
| DFA           | 0.099157   |
| HNR           | 0.051759   |
| Jitter(Abs)   | 0.049840   |
| RPDE          | 0.048568   |
| PPE           | 0.047893   |
| test\_time    | 0.047070   |
| NHR           | 0.036099   |
| Shimmer:APQ11 | 0.032347   |
| sex           | 0.031273   |
| Shimmer:DDA   | 0.027338   |
| Jitter:PPQ5   | 0.026300   |
| Shimmer:APQ3  | 0.026120   |
| Shimmer:APQ5  | 0.025629   |
| Jitter(%)     | 0.024862   |

## Acknowledgments

-   This project utilizes the Parkinsons Telemonitoring dataset from the UCI Machine Learning Repository.
-   This README was created using [gitreadme.dev](https://gitreadme.dev) — an AI tool that looks at your entire codebase to instantly generate high-quality README files.
