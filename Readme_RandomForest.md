### ASSIGNMENT 4 - HYPER PARAMETER TUNING & CROSS VALIDATION

### Project : Drug Classification with Random Forest

***Project Description:*** The project is to classify and predict the drugs type that might be suitable for the patient.

***Goal***The goal of the project is to train the ***Random Forest Machine learning model*** to classify and predict the drugs type that might be suitable for the patient based on the features/attributes.


In this assignment,I applied a supervised machine learning pipeline to classify patients into appropriate drug categories based on medical attributes such as Age, Sex, Blood Pressure, Cholesterol, and Na_to_K ratio.

-The chosen model was a Random Forest, trained with cross-validation and hyperparameter tuning (GridSearchCV) to ensure robust performance and avoid overfitting.

***Random Forest*** was chosen for its ensemble power and generalization strength, combining multiple decision trees to reduce overfitting and improve accuracy.


***CROSS-VALIDATION***

A systematic evaluation technique that repeatedly splits the dataset into training and testing folds to assess model generalization. This ensured that the Random Forest model was not just memorizing patterns but learning to predict reliably.

***HYPER PARAMETER TUNING***

Hyperparameter Tuning = Optimizing the Forest’s Structure and Behavior

***Key Steps:***

- Data Preparation → Features: Age, Sex, BP, Cholesterol, Na_to_K  Target: Drug
- Train/Test Split → 80% training, 20% testing
- GridSearchCV explored:
- n_estimators: 50, 100, 200
- max_depth: 3, 5, 10, None
- min_samples_split: 2, 5, 10
- criterion: gini, entropy

***Outcome Analysis of Best CV score & Test accuracy***

- Best CV Score: 99.4%
- Test Accuracy: 98.0%
- Model is highly accurate and generalizable.
- Na_to_K emerged as the most influential feature, followed by BP, Age, and Cholesterol.
- Feature importance was visualized using a bar chart (Seaborn), highlighting the top predictors across all trees.
- Random Forest’s ensemble approach provided stability, reduced variance, and strong predictive power.

***conclusion***

The Drug Classification with Random Forest model showed that patient attributes like age, sex, blood pressure, cholesterol, and sodium‑to‑potassium ratio can be used to reliably predict suitable drug types.
With hyperparameter tuning and cross‑validation, the model achieved strong accuracy and balanced classification by combining multiple decision trees, reducing overfitting and improving robustness. Feature importance analysis highlighted key patient attributes, making this ensemble approach both reliable and interpretable for drug prediction tasks.

***Random Forest*** modeling delivered accurate, robust, and interpretable drug classification by combining multiple decision trees.

-------------------------------

***Assignment Files***

- [ML_Assignment_4_supervised_random_forest_tree_hyperparameter_tuning_drug200.py](ML_Assignment_4_supervised_random_forest_tree_hyperparameter_tuning_drug200.py)

- [ML_Assignment_4_supervised_random_forest_tree_hyperparameter_tuning_drug200.ipynb](ML_Assignment_4_supervised_random_forest_tree_hyperparameter_tuning_drug200.ipynb)



***Data***

- [drug200.csv](drug200.csv)


