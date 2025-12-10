"""Drug200 Random Forest Pipeline"""

print("-----ASSIGNMENT 4 - HYPER PARAMETER TUNING & CROSS VALIDATION-----")
print("-----MODEL - RANDOM FOREST -----")

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.preprocessing import OrdinalEncoder
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, accuracy_score
from sklearn.tree import plot_tree

# 1. Load dataset
df = pd.read_csv("drug200.csv")

# 2. Data Preprocessing
X = df.drop("Drug", axis=1)
y = df["Drug"]

categorical_cols = ["Sex", "BP", "Cholesterol"]
encoder = OrdinalEncoder()
X[categorical_cols] = encoder.fit_transform(X[categorical_cols])

# 3. Train/Test split
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

# 4. Define Random Forest + Hyperparameter Grid
param_grid = {
    "n_estimators": [25, 50, 75],       # number of trees
    "criterion": ["gini", "entropy"],     # split criteria
    "max_depth": [3, 5, 10, None],        # depth of trees
    "min_samples_split": [2, 5, 10]       # minimum samples to split
}

grid = GridSearchCV(
    RandomForestClassifier(random_state=42),
    param_grid,
    cv=5,
    scoring="accuracy"
)

# 5. Fit model
grid.fit(X_train, y_train)



# 6. Evaluate the model

best_model = grid.best_estimator_
y_pred = best_model.predict(X_test)

# Check how many trees were built
print("Number of trees in the forest:", len(best_model.estimators_))

# Check on the Best parameters,CV Score and Test Accuracy

print("\nBest Parameters:\n", grid.best_params_)
print("\nBest CV Score:\n", grid.best_score_)
print("\nTest Accuracy:\n", accuracy_score(y_test, y_pred))
print("==============================================")
print("\nClassification Report:\n", classification_report(y_test, y_pred))
print("============================")



# """ 7.Visualisation"""
#
# Feature Importance Visualization"""

importances = best_model.feature_importances_


# Create a DataFrame for feature importances
feat_imp = pd.DataFrame({
    "Feature": X.columns,
    "Importance": importances
})

#  To display numeric score- weightage each feature carried in the overall prediction process.

print("\nFeature Importance Scores:")
print(feat_imp.sort_values(by="Importance", ascending=False))



# Plot stacked bar chart

# Create a crosstab of Drug vs BP and Cholesterol

ct = pd.crosstab(df["Drug"], [df["BP"], df["Cholesterol"]])

# stacked bar chart : BP & Cholesterol across Drug Types

ct.plot(kind="bar", stacked=True, figsize=(10,6), colormap="Set2")

plt.title("Distribution of BP & Cholesterol across Drug Types")
plt.xlabel("Drug")
plt.ylabel("Count")
plt.legend(title="BP & Cholesterol", bbox_to_anchor=(1.05, 1), loc="upper left")
plt.grid(axis="y", linestyle="--", alpha=0.7)
plt.tight_layout()
plt.show()


# Scatterplot: Age vs Gender colored by Drug

# Convert Sex to numeric and add jitter
df["Sex_jitter"] = df["Sex"].map({"M": 0, "F": 1}) + np.random.uniform(-0.1, 0.1, size=len(df))

# Scatter Plot with jittered gender

plt.figure(figsize=(8,6))
sns.scatterplot(x="Age", y="Sex_jitter", hue="BP", data=df, palette="Set2")
plt.title("Age vs Gender by BP ")
plt.xlabel("Age")
plt.ylabel("Gender")
plt.yticks([0, 1], ["M", "F"])  # restore labels
plt.grid(True, linestyle="--", alpha=0.7)
plt.show()



# Bar Plot with Seaborn to display feature importance in prediction

plt.figure(figsize=(8,6))
sns.barplot(data=feat_imp, x="Feature", y="Importance", hue="Feature", palette="Set2", dodge=False)
plt.title("Feature Importance in Random Forest")
plt.ylabel("Importance")
plt.xlabel("Features")
plt.show()


# Tree Plot :Visualize one tree from the forest

plt.figure(figsize=(20,10))
plot_tree(
    best_model.estimators_[0],           # pick the first tree
    feature_names=X.columns,
    class_names=best_model.classes_,
    filled=True,
    rounded=True,
    fontsize=10
)
plt.title("Random Forest - Example Tree")
plt.show()