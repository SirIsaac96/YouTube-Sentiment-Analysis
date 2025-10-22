
# Step 1: Libraries
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report
from imblearn.over_sampling import SMOTE
import mlflow
import mlflow.sklearn
import optuna
from lightgbm import LGBMClassifier
import matplotlib.pyplot as plt
import mlflow


# Step 2: Set up the MLflow tracking server and create an experiment
mlflow.set_tracking_uri("http://ec2-34-224-86-44.compute-1.amazonaws.com:5000/")
mlflow.set_experiment("Exp 6 - LightGBM HP Tuning")


# Step 3: Load dataset
df = pd.read_csv('data/processed.csv').dropna()


# Preprocessing
# Step 1: Remap the class labels from [-1, 0, 1] to [2, 0, 1]
df['category'] = df['category'].map({-1: 2, 0: 0, 1: 1})


# Step 2: Remove rows where the target labels (category) are NaN
df = df.dropna(subset=['category'])


# Step 3: TF-IDF vectorizer setup
ngram_range = (1, 3)  # Trigram
max_features = 1000  # Set max_features to 1000
vectorizer = TfidfVectorizer(ngram_range=ngram_range, max_features=max_features)
X = vectorizer.fit_transform(df['clean_comment'])
y = df['category']


# Step 4: Apply SMOTE to handle class imbalance
smote = SMOTE(random_state=42)
X_resampled, y_resampled = smote.fit_resample(X, y)


# Step 5: Train-test split
X_train, X_test, y_train, y_test = train_test_split(X_resampled, y_resampled, test_size=0.2, random_state=42, stratify=y_resampled)


# Function to log results in MLflow
def log_mlflow(model_name, model, X_train, X_test, y_train, y_test, params, trial_number):
    with mlflow.start_run():
        # Log model type and trial number
        mlflow.set_tag("mlflow.runName", f"Trial_{trial_number}_{model_name}_SMOTE_TFIDF_Trigrams")
        mlflow.set_tag("experiment_type", "algorithm_comparison")

        # Log algorithm name as a parameter
        mlflow.log_param("algo_name", model_name)

        # Log hyperparameters
        for key, value in params.items():
            mlflow.log_param(key, value)

        # Train model
        model.fit(X_train, y_train)
        y_pred = model.predict(X_test)

        # Log accuracy
        accuracy = accuracy_score(y_test, y_pred)
        mlflow.log_metric("accuracy", accuracy)

        # Log classification report
        classification_rep = classification_report(y_test, y_pred, output_dict=True)
        for label, metrics in classification_rep.items():
            if isinstance(metrics, dict):
                for metric, value in metrics.items():
                    mlflow.log_metric(f"{label}_{metric}", value)

        # Log the model
        mlflow.sklearn.log_model(model, f"{model_name}_model")

        return accuracy


# Step 6: Optuna objective function for LightGBM
def objective_lightgbm(trial):
    # Hyperparameter space to explore
    n_estimators = trial.suggest_int('n_estimators', 100, 1000)
    learning_rate = trial.suggest_float('learning_rate', 1e-4, 1e-1, log=True)
    max_depth = trial.suggest_int('max_depth', 3, 15)
    num_leaves = trial.suggest_int('num_leaves', 20, 150)
    min_child_samples = trial.suggest_int('min_child_samples', 10, 100)
    colsample_bytree = trial.suggest_float('colsample_bytree', 0.5, 1.0)
    subsample = trial.suggest_float('subsample', 0.5, 1.0)
    reg_alpha = trial.suggest_float('reg_alpha', 1e-4, 10.0, log=True)  # L1 regularization
    reg_lambda = trial.suggest_float('reg_lambda', 1e-4, 10.0, log=True)  # L2 regularization

    # Log trial parameters
    params = {
        'n_estimators': n_estimators,
        'learning_rate': learning_rate,
        'max_depth': max_depth,
        'num_leaves': num_leaves,
        'min_child_samples': min_child_samples,
        'colsample_bytree': colsample_bytree,
        'subsample': subsample,
        'reg_alpha': reg_alpha,
        'reg_lambda': reg_lambda
    }

    # Create LightGBM model
    model = LGBMClassifier(n_estimators=n_estimators,
                           learning_rate=learning_rate,
                           max_depth=max_depth,
                           num_leaves=num_leaves,
                           min_child_samples=min_child_samples,
                           colsample_bytree=colsample_bytree,
                           subsample=subsample,
                           reg_alpha=reg_alpha,
                           reg_lambda=reg_lambda,
                           random_state=42)

    # Log each trial as a separate run in MLflow
    accuracy = log_mlflow("LightGBM", model, X_train, X_test, y_train, y_test, params, trial.number)

    return accuracy


# Step 7: Run Optuna for LightGBM, log the best model, and plot the importance of each parameter
def run_optuna_experiment():
    study = optuna.create_study(direction="maximize")
    study.optimize(objective_lightgbm, n_trials=10)  # Increased to 100 trials

    # Get the best parameters
    best_params = study.best_params
    best_model = LGBMClassifier(n_estimators=best_params['n_estimators'],
                                learning_rate=best_params['learning_rate'],
                                max_depth=best_params['max_depth'],
                                num_leaves=best_params['num_leaves'],
                                min_child_samples=best_params['min_child_samples'],
                                colsample_bytree=best_params['colsample_bytree'],
                                subsample=best_params['subsample'],
                                reg_alpha=best_params['reg_alpha'],
                                reg_lambda=best_params['reg_lambda'],
                                random_state=42)

    # Log the best model with MLflow and print the classification report
    log_mlflow("LightGBM", best_model, X_train, X_test, y_train, y_test, best_params, "Best")

    # Plot parameter importance
    optuna.visualization.plot_param_importances(study).show()

    # Plot optimization history
    optuna.visualization.plot_optimization_history(study).show()


# Run the experiment for LightGBM
run_optuna_experiment()
