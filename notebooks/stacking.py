
# Libraries
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.ensemble import StackingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from lightgbm import LGBMClassifier
from sklearn.metrics import classification_report


# Load the dataset
dataset = pd.read_csv('data/processed.csv')

# Drop rows with NaN values in 'clean_comment'
cleaned_dataset = dataset.dropna()

# Separate features and target
X_cleaned = cleaned_dataset['clean_comment']
y_cleaned = cleaned_dataset['category']

# Split the cleaned data into train and test sets (80-20 split)
X_train_cleaned, X_test_cleaned, y_train_cleaned, y_test_cleaned = train_test_split(X_cleaned, y_cleaned, test_size=0.2, random_state=42)

# Apply TfidfVectorizer with trigram setting and max_features=10000
tfidf_cleaned = TfidfVectorizer(ngram_range=(1, 3), max_features=10000)

# Fit the vectorizer on the training data and transform both train and test sets
X_train_tfidf_cleaned = tfidf_cleaned.fit_transform(X_train_cleaned)
X_test_tfidf_cleaned = tfidf_cleaned.transform(X_test_cleaned)

# Base learners
lightgbm_model = LGBMClassifier(
    objective='multiclass',
    num_class=3,
    metric="multi_logloss",
    is_unbalance=True,
    class_weight="balanced",
    reg_alpha=0.1,  # L1 regularization
    reg_lambda=0.1,  # L2 regularization,
    learning_rate=0.08081298097796712,
    n_estimators=367,
    max_depth=20
)

logreg_model = LogisticRegression(max_iter=1000, class_weight='balanced', solver='lbfgs', multi_class='multinomial')

# Meta-learner
knn_meta_learner = KNeighborsClassifier(n_neighbors=5)

# Create the StackingClassifier with LightGBM and LogisticRegression as base models, and KNN as meta-learner
stacking_model = StackingClassifier(
    estimators=[
        ('lightgbm', lightgbm_model),
        ('logistic_regression', logreg_model)
    ],
    final_estimator=knn_meta_learner,
    cv=5
)

# Train the stacking model
stacking_model.fit(X_train_tfidf_cleaned, y_train_cleaned)

# Make predictions on the test data
y_pred = stacking_model.predict(X_test_tfidf_cleaned)

# Generate classification report
print(classification_report(y_test_cleaned, y_pred))
