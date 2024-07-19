import pandas as pd
from sklearn.model_selection import RandomizedSearchCV, cross_val_score, train_test_split
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.pipeline import Pipeline
from xgboost import XGBClassifier
from sklearn.svm import SVC

# Načítanie dát
data_2014_2015 = pd.read_csv('epl_dataset/2014-2015.csv')
data_2015_2016 = pd.read_csv('epl_dataset/2015-2016.csv')
data_2016_2017 = pd.read_csv('epl_dataset/2016-2017.csv')
data_2017_2018 = pd.read_csv('epl_dataset/2017-2018.csv')
data_2018_2019 = pd.read_csv('epl_dataset/2018-2019.csv')
data_2019_2020 = pd.read_csv('epl_dataset/2019-2020.csv')
data_2020_2021 = pd.read_csv('epl_dataset/2020-2021.csv')
data_2021_2022 = pd.read_csv('epl_dataset/2021-2022.csv')


# Spojenie dát
data = pd.concat([data_2014_2015, data_2015_2016, data_2016_2017, data_2017_2018, data_2018_2019, data_2019_2020, data_2020_2021, data_2021_2022], ignore_index=True)

# Potrebný výber stĺpcov z datasetu
columns_of_interest = ['HomeTeam', 'AwayTeam', 'FTR', 'HS', 'AS', 'HST', 'AST', 'HF', 'AF', 'HC', 'AC', 'HY', 'AY']
data = data[columns_of_interest]

# Encoder pre cieľovú premennú a tím
le = LabelEncoder()
data['FTR'] = le.fit_transform(data['FTR'])  # 'H', 'D', 'A' -> 0, 1, 2
data['HomeTeam'] = le.fit_transform(data['HomeTeam'])
data['AwayTeam'] = le.fit_transform(data['AwayTeam'])

# Vytvorenie cieľovej premennej a príznakov
y = data['FTR']
X = data[['HomeTeam', 'AwayTeam', 'HS', 'AS', 'HST', 'AST', 'HF', 'AF', 'HC', 'AC', 'HY', 'AY']]
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Funkcia na vytvorenie a tréning modelov
def train_model(classifier, params, X_train, y_train, cv_folds):
    pipeline = Pipeline([
        ('scaler', StandardScaler()),
        ('classifier', classifier)
    ])
    random_search = RandomizedSearchCV(estimator=pipeline, param_distributions=params, n_iter=100, cv=cv_folds, verbose=2, random_state=42, n_jobs=-1)
    random_search.fit(X_train, y_train)
    cv_scores = cross_val_score(random_search.best_estimator_, X_train, y_train, cv=cv_folds)
    return {
        "best_params": random_search.best_params_,
        "best_score": random_search.best_score_,
        "cv_scores": cv_scores,
        "cv_average": cv_scores.mean()
    }

# Modely a ich nastavenia
models = {
    'RandomForest': (RandomForestClassifier(), {
        'classifier__n_estimators': [100, 200, 300],
        'classifier__max_features': ['sqrt', 'log2'],
        'classifier__max_depth': [None, 10, 20],
        'classifier__min_samples_split': [2, 5],
        'classifier__min_samples_leaf': [1, 2]
    }, 5),
    'XGBoost': (XGBClassifier(use_label_encoder=False, eval_metric='mlogloss'), {
        'classifier__n_estimators': [100, 200, 300],
        'classifier__max_depth': [3, 5, 7],
        'classifier__learning_rate': [0.01, 0.1, 0.2]
    }, 5),
    'SVM': (SVC(), {
        'classifier__C': [0.1, 1, 10],
        'classifier__gamma': ['scale', 'auto'],
        'classifier__kernel': ['linear', 'rbf']
    }, 5),
    'MultinomialLR': (LogisticRegression(multi_class='multinomial', solver='lbfgs'), {
        'classifier__C': [0.1, 1, 10],
        'classifier__max_iter': [100, 200, 300]
    }, 5),
    'DecisionTree': (DecisionTreeClassifier(), {
        'classifier__max_depth': [None, 10, 20, 30],
        'classifier__min_samples_split': [2, 5, 10],
        'classifier__min_samples_leaf': [1, 2, 4]
    }, 5),
    'GradientBoosting': (GradientBoostingClassifier(), {
        'classifier__n_estimators': [100, 200, 300],
        'classifier__learning_rate': [0.01, 0.1, 0.2],
        'classifier__max_depth': [3, 5, 7]
    }, 5)
}

results = {}
for name, (model, params, cv_folds) in models.items():
    print(f"Training and evaluating model: {name}")
    results[name] = train_model(model, params, X_train, y_train, cv_folds)

# Vypísanie výsledkov po dokončení všetkých modelov
for model_name, result in results.items():
    print(f"\nModel: {model_name}")
    print(f"Best Params: {result['best_params']}")
    print(f"Best Score: {result['best_score']}")
    print(f"Cross-validated Scores: {result['cv_scores']}")
    print(f"Average CV Accuracy: {result['cv_average']}")
