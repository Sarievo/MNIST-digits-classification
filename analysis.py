# import numpy as np
# import pandas as pd
# import tensorflow as tf
from tensorflow import keras
from sklearn.model_selection import train_test_split
from sklearn.metrics import precision_score
from sklearn.preprocessing import Normalizer

from sklearn.neighbors import KNeighborsClassifier
# from sklearn.tree import DecisionTreeClassifier
# from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier

from sklearn.model_selection import GridSearchCV

(X_train, y_train), (_, _) = keras.datasets.mnist.load_data()
n, h, w = X_train.shape
X_train = X_train.reshape(n, h * w)

# print(f"""Classes: {np.unique(y_train)}
# Features' shape: {X_train.shape}
# Target's shape: {y_train.shape}
# min: {X_train.min()}, max: {X_train.max()}
# """)

X_train, X_test, y_train, y_test = train_test_split(X_train[:6000], y_train[:6000], test_size=0.3, random_state=40)


# print(f"""x_train shape: {X_train.shape}
# x_test shape: {X_test.shape}
# y_train shape: {y_train.shape}
# y_test shape: {y_test.shape}
# Proportion of samples per class in train set:
# {pd.Series(y_train).value_counts(normalize=True)}
# """)

# Stage 4

# the function
def fit_predict_eval(models, features_train, features_test, target_train, target_test):
    ret = []

    for name, model in models:
        model.fit(features_train, target_train)  # fit the model
        y_pred = model.predict(features_test)  # make a prediction

        # calculate accuracy and save it to score
        score = precision_score(target_test, y_pred, average='macro')
        print(f'Model: {model}\nAccuracy: {score:.4f}\n')
        ret.append([score, name])

    ret.sort(key=lambda x: -x[0])

    print(f'The ret to the 1st question: yes\n')
    print(f'The ret to the 2st question: {ret[0][1]}-{ret[0][0]:.3f}, {ret[1][1]}-{ret[1][0]:.3f}')


# models = [('KNeighborsClassifier', KNeighborsClassifier()),
#           ('DecisionTreeClassifier', DecisionTreeClassifier(random_state=40)),
#           ('LogisticRegression', LogisticRegression()),
#           ('RandomForestClassifier', RandomForestClassifier(random_state=40))]

# without normalization
# fit_predict_eval(
#     models=models,
#     features_train=X_train,
#     features_test=X_test,
#     target_train=y_train,
#     target_test=y_test
# )

transformer = Normalizer()
X_train_norm = transformer.transform(X_train)
X_test_norm = transformer.transform(X_test)


# fit_predict_eval(
#     models=models,
#     features_train=X_train_norm,
#     features_test=X_test_norm,
#     target_train=y_train,
#     target_test=y_test
# )

# Stage 5
def hyperparam_tuning(model, param_grid):
    (name, estimator) = model
    print(f'{name} algorithm')
    search = GridSearchCV(estimator, param_grid, scoring='accuracy', n_jobs=-1).fit(X_train_norm, y_train)
    print(f'best estimator: {search.best_estimator_}')

    ret = search.best_estimator_
    ret.fit(X_train_norm, y_train)  # fit the model
    y_pred = ret.predict(X_test_norm)  # make a prediction

    # calculate accuracy and save it to score
    score = precision_score(y_test, y_pred, average='macro')
    print(f'accuracy: {score:.3f}\n')


param_grid_KNN = {'n_neighbors': [3, 4], 'weights': ['uniform', 'distance'], 'algorithm': ['auto', 'brute']}

param_grid_RF = {'n_estimators': [300, 500], 'max_features': ['auto', 'log2'],
                 'class_weight': ['balanced', 'balanced_subsample']}

hyperparam_tuning(('K-nearest neighbours', KNeighborsClassifier()), param_grid_KNN)
hyperparam_tuning(('Random forest', RandomForestClassifier(random_state=40)), param_grid_RF)
