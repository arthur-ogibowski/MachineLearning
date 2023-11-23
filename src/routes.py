import os

import pandas as pd

from flask import Flask, render_template, request, send_file, current_app
from sklearn import datasets
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix
from sklearn.neighbors import KNeighborsClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from io import BytesIO
import base64

from src import app
from src.forms import LearningForm
import matplotlib
matplotlib.use('agg')

@app.route('/', methods=['GET', 'POST'])
def index():
    form = LearningForm()

    return render_template('index.html', form=form)

@app.route('/predict', methods=['POST'])
def predict():
    classifier_name = request.form.get('classifier')
    parameters = get_parameters(request.form, classifier_name)

    iris = datasets.load_iris()
    X = iris.data
    y = iris.target

    # Dividindo o conjunto de dados entre treinamento e teste
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.4)

    # Inicializando o classificador
    classifier = get_classifier(classifier_name, parameters)

    # Treinando o classificador
    classifier.fit(X_train, y_train)

    # Realizando previsões
    y_pred = classifier.predict(X_test)

    # Calculando métricas
    accuracy = accuracy_score(y_test, y_pred)
    precision = precision_score(y_test, y_pred, average='macro')
    recall = recall_score(y_test, y_pred, average='macro')
    f1 = f1_score(y_test, y_pred, average='macro')

    # Criando matriz de confusão
    classes = iris.target_names.tolist()
    cm = confusion_matrix(y_test, y_pred)
    plot_confusion_matrix(cm, classes)

    image_64 = to_base64()

    result = {
        'accuracy': accuracy,
        'precision': precision,
        'recall': recall,
        'f1': f1,
        'image': image_64,
    }

    return render_template('index.html', form=LearningForm(), result=result)


def get_parameters(form_data, classifier_name):
    # Recebendo os parâmetros do formulário
    params = {}
    for i in range(1, 4):
        param_key = f'param{i}'
        param_value = form_data.get(param_key)
        params[param_key] = param_value


    # classificador K-nearest Neighbor
    if classifier_name == 'knn':
        params['n_neighbors'] = int(params.get('param1')) # ex=5
        params['weights'] = str(params.get('param2')) # ex=uniform
        params['leaf_size'] = int(params.get('param3')) # ex=30

    # classificador Multilayer Perceptron
    elif classifier_name == 'mlp':
        params['hidden_layer_size'] = int(params.get('param1')) # ex=100
        params['max_iter'] = int(params.get('param2')) # ex=200
        params['learning_rate'] = str(params.get('param3')) # ex=constant

    # classificador Decision Tree
    elif classifier_name == 'dt':
        params['max_depth'] = int(params.get('param1')) # ex=5
        params['min_sample_leaf'] = int(params.get('param2')) # ex=30
        params['criterion'] = str(params.get('param3')) # ex=gini

    # classificador Random Forest
    elif classifier_name == 'rf':
        params['n_estimators'] = int(params.get('param1')) # ex=100
        params['max_depth'] = int(params.get('param2')) # ex=5
        params['max_features'] = int(params.get('param3')) # ex=2

    return params

def get_classifier(name, params):
    classifier = None
    if name == 'knn':
        classifier = KNeighborsClassifier(n_neighbors=params['n_neighbors'], weights=params['weights'], leaf_size=params['leaf_size'])
    elif name == 'mlp':
        classifier = MLPClassifier(hidden_layer_sizes=params['hidden_layer_size'], max_iter=params['max_iter'], learning_rate=params['learning_rate'])
    elif name == 'dt':
        classifier = DecisionTreeClassifier(max_depth=params['max_depth'], min_samples_leaf=params['min_sample_leaf'], criterion=params['criterion'])
    elif name == 'rf':
        classifier = RandomForestClassifier(n_estimators=params['n_estimators'], max_depth=params['max_depth'], max_features=params['max_features'])

    return classifier


def plot_confusion_matrix(cm, classes):
    plt.figure(figsize=(12, 8))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Greens', xticklabels=classes, yticklabels=classes)
    plt.xlabel('Predicted')
    plt.ylabel('Actual')
    plt.title('Confusion Matrix')

    # Verificando se o diretório existe
    conf_photos_dir = 'src/static/assets'
    if not os.path.exists(conf_photos_dir):
        os.makedirs(conf_photos_dir)

    # Salvando a matriz de confusão
    plt.savefig(os.path.join(conf_photos_dir, 'confusion_matrix.png'))

def to_base64():
    img = BytesIO()
    plt.savefig(img, format='png')
    img.seek(0)
    img_str = base64.b64encode(img.read()).decode()
    return img_str


@app.route('/download_confusion_matrix')
def download_confusion_matrix():
    assets_path = os.path.join(current_app.root_path, 'static/assets')
    file_path = os.path.join(assets_path, 'confusion_matrix.png')
    return send_file(file_path, as_attachment=True, mimetype='image/png')
