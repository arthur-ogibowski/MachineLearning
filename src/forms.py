from flask_wtf import FlaskForm
from wtforms import StringField, SubmitField, SelectField


class LearningForm(FlaskForm):
    classifier = SelectField('Classificador', choices=[('knn', 'KNN'), ('mlp', 'MLP'), ('dt', 'Decision Tree'), ('rf', 'Random Forest')])
    param1 = StringField('Primeiro parâmetro')
    param2 = StringField('Segundo parâmetro')
    param3 = StringField('Terceiro parâmetro')
    submit = SubmitField('PREVER')