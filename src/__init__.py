from flask import Flask
from flask_sqlalchemy import SQLAlchemy


app = Flask(__name__, static_url_path='/static')
app.static_folder = 'static'
app.config['SQLALCHEMY_DATABASE_URI'] = 'sqlite:///comunidade.db'
app.config['SECRET_KEY'] = 'p2312opdj31hd013jn'
app.config['UPLOAD_FOLDER'] = 'static/photos_posts'

database = SQLAlchemy(app)

from src import routes