from flask import Flask
app = Flask(__name__)

app.config.from_pyfile('default_config.py')

from review import views