from . import app
from .forms import FeatureForm

from flask import render_template

from models.review import ReviewModel, ReviewModel_LR, ReviewModel_NB

review_model = ReviewModel()
review_model_lr = ReviewModel_LR()
review_model_nb = ReviewModel_NB()

@app.route('/')
def index():
    return render_template('welcome.html')


@app.route('/form/', methods=('GET', 'POST'))
def form():
    myform = FeatureForm()

    if myform.is_submitted():
        line = myform.review_text.data
        # review_model = ReviewModel()
        sentiment, hightwords = review_model.predict(line, highlight=True)
        sentiment_lr, hightwords_lr = review_model_lr.predict(line, highlight=True)
        sentiment_nb, hightwords_nb = review_model_nb.predict(line, highlight=True)

        return render_template('result.html',
                               line=line,
                               highlight_words=hightwords, hightwords_lr=hightwords_lr, hightwords_nb=hightwords_nb,
                               sentiment=sentiment, sentiment_lr=sentiment_lr, sentiment_nb=sentiment_nb)

    return render_template('form.html', form=myform)


@app.route('/result/')
def submit():
    return render_template('result.html')


@app.route('/about')
def about():
    return 'The about page'


@app.route('/author')
def author():
    return 'Put your name here'
