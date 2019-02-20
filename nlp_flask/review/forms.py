from flask_wtf import FlaskForm
from wtforms import StringField, SubmitField


class FeatureForm(FlaskForm):
    review_text = StringField('Review Text')
    go = SubmitField('Submit')
