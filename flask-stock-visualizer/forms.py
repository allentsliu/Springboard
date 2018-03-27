from flask_wtf import FlaskForm
from wtforms import StringField, SubmitField
from wtforms.validators import DataRequired

class AddForm(FlaskForm):
    m = StringField('M')
    n = StringField('N')
    submit = SubmitField('Add')
