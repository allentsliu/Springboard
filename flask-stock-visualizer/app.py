from flask import Flask, render_template, jsonify
from stock_scraper import get_data
from forms import AddForm
from config import Config
import os


app = Flask(__name__)
app.config.from_object(Config)


@app.route("/data")
def data():
    return jsonify(get_data())


@app.route("/", methods=['GET', 'POST'])
def index():
    form = AddForm()
    if form.validate_on_submit():
        m = form.m.data
        n = form.n.data
        result = addition(m, n)
    return render_template("index.html", form=form)

def addition(m, n):
    return m + n

if __name__ == '__main__':
    port = int(os.environ.get('PORT', 5000))
    app.run(host='0.0.0.0', port=port)
