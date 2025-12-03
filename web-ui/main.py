from flask import Flask, render_template, send_from_directory, url_for, request
from werkzeug.utils import secure_filename

from flask_wtf import FlaskForm
from flask_wtf.file import FileField, FileRequired, FileAllowed
from wtforms import SubmitField

from script.tb_detect import predict_tb

import os
import json

app = Flask(__name__)
app.config['SECRET_KEY'] = 'your_secret_key'
app.config['UPLOAD_FOLDER'] = 'uploads'

os.makedirs(app.config['UPLOAD_FOLDER'], exist_ok=True)

class UploadForm(FlaskForm):
    photo = FileField('Photo', validators=[
        FileRequired('Please select a file!'),
        FileAllowed(['jpg', 'jpeg', 'png'], 'Images only!')
    ])
    submit = SubmitField('Analyze')

@app.route('/uploads/<filename>')
def get_file(filename):
    return send_from_directory(app.config['UPLOAD_FOLDER'], filename)

@app.route('/', methods=['GET', 'POST'])
def upload_image():
    form = UploadForm()
    file_url = None
    prediction = None

    if form.validate_on_submit():
        file = form.photo.data
        filename = secure_filename(file.filename)

        filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
        file.save(filepath)

        file_url = url_for('get_file', filename=filename)

        # Run prediction
        prediction = predict_tb(filepath)

        return render_template(
            "index.html",
            form=form,
            file_url=file_url,
            prediction=prediction
        )


    return render_template(
            "index.html",
            form=form,
            file_url=file_url,
            prediction=prediction
        )

if __name__ == '__main__':
    app.run(host="0.0.0.0", port=8080, debug=True)
