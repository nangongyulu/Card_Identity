from flask_wtf.file import FileRequired, FileAllowed
from wtforms import Form, FileField

class UploadForm(Form):
    file = FileField(validators=[FileRequired(), FileAllowed(['jpg', 'png'])])
