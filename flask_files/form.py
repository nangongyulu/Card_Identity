from wtforms import Form, FileField
from flask_wtf.file import FileRequired, FileAllowed


class UploadForm(Form):
    file = FileField(validators=[FileRequired(), FileAllowed(['jpg', 'png'])])
