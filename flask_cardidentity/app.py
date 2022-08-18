from flask import Flask, redirect, url_for, request, render_template
import config

app = Flask(__name__)
app.config.from_object(config)


@app.route('/')
def index():
    return render_template("templates/login.html")


@app.route('/success/<name>')
def success(name):
    return 'welcome %s' % name


@app.route('/login', methods=['POST', 'GET'])
def login():
    if request.method == 'POST':
        print("post_1")
        user = request.form['nm']
        return redirect(url_for('success', name=user))
    else:
        print("post_2")
        user = request.args.get('nm')
        return redirect(url_for('success', name=user))


if __name__ == '__main__':
    app.run()
