from flask import Flask, render_template_string, send_from_directory
import os

app = Flask(__name__)

@app.route('/')
def index():
    with open(os.path.join(os.path.dirname(__file__), 'index.html')) as f:
        return f.read()

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=int(os.environ.get('PORT', 5010)))
