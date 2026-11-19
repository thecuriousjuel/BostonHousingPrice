import os
from flask import Flask, request, render_template, url_for
from werkzeug.middleware.proxy_fix import ProxyFix
import joblib
import pandas as pd

app = Flask(__name__)
app.config['TEMPLATES_AUTO_RELOAD'] = True
app.wsgi_app = ProxyFix(app.wsgi_app, x_for=1, x_proto=1, x_host=1, x_prefix=1)

BASE_DIR = os.path.dirname(os.path.abspath(__file__))


@app.route('/', methods=['GET'])
def homepage():
    return render_template('index.html')


@app.route('/predict', methods=['POST'])
def predict():
    CRIM = float(request.form['CRIM'])
    ZN = float(request.form['ZN'])
    INDUS = float(request.form['INDUS'])
    CHAS = float(request.form['CHAS'])
    NOX = float(request.form['NOX'])
    RM = float(request.form['RM'])
    AGE = float(request.form['AGE'])
    DIS = float(request.form['DIS'])
    RAD = float(request.form['RAD'])
    TAX = float(request.form['TAX'])
    PTRATIO = float(request.form['PTRATIO'])
    B = float(request.form['B'])
    LSTAT = float(request.form['LSTAT'])

    data = pd.DataFrame(
        [[CRIM, ZN, INDUS, CHAS, NOX, RM, AGE, DIS, RAD, TAX, PTRATIO, B, LSTAT]],
        columns=['CRIM', 'ZN', 'INDUS', 'CHAS', 'NOX', 'RM', 'AGE', 'DIS', 'RAD', 'TAX', 'PTRATIO', 'B', 'LSTAT']
    )

    lin_reg = joblib.load(os.path.join(BASE_DIR, 'model', 'Linear-Regression-Model.pkl'))
    numerical_transformer = joblib.load(os.path.join(BASE_DIR, 'model', 'numerical_transformer.pkl'))
    column_transformer = joblib.load(os.path.join(BASE_DIR, 'model', 'column_transformer.pkl'))

    prepared_data = column_transformer.transform(data)
    output = lin_reg.predict(prepared_data)
    final_output = numerical_transformer.inverse_transform(output)
    final_output = '{:.2f}'.format(final_output[0, 0])

    return render_template('index.html', result=final_output)


if __name__ == '__main__':
    os.environ.pop("WERKZEUG_SERVER_FD", None)
    os.environ.pop("WERKZEUG_RUN_MAIN", None)
    port = int(os.environ.get("PORT", os.environ.get("BOSTON_LINEAR_PORT", 6207)))
    host = os.environ.get("HOST", "127.0.0.1")
    print(f"🚀 Boston Housing Linear Regression App running on http://{host}:{port}")
    app.run(host=host, port=port, debug=False, use_reloader=False)
