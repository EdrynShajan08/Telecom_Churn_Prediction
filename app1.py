import numpy as np
from flask import Flask, request, jsonify, render_template
import pickle

app = Flask(__name__)
path = r'D:\dep5\Telecom_Churn_Prediction\model_churn.pkl'
model = pickle.load(open(path, 'rb'))
# Churn_pred


@app.route('/')
def home():
    return render_template('Churn_pred.html')


@app.route('/predict', methods=['POST'])
def predict():
    '''
    For rendering results on HTML GUI
    '''
    int_features = [float(x) for x in request.form.values()]
    final_features = [np.array(int_features)]
    prediction = model.predict(final_features)

    output = round(prediction[0], 2)
    if output > 0.5:
        output = 'A Churn case'
    else:
        output = 'Not A Churn case'
    return render_template('Churn_pred.html', prediction_text='This case is  {}'.format(output))


if __name__ == "__main__":
    app.run(debug=True)
