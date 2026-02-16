from flask import Flask, request, render_template
import numpy as np
import pickle

app = Flask(__name__)

# Load trained model
model = pickle.load(open("artifacts/model.pkl", "rb"))

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    int_features = [float(x) for x in request.form.values()]
    final_features = np.array([int_features])
    prediction = model.predict(final_features)

    output = round(prediction[0], 2)

    return render_template('result.html', prediction_text=f'Predicted Value: {output}')

if __name__ == "__main__":
    app.run(debug=True)
