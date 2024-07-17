from flask import Flask, request, render_template
import pickle
import numpy as np

# load the trained model
model_path = 'model.pkl'
with open(model_path, 'rb') as f:
    model = pickle.load(f)

app = Flask(__name__)

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    # extract data from form
    int_features = [float(x) for x in request.form.values()]  # Changed to float in case your input features are floats
    final_features = [np.array(int_features)]
    
    # Make Prediction
    prediction = model.predict(final_features)
    output = 'setosa' if prediction[0] == 0 else ('versicolor' if prediction[0] == 1 else 'virginica')

    return render_template('index.html', prediction_text='Prediction: {}'.format(output))

if __name__ == '__main__':
    app.run(debug=True)
