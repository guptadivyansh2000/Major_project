import numpy as np
from flask import Flask, request, jsonify, render_template
import pickle
app=Flask(__name__)
dt_model = pickle.load(open('dt_model.pkl','rb'))
rf_model = pickle.load(open('rf_model.pkl','rb'))
svm_model = pickle.load(open('svm_model.pkl','rb'))
loaded_models = {
    'dt': dt_model,
    'rf': rf_model,
    'svm': svm_model
}

# Function to decode predictions 
def decode(pred):
    if pred == 1: return 'Customer Exits'
    else: return 'Customer Stays'

@app.route('/')
def home():
    # Initial rendering
    result = [{'model':'Decision Tree', 'prediction':' '},
              {'model': 'Random Forest', 'prediction': ' '},
              {'model': 'SVM', 'prediction': ' '}]
    
    # Create main dictionary
    maind = {}
    maind['customer'] = {}
    maind['predictions'] = result

    return render_template('index.html', maind=maind)

@app.route('/predict', methods=['POST'])
def predict():

    # List values received from index
    values = [x for x in request.form.values()]

    # new_array - input to models
    new_array = np.array(values).reshape(1, -1)
    print(new_array)
    print(values)
    
    # Key names for customer dictionary custd
    cols = ['CreditScore',
            'Geography',
            'Gender',
            'Age',
            'Tenure',
            'Balance',
            'NumOfProducts',
            'HasCrCard',
            'IsActiveMember',
            'EstimatedSalary']

    # Create customer dictionary
    custd = {}
    for k, v in  zip(cols, values):
        custd[k] = v

    # Convert 1 or 0 to Yes or No    
    yn_val = ['HasCrCard', 'IsActiveMember']
    for val in  yn_val:
        if custd[val] == '1': custd[val] = 'Yes'
        else: custd[val] = 'No'

    # Loop through 'loaded_models' dictionary and
    # save predictiond to the list
    predl = []
    for m in loaded_models.values():
        predl.append(decode(m.predict(new_array)[0]))

    result = [
            {'model':'Decision Tree', 'prediction':predl[0]},
            {'model': 'Random Forest', 'prediction': predl[1]},
            {'model': 'SVM', 'prediction': predl[2]}
            ]            

    # Create main dictionary
    maind = {}
    maind['customer'] = custd
    maind['predictions'] = result

    return render_template('index.html', maind=maind)


if __name__ == "__main__":
    app.run(debug=False)