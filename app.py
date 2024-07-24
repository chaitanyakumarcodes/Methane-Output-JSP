from flask import Flask, request, render_template
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVR
import pickle

app = Flask(__name__)

# Load the trained model and scaler
model = pickle.load(open('svr_model.pkl', 'rb'))
scaler = pickle.load(open('scaler.pkl', 'rb'))

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict', methods=['GET', 'POST'])
def predict():
    if request.method == 'POST':
        # Get the input values from the form
        count = float(request.form['Count'])
        syngas_header_pressure = float(request.form['Syngas_Header_Pressure'])
        avg_tr5 = float(request.form['Avg_TR5'])
        avg_hp_bfw = float(request.form['Avg_HP_BFW'])
        avg_tr4 = float(request.form['Avg_TR4'])
        avg_co2 = float(request.form['Avg_CO2'])
        avg_hp_o2 = float(request.form['Avg_HP_O2'])
        vm_wt = float(request.form['VM_wt'])
        ash_wt = float(request.form['ASH_wt'])
        fc_wt = float(request.form['FC_wt'])

        # Create a DataFrame for the input
        input_data = pd.DataFrame([[count, syngas_header_pressure, avg_tr5, avg_hp_bfw, avg_tr4, avg_co2, avg_hp_o2, vm_wt, ash_wt, fc_wt]], 
                                  columns=['Count', 'Syngas Header Pressure', 'Avg TR5', 'Avg HP_BFW', 'Avg TR4', 'Avg CO2', 'Avg HP_O2', 'VM(wt%)', 'ASH(wt%)', 'FC(wt%)'])

        # Ensure the column order matches the order used during training
        input_data = input_data[['Syngas Header Pressure', 'Count', 'VM(wt%)', 'ASH(wt%)', 'FC(wt%)',
       'Avg TR5', 'Avg HP_BFW', 'Avg TR4', 'Avg CO2', 'Avg HP_O2']]
        
        # Standardize the input data
        input_data_scaled = scaler.transform(input_data)
        
        # Make the prediction
        prediction = model.predict(input_data_scaled)

        # Render the result page with the prediction
        return render_template('result.html', prediction=prediction[0])

    return render_template('predict.html')

if __name__ == "__main__":
    app.run(debug=True)