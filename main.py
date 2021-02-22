# importing the necessary dependencies
from flask import Flask, render_template, request,jsonify
from flask_cors import CORS,cross_origin
import numpy as np
import pickle

app = Flask(__name__) # initializing a flask app

@app.route('/',methods=['GET'])  # route to display the home page
@cross_origin()
def homePage():
    return render_template("index.html")

@app.route('/predict',methods=['POST','GET']) # route to show the predictions in a web UI
@cross_origin()
def index():
    if request.method == 'POST':
        try:
            #  reading the inputs given by the user
            Term = float(request.form['Term'])
            NoEmp = float(request.form['NoEmp'])
            is_NewExist = request.form['NewExist']
            if (is_NewExist == 'Existing business'):
                NewExist = 0
            else:
                NewExist = 1
            is_LowDoc = request.form['LowDoc']
            if (is_LowDoc == 'No'):
                LowDoc = 0
            else:
                LowDoc = 1
            GrAppv = float(request.form['GrAppv'])
            Backed_by_SBA = float(request.form['Backed_by_SBA'])

            """
            is_research = request.form['research']
            if(is_research=='yes'):
                research=1
            else:
                research=0
                """

            filename = 'finalized_model.pickle'
            loaded_model = pickle.load(open(filename, 'rb'))  # loading the model file from the storage

            prediction = loaded_model.predict(np.array([[Term, NoEmp, NewExist, LowDoc,
                                                         GrAppv, Backed_by_SBA]]).reshape(1, -1))
            print('prediction is', prediction)
            # showing the prediction results in a UI
            translate = {0: 'The company is likely to default on the loan.',
                         1: 'The company is likely to payout the loan in full.'}
            return render_template('results.html', prediction=translate[prediction[0]])

        except Exception as e:
            print('The Exception message is: ', e)
            return 'something is wrong'

    # return render_template('results.html')

    else:
        return render_template('index.html')


if __name__ == "__main__":
    # app.run(host='127.0.0.1', port=8001, debug=True)
    app.run(debug=True)  # running the app
