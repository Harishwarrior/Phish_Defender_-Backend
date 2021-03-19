# importing libraries
from flask import Flask, abort, jsonify, request, render_template
from flask_cors import CORS, cross_origin
from flask_ngrok import run_with_ngrok
import time
import json
import datetime
import os
from flask_restful import Resource, Api
import joblib
import inputScript
import numpy as np

# Start ngrok when app is run
app = Flask(__name__)   
# run_with_ngrok(app)


# root directory
@app.route("/")
def hello():
    return render_template('index.html')


@app.route("/api", methods=['POST'])
def make_predict():
    # error checking
    data = request.get_json(force=True)

    # extract url from request
    url_to_be_predicted = data['url']
    url = str(url_to_be_predicted)

    # load the pickle file
    classifier = joblib.load('final_models/rf_final.pkl')

    #checking and predicting
    try:
        checkprediction = inputScript.main(url)
        prediction = int(classifier.predict(checkprediction))
        print(prediction)
        result = {"prediction": prediction}
    except Exception as e:
        print(e)
        result = {"prediction": -9999}

    return jsonify(result)


if __name__ == "__main__":

    # when deploying in NGROK server(PC as server)
    # app.run()

    # When using Heroku or any other online platform for deployment
    app.run(debug=True,host='0.0.0.0', port=int(os.environ.get("PORT", 5000)))
