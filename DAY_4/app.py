from flask import Flask, render_template, request
import pickle
import numpy as np

# https://youtu.be/pMIwu5FwJ78

model = pickle.load(open("rf_model.pkl", "rb"))

app = Flask(__name__)

@app.route("/", methods=["POST", "GET"])
def home():
    if request.method=="POST":
        input_dict = request.form.to_dict()
        input_values = input_dict.values()
        input_values = list(map(float, list(input_values)))
        input_values = np.array(input_values)
        input_values = input_values.reshape(1, -1)
        prediction = model.predict(input_values)[0]
        message = ""
        if prediction==0:
            message="Patient is not diabetic"
        else:
            message="Patient is diabetic"
        return message
        
    return render_template("home.html")


if __name__=="__main__":
    app.run(debug=True)