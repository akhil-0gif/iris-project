from tensorflow.keras.models import load_model
from flask import Flask,render_template,request
import numpy as np

app=Flask(__name__)
#loadedModel=load(open('model.h5','rb'))
loadedModel=load_model('model.h5')

@app.route('/')
def home():
    return render_template('iris.html')

@app.route('/prediction',methods=['POST'])
def prediction():
    SepalLengthCm=float(request.form['SepalLengthCm'])
    SepalWidthCm=float(request.form['SepalWidthCm'])
    PetalLengthCm=float(request.form['PetalLengthCm'])
    PetalWidthCm=float(request.form['PetalWidthCm'])
    
    prediction=loadedModel.predict([[SepalLengthCm,SepalWidthCm,PetalLengthCm,PetalWidthCm]])
    prediction = prediction.argmax(axis=1)
    
    if prediction[0]==0:
        prediction = "Setosa"
    elif prediction[0] ==1:
        prediction = "Versicolor"
    elif prediction[0] ==2:  
        prediction = "Virginica"
    else: 
        prediction = "this species is not exist"
      
    return render_template('iris.html',api_output=prediction) 

if __name__== "__main__":
    app.run(debug = 'True')

    
#SepalWidthCm	PetalLengthCm	PetalWidthCm