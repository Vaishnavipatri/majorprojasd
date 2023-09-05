import numpy as np
import pandas as pd
import win32api
import pyttsx3
from flask import Flask, request, jsonify, render_template
import joblib

model = joblib.load('ASD_final.pkl')

app = Flask(__name__)
text_speech=pyttsx3.init()
 

@app.route('/')
def home():
    return render_template('firstpage1.html')

@app.route('/Autism_Disease_prediction')
def Autism_Disease_prediction():
    return render_template('firstpage1.html')

@app.route('/Autism_Disease_prediction/predict',methods=['POST'])

def predict():

    '''for rendering results on HTML GUI'''
    int_features = [str(x) for x in request.form.values()]
    print(int_features)
    print( len(int_features))
    a=int_features
    #a=int_features[:len(int_features)-1]

    print(a)

    data=a

    print(data)
    dictonary= {'Yes':1, 'No':0,'Male':1, 'Female':0,'Self': 1,'Parent': 2, 'Health care professional':3, 'Relative':4, 'Others':5}
    data[0]=int(data[0])
    def get_key(val): 
            for key, value in dictonary.items():
                if val == key:
                    return value
    data[1]=get_key(data[1])
    data[2]=get_key(data[2])
    data[3]=get_key(data[3])
    data[4]=get_key(data[4])
    data[5]=get_key(data[5])
    data[6]=get_key(data[6])
    data[7]=get_key(data[7])
    data[8]=get_key(data[8])
    data[9]=get_key(data[9])
    data[10]=get_key(data[10])
    data[11]=get_key(data[11])
    data[12]=get_key(data[12])
    data[13]=get_key(data[13])
    data[14]=get_key(data[14])
    data[15]=get_key(data[15])

    print(data)

    data_array=np.array(data)
    data_array=data_array.reshape(1,-1)
    print(data_array)
    my_prediction = model.predict(data_array)
    my_prediction=int(my_prediction)
    print("user custom output: "+str(my_prediction))

   
    #test input
    t=[27,'m','yes','yes','no','Parent',1,1,0,1,1,0,1,1,1,1]
    dic= {'yes':1, 'no':0,'m':1, 'f':0,'Self': 1,'Parent': 2, 'Health care professional':3, 'Relative':4, 'Others':5,'1':1,'0':0}
    
    def get_key2(val2): 
            for key, value in dic.items():
                if str(val2) == key:
                    return value
    t[1]=get_key2(t[1])
    t[2]=get_key2(t[2])
    t[3]=get_key2(t[3])
    t[4]=get_key2(t[4])
    t[5]=get_key2(t[5])
    t[6]=get_key2(t[6])
    t[7]=get_key2(t[7])
    t[8]=get_key2(t[8])
    t[9]=get_key2(t[9])
    t[10]=get_key2(t[10])
    t[11]=get_key2(t[11])
    t[12]=get_key2(t[12])
    t[13]=get_key2(t[13])
    t[14]=get_key2(t[14])
    t[15]=get_key2(t[15])
    t[0]=int(t[0])
    print(t)

    t_array=np.array(t)
    t_array=t_array.reshape(1,-1)
    print(t_array)
    test_prediction = model.predict(t_array)
    test_prediction=int(test_prediction)
    print("test prediction: "+str(test_prediction))

    lvl=""
    cnt=data[6:]
    #print(cnt)
    c=0
    for e in cnt:
        if e==1:
            c+=1
    if c>6:
        lvl="High level or severe autism traits"
    else:
        lvl="Very low or No autism traits"
      
    if my_prediction==1:
        print("The Person has the Autism spectrum disorder ")
        return render_template('firstpage1.html',prediction_text="user custom output is The Person has the Autism Spectrum Disorder with "+lvl)
        #return render_template('firstpage.html',prediction_text="user custom output is The Person has the Autism Spectrum Disorder and "+lvl+" existing test output is "+ str(test_prediction))
    else:
        print("The Person does not have the Autism spectrum disorder")
        return render_template('firstpage1.html',prediction_text="user custom output is The Person not have the Autism Spectrum Disorder with "+ lvl)
    
    

    
    

    #print("Test data prediction:")
    '''print(test_prediction)

    if test_prediction==1:
        print("The Person has the Autism spectrum disorder ")
    else:
        print("The Person does not have the Autism spectrum disorder ")'''
    




    #text to speech code
    #reply=int_features[-1]
    #text_speech.say(reply)
    #text_speech.runAndWait()

@app.route("/texttospeech",methods=['GET','POST'])
def texttospeech():
    if request.method=="POST":
        reply=request.form["enter text"]
        text_speech.say(reply)
        text_speech.runAndWait()
        text_speech.endLoop()
        text_speech.stop()
        return redirect(url_for("reply",rep=reply))
    else:
        return render_template("txt.html")
@app.route("/<rep>")
def reply(rep):
    return f"<h1>{rep}</h1>"
    
if __name__=="__main__":
                        

    app.run(debug=False)




 

    

    
                        
            

 



 
