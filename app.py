from tensorflow import keras
import pandas as pd
from flask import Flask, render_template, request,url_for,redirect
import numpy as np
from tqdm import tqdm
import tensorflow as tf

#import matplotlib.pyplot as plt
import os



app = Flask(__name__)
app.config['UPLOAD_FOLDER']=''



@app.route("/")
def index():
    return render_template("index.html",result="NULL")

@app.route("/api",methods=["POST"])
def api():

    uploaded_file =request.files['stock']
    '''if uploaded_file.filename != '':
           file_path = os.path.join(app.config['UPLOAD_FOLDER'], uploaded_file.filename)

           uploaded_file.save(file_path)'''


    df = pd.read_csv(uploaded_file)
    result=model_train(df)



    #os.remove(file_path)
    return render_template("index.html",result=result)




#for deeplearning

def dropit(stock):
  stock=stock.drop(['Date','WAP','No.of Shares' , 'No. of Trades',	'Total Turnover (Rs.)',	'Deliverable Quantity',	'% Deli. Qty to Traded Qty',	'Spread High-Low',	'Spread Close-Open'],axis=1)
  stock =stock.iloc[::-1].reset_index()
  stock=stock.drop(['index'],axis=1)
  return stock

def sma(stock):
  
  stock = stock.assign(SMA=pd.Series([1 for i in range(0,stock.shape[0])]).values)
  stock["SMA"]=stock["Close Price"].rolling(window=10).mean()
  
  return stock

def ema(stock):
  stock.assign(EMA=pd.Series([1 for i in range(0,stock.shape[0])]).values)
  a=stock["Close Price"].ewm(adjust=False,alpha=0.5)
  stock["EMA"]=a.mean()
  return stock

def norm(stonk):
  '''min=stonk[1:4].min().to_numpy()
  max=stonk[1:4].max().to_numpy()'''
  stonk.iloc[:,1:]-=stonk.iloc[:,1:].min()
  stonk.iloc[:,1:]/=stonk.iloc[:,1:].max()
  return stonk

def batch(stocks,cols,size):
    d=stocks.to_numpy()
    X=np.empty((0,cols))
    Y=np.array([])
    c=0
    for i in tqdm(d):
  
  #i=i.reshape(5,1)  
  
        if c%10==0 and c is not 0:
            Y=np.append(Y,i[3])
    
  
  
        i=np.array([i])
        X=np.append(X,i,axis=0)
        c+=1
        if c==size:
            break

    return X,Y


def model_train(df):
   
    df=dropit(df)
    df=sma(df)
    df=ema(df)
    
    print(df.min)
    min=df.min().to_numpy()[3]
    max=df.max().to_numpy()[3]

    df=norm(df)
    df=df.iloc[10:]
    cols=df.shape[1]
    print(df.head)
    size_total=df.shape[0]
    print(f"size_total: {size_total}")
    X,Y=batch(df,cols,int((size_total//10)*10))
    size_floor=df.shape[0]//10
    print(X.shape)
    print(Y.shape)
    X=np.split(X,X.shape[0]//Y.shape[0])
    X=np.array(X)
    X=X.reshape(size_floor,10,cols)
    X=X[0:size_floor-1,:,:]
    print(X.shape)
    print(Y.shape)
    model=keras.Sequential([
                      
                      keras.layers.LSTM(1024,return_sequences=True, input_shape=(10,cols)),
                      keras.layers.LSTM(512,return_sequences=True),
                      keras.layers.LSTM(256,return_sequences=True),
                      keras.layers.LSTM(64),
                      keras.layers.Dense(128,activation=keras.activations.relu),
                      keras.layers.Dense(64,activation=keras.activations.relu),
                      keras.layers.Dense(1)
    ])

    model.compile(loss=tf.keras.losses.mean_squared_error, optimizer=tf.keras.optimizers.Adam(),metrics=['mae', 'mse'])
    print("start training")
    model.fit(x=X,y=Y,batch_size=32,epochs=10)
    print("end training")

    test=df[-10:].to_numpy()
    #print(f"real: {df.iloc[4026][3]*max+min}")
    print(f"Predicted: {model.predict(test.reshape(1,10,cols))*max+min}")
    return model.predict(test.reshape(1,10,cols))*max+min
    
    






    






