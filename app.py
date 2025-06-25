import re
import numpy as np
from flask import Flask, render_template, request, send_file
import os
import io
import json
import matplotlib.pyplot as plt
import mplfinance as fplt
import plotly
import plotly.express as px
import plotly.graph_objects as go
import seaborn as sns
from werkzeug.utils import send_file
app = Flask(__name__)
from model import *


data=loadData()
#data['Date']=pd.to_datetime(data['Date'])
#emas=data.set_index(pd.DatetimeIndex(data['Date']))
dataclear,dataNorm=normData(data)
inp=pd.DataFrame(dataNorm, columns=["Open","High","Low","Volume"])
out=pd.DataFrame(dataNorm, columns=["Close"])
Xtrain, Xtest, ytrain, ytest=splitData(dataNorm)
xt=pd.DataFrame(Xtrain, columns=["Open","High","Low","Volume"])
yt=pd.DataFrame(ytrain, columns=["Close"])
xts=pd.DataFrame(Xtest,columns=["Open","High","Low","Volume"])
yts=pd.DataFrame(ytest, columns=["Close"])
nn=NeuralNet(layers=[4,10,1], learning_rate=0.01, epoch=500)
w1,b1,w2,b2=nn.init_weights()
w1=pd.DataFrame(w1)
w2=pd.DataFrame(w2).T
b1=pd.DataFrame(b1).T
b2=pd.DataFrame(b2)
wb1,bb1,wb2,bb2=nn.fit(Xtrain,ytrain)
wb1=pd.DataFrame(wb1)
wb2=pd.DataFrame(wb2).T
bb1=pd.DataFrame(bb1).T
bb2=pd.DataFrame(bb2)
train_pred = nn.predict(Xtrain)
test_pred = nn.predict(Xtest)
tp=pd.DataFrame(train_pred,columns=["trainpred"])
t=pd.DataFrame(ytrain,columns=["ytrain"])
dc=pd.concat([t,tp],axis=1)
#nmse=nn.mse(ytrain, train_pred)
nmse=mean_squared_error(ytrain, train_pred)
nmse_test = mean_squared_error(ytest, test_pred)
@app.route("/")
def main():
    #close=json.dumps(close),date=json.dumps(date)
    return render_template('index.html', data=data.to_html(classes='table table-bordered table-striped table-hover'))

@app.route('/chart')
def chart():
    # fplt.plot(emas,type='candle',style='yahoo',savefig='static/assets/img/plot.png')
    figure=go.Figure(
    data= [
        go.Candlestick(
            #x=data['Date'],
            low=data['Low'],
            high=data['High'],
            close=data['Close'],
            open=data['Open']
            # increasing_line_color='green',
            # decreasing_line_color='red'
        )
    ]
    )
    figure.update_layout(
    title= 'GOLD Price',
    yaxis_title='Currency in USD',
    xaxis_title='Date'
    )
    graphJSON = json.dumps(figure, cls=plotly.utils.PlotlyJSONEncoder)
    #print(data)
    #print(type(data))
    # fig=px.line(emas,x='Date',y="Close")
    # graphJSON = json.dumps(fig, cls=plotly.utils.PlotlyJSONEncoder)
    return render_template('chart.html',graphJSON=graphJSON)

@app.route("/normalisasi")
def normalisasi():
    return render_template('normalisasi.html',  dataNorm=dataNorm.to_html(classes='table table-bordered table-striped table-hover'))

@app.route("/split")
def split():
    return render_template('split.html',inp=inp.to_html(classes='table table-bordered table-striped table-hover'),out=out.to_html(classes='table table-bordered table-striped table-hover'))

@app.route("/training")
def training():
    return render_template('training.html',xt=xt.to_html(classes='table table-bordered table-striped table-hover'),yt=yt.to_html(classes='table table-bordered table-striped table-hover'))

@app.route("/testing")
def testing():
    return render_template('testing.html',xts=xts.to_html(classes='table table-bordered table-striped table-hover'),yts=yts.to_html(classes='table table-bordered table-striped table-hover'))

@app.route("/process")
def process():
    fig=px.line(dc,y=['ytrain','trainpred'])
    graphJSON = json.dumps(fig, cls=plotly.utils.PlotlyJSONEncoder)
    figg=px.line(dc,y=['ytrain','trainpred'])
    gJSON = json.dumps(figg, cls=plotly.utils.PlotlyJSONEncoder)
    return render_template('process.html',w1=w1.to_html(classes='table table-bordered table-striped table-hover'),w2=w2.to_html(classes='table table-bordered table-striped table-hover'),b1=b1.to_html(classes='table table-bordered table-striped table-hover'),b2=b2.to_html(classes='table table-bordered table-striped table-hover'),wb1=wb1.to_html(classes='table table-bordered table-striped table-hover'),wb2=wb2.to_html(classes='table table-bordered table-striped table-hover'),bb1=bb1.to_html(classes='table table-bordered table-striped table-hover'),bb2=bb2.to_html(classes='table table-bordered table-striped table-hover'), graphJSON=graphJSON, nmse=nmse)
@app.route("/predict", methods=['GET','POST'])
def predict():
    if request.method =='POST':
        open=float(request.form['open'])
        high=float(request.form['high'])
        low=float(request.form['low'])                                                        
        volume=float(request.form['volume'])
        q=[open, high, low, volume]
        q=pd.DataFrame(q).T
        q.columns=['Open', 'High', 'Low', 'Volume']
        d=(q-dataclear.min())/(dataclear.max()-dataclear.min())
        a=np.array(d.drop(columns=['Close']))
        test = nn.predict(a)
        hasil= test*(dataclear['Close'].max()-dataclear['Close'].min())+dataclear['Close'].min()
        
        if volume==0 or high==0 or open==0 or low==0:
            hasil='Gold Market Closed'
        elif volume<0 or high<0 or open<0 or low<0:
            hasil='Harga tidak mungkin bernilai negatif'
        elif high<low:
            hasil='High harus lebih besar dari low'
        elif open<low:
            hasil='Open harus lebih besar atau sama dengan low'
        elif open>high:
            hasil='Open harus lebih kecil atau sama dengan high'
        else:
            hasil = round(float(hasil[0][0]), 1)
        
        return render_template('predict.html',hasil=hasil, nmse_test=nmse_test)
    return render_template('predict.html')

if __name__ == "__main__":
    app.run(debug=True)

