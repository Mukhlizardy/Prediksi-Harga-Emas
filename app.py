import re
import pandas as pd
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
@app.route("/predict", methods=['GET', 'POST'])
def predict():
    hasil = None
    error = None

    if request.method == 'POST':
        if 'file' not in request.files:
            error = 'File tidak ditemukan.'
            return render_template('predict.html', error=error)

        file = request.files['file']

        if file.filename == '':
            error = 'Tidak ada file yang dipilih.'
            return render_template('predict.html', error=error)

        try:
            df = pd.read_excel(file)
            required_columns = ['Open', 'High', 'Low', 'Volume']
            if not all(col in df.columns for col in required_columns):
                error = f"Kolom Excel harus mengandung: {', '.join(required_columns)}"
                return render_template('predict.html', error=error)

            pred_results = []

            for idx, row in df.iterrows():
                open_, high, low, volume = row['Open'], row['High'], row['Low'], row['Volume']

                if any(v <= 0 for v in [open_, high, low, volume]):
                    pred = 'Data tidak valid (nilai ≤ 0)'
                elif high < low:
                    pred = 'High harus lebih besar dari Low'
                elif open_ < low:
                    pred = 'Open harus ≥ Low'
                elif open_ > high:
                    pred = 'Open harus ≤ High'
                else:
                    q = pd.DataFrame([[open_, high, low, volume]], columns=required_columns)
                    d = (q - dataclear.min()) / (dataclear.max() - dataclear.min())
                    a = np.array(d[['Open', 'High', 'Low', 'Volume']])
                    pred_val = nn.predict(a)
                    pred = round(float(pred_val[0][0] * (dataclear['Close'].max() - dataclear['Close'].min()) + dataclear['Close'].min()), 1)

                pred_results.append(pred)

            df['Prediksi Close'] = pred_results
            hasil = df.to_html(classes='table table-striped', index=False)

        except Exception as e:
            error = f'Terjadi kesalahan: {str(e)}'

    return render_template('predict.html', hasil=hasil, error=error, nmse_test=nmse_test if 'nmse_test' in globals() else None)

if __name__ == "__main__":
    app.run(debug=True)

