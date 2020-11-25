# pylint: disable=all
from django.shortcuts import render, redirect
from django.http import HttpResponse
from django.contrib import messages

#Import the libraries
import math
import pandas_datareader as web
import numpy as np
import pandas as pd
from sklearn.preprocessing import MinMaxScaler
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, LSTM
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import datetime as dt
import matplotlib.dates as d  
from pandas import DataFrame
from numpy import array

# Create your views here.


def companyStock(request):
    company = request.GET['company']
    ticker = search(company)
    if(ticker == None):
        messages.error(request, 'Sorry, we are unable to predict for requested company. Please try again.')
        return redirect('home')
    else:
        x,y = model(ticker)
        PriceTom, day, SP, SPD, BP, BPD = graph(x,y)
        Table = table(x,y)
        context = {'PriceTom':PriceTom, 'day':day, 'SP':SP, 'SPD':SPD, 'BP':BP, 'BPD':BPD, 'Table':Table,  "companynm":company}
        return render(request, 'companyStock.html', context)

def index(request):
    return render(request, 'home.html')

def predictStocks(request):
    return render(request,'predictStocks.html')

def tradingtime(request):
    return render(request,'tradingtime.html')

def contact(request):
    return render(request,'contact.html')

def about(request):
    return render(request,'about.html')

def loader(request):
    return render(request,'loader.html')



def search(company):
  name = company
  name = name.lower()
  name = name.strip()
  clist = {
      'google': 'GOOG',
      'apple': 'AAPL',
      'microsoft': 'MSFT',
      'facebook': 'FB',
      'wipro': 'WIT',
      'infosys': 'INFY',
      'qualcomm': 'QCOM'
  }
  return clist.get(name)

def isholiday(given_date):
    # 0: "MONDAY",
    # 1: "TUESDAY",
    # 2: "WEDNESDAY",
    # 3: "THURSDAY",
    # 4: "FRIDAY",
    # 5: "SATURDAY",
    # 6: "SUNDAY"
    return given_date == 5 or given_date == 6

def model(ticker):
  df = web.DataReader(ticker, data_source='yahoo')
  data = df.filter(['Close'])
  dataset=data.values
  training_data_len=math.ceil(len(dataset)*.8)

  scaler=MinMaxScaler(feature_range=(0,1))
  scaled_data = scaler.fit_transform(dataset)

  train_data=scaled_data[0:training_data_len, :]
  x_train=[]
  y_train=[]
  for i in range(60, len(train_data)):
    x_train.append(train_data[i-60:i, 0])
    y_train.append(train_data[i, 0])

  x_train, y_train = np.array(x_train), np.array(y_train)

  x_train = np.reshape(x_train, (x_train.shape[0], x_train.shape[1],1))

  # Model training
  model=Sequential()
  model.add(LSTM(100, return_sequences=True, input_shape=(x_train.shape[1],1)))
  model.add(LSTM(100, return_sequences=False))
  model.add(Dense(25))
  model.add(Dense(1))

  model.compile(optimizer='adam', loss='mean_squared_error')

  model.fit(x_train,y_train, batch_size=30, epochs=10)

  # Model testing 
  test_data=scaled_data[training_data_len - 60: , :]
  x_test=[]
  y_test=dataset[training_data_len:, :]
  for i in range(60,len(test_data)):
    x_test.append(test_data[i-60:i, 0])

  x_test=np.array(x_test)

  x_test= np.reshape(x_test, (x_test.shape[0], x_test.shape[1], 1))

  predictions=model.predict(x_test)
  predictions = scaler.inverse_transform(predictions)

  rmse = np.sqrt( np.mean( predictions - y_test)**2 )

  # Prediction
  valid = data[training_data_len:]
  valid['Predictions'] = predictions

  x_input = test_data[(len(test_data)-60):].reshape(1,-1)
  temp_input=list(x_input)
  temp_input=temp_input[0].tolist()

  

  lst_output=[]
  n_steps=60

  # x and y coordinate array

  # x coordinates
  x = []
  now = valid.last_valid_index() + dt.timedelta(days=1)
  i = 0
  count = 0
  while count < 5:
    current = now + dt.timedelta(days = i)
    if not isholiday(current.weekday()):
        x.append((current, i))
        count += 1
    i += 1
  
  # y coordinates
  j=0
  while j < i:
    if(len(temp_input)>60):
      x_input=np.array(temp_input[1:])
      x_input=x_input.reshape(1,-1)
      x_input=x_input.reshape((1, n_steps, 1))
      yhat=model.predict(x_input)
      temp_input.extend(yhat[0].tolist())
      temp_input=temp_input[1:]
      lst_output.extend(yhat.tolist())
      j=j+1
    else:
      x_input=x_input.reshape((1, n_steps, 1))
      yhat=model.predict(x_input)
      temp_input.extend(yhat[0].tolist())
      lst_output.extend(yhat.tolist())
      j=j+1

 
  # get the corresponding 5 y coordinates w.r.t selected x coordinates
  next5 = scaler.inverse_transform(
    np.reshape(
      np.array(
        [lst_output[i[1]] for i in x]
      ), 
      (5, 1)
    )
  )
  x = [i[0] for i in x]
  y = next5
  #print(x, y, sep = '\n')
  return (x,y)

def graph(x,y):
  # Prediction plot 
  plt.style.use("dark_background")
  for param in ['text.color', 'axes.labelcolor', 'xtick.color', 'ytick.color']:
    plt.rcParams[param] = '0.9'  # very light grey
  for param in ['figure.facecolor', 'axes.facecolor', 'savefig.facecolor']:
    plt.rcParams[param] = '#212946'  # bluish dark grey
  fig = plt.figure(figsize=(10,5))
  plt.gca().xaxis.set_major_formatter(d.DateFormatter('%Y-%m-%d'))
  plt.gca().xaxis.set_major_locator(d.DayLocator(interval=1))
  plt.title('Prediction',fontsize=30, color='#f6efa6')
  plt.xlabel('Date', fontsize=18)
  plt.ylabel('Close Price', fontsize=18)
  plt.plot(x, y, '-ok', color='#3BFEB8')
  n_max = y.argmax()
  plt.plot(x[n_max],y[n_max],'o', markersize=10, color='#F0115F')
  n_min = y.argmin()
  plt.plot(x[n_min],y[n_min],'o',markersize=10, color='yellow')
  plt.legend( ["Prediction","Best Selling Price", "Best Buying Price"])
  
  # Redraw the graph with low alpha and slighty increased linewidth to make it glow:
  n_shades = 10
  diff_linewidth = 1.5
  alpha_value = 0.3 / n_shades
  for n in range(1, n_shades+1):
    plt.plot(x, y, '-ok',linewidth=2+(diff_linewidth*n),alpha=alpha_value,color='#3BFEB8')
  
  plt.savefig('PredictApp\static\Image\prediction.png', dpi=300, bbox_inches='tight')

  max_y = round(max(y)[0], 2)  
  max_x = x[y.argmax()]  
  min_y = round(min(y)[0], 2)
  min_x = x[y.argmin()]

  if x[0].day_name()=='Monday':
    day='PRICE ON MONDAY: $ '
  else:
    day='PRICE TOMORROW: $ '

  PriceTom=round(y[0][0], 2)
  SP=max_y
  SPD=max_x.date()
  BP=min_y
  BPD=min_x.date()

  plt.close()
  return (PriceTom,day,SP,SPD,BP,BPD)

def table(x,y):
  price=[]
  for i in range(len(y)):
   price.append(y[i][0])
  p = [round(x, 2) for x in price]
  Data = {'DATE': x, 'PRICE': p}
  table = DataFrame(Data,columns=['DATE','PRICE'])
  return table



