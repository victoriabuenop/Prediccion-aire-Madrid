import streamlit as st
import datetime
from tensorflow import keras
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from statsmodels.tsa.deterministic import Fourier
from sklearn.preprocessing import MinMaxScaler

archivos = { "Paseo de la Castellana": {"aire":"./data/aire_48_data.csv",
                    "1": "./modelos/48_modelo_1_dia.h5",
                    "2": "./modelos/48_modelo_2_dia.h5",
                    "3": "./modelos/48_modelo_3_dia.h5",
                    "4": "./modelos/48_modelo_4_dia.h5",
                    "5": "./modelos/48_modelo_5_dia.h5"},
            "Plaza del Carmen": {"aire":"./data/aire_35_data.csv",
                    "1": "./modelos/35_modelo_1_dia.h5",
                    "2": "./modelos/35_modelo_2_dia.h5",
                    "3": "./modelos/35_modelo_3_dia.h5",
                    "4": "./modelos/35_modelo_4_dia.h5",
                    "5": "./modelos/35_modelo_5_dia.h5"},
            "Retiro": {"aire":"./data/aire_49_data.csv",
                    "1": "./modelos/49_modelo_1_dia.h5",
                    "2": "./modelos/49_modelo_2_dia.h5",
                    "3": "./modelos/49_modelo_3_dia.h5",
                    "4": "./modelos/49_modelo_4_dia.h5",
                    "5": "./modelos/49_modelo_5_dia.h5"}}

config = {"1": {"window": 7, "fourier": False},
        "2": {"window": 7, "fourier": False},
        "3": {"window": 14, "fourier": False},
        "4": {"window": 7, "fourier": False},
        "5": {"window": 7, "fourier": False}}

meses = [0, 31, 28, 31, 30, 31, 30]

def datasetCreate(fourier=bool, aire=str):
    aire_data = pd.read_csv(aire)
    meteo_data = pd.read_csv('./data/meteo_data.csv')

    aire_data['DATE']=pd.to_datetime(aire_data['DATE'])
    meteo_data['DATE']=pd.to_datetime(meteo_data['DATE'])

    data = pd.merge(aire_data, meteo_data, on=['DATE'])
    data = data.replace('Ip', '0.0')
    data['NO2'] = data['NO2'].astype(float)
    data['PRECIPITACION'] = data['PRECIPITACION'].astype(float)
    data['DATE']=pd.to_datetime(data['DATE'])
    data.index= data['DATE']

    data = data.drop(data.columns[[0]], axis=1)

    if fourier:
        #Fourier(period, order)
        fourier_gen = Fourier(365, order=2)
        #2 PAREJAS sin y cos
        fourier_data=fourier_gen.in_sample(data.index)
        # se basa en los datos de la serie para generar valores de senos y cosenos
        #Sería un dataframe de 4200 filas y 4 columnas (orden 2: 2 parejas de senos y cosenos)

        data = pd.merge(data, fourier_data, left_index=True, right_index=True)
    
    return data

def dataPreparation(dataset, fourier, window, horizon, display):
        x_scaler = MinMaxScaler()
        y_scaler = MinMaxScaler()
        
        if fourier:
            data = x_scaler.fit_transform(dataset[['NO2', 'TMEDIA', 'PRECIPITACION', 'VELMEDIA', 'sin(1,365)', 'cos(1,365)', 'sin(2,365)', 'cos(2,365)']]) 
        else:
            data = x_scaler.fit_transform(dataset[['NO2', 'TMEDIA', 'PRECIPITACION', 'VELMEDIA']])

        indices = range(display, window+display)
        input_data = [(data[indices])]
        input_data = np.array(input_data)

        indice_y = range(window+horizon-1+display, window+horizon+display)
        expected_value = [(dataset['NO2'][indice_y])]


        return input_data, x_scaler, y_scaler, expected_value

header = st.container()
layout = st.container()

with header:
    st.title('Predicción de contaminación del aire en Madrid')

with layout:
    image_col, sel_col = st.columns(2)
    image_col.image('./data/madrid.png', caption='Mapa de la ciudad de Madrid con varias estaciones de control de calidad del aire')

    sel_col.text("")
    sel_col.text("")
    sel_col.text("")
    station = sel_col.selectbox('Elija una estación de control:', options=['Paseo de la Castellana','Plaza del Carmen','Retiro'], index=0)
    sel_col.text("")

    date = sel_col.date_input('Elija la fecha actual:', value=datetime.date(2023, 1, 14), min_value=datetime.date(2023, 1, 14), max_value=datetime.date(2023, 6, 25))
    #horizon = sel_col.slider('¿A cuántos días vista desea hacer la predicción?', min_value=1, max_value=5, step=1)
    sel_col.text("")

    checked = sel_col.button('Obtener predicción')
    sel_col.text("")

    day = date.day
    month = date.month

    if checked:
        n_dias = 0
        for m in range(0,month):
             n_dias += meses[m]
        n_dias += day
        
        st.write('Predicción de los próximos 5 días para la estación ' +str(station) + ':')
        predictions = []

        for i in range(1,6):
             
            model = keras.models.load_model(archivos[station][str(i)], compile=False)
            dataset = datasetCreate(fourier=config[str(i)]["fourier"], aire=archivos[station]["aire"])
            input_data, x_scaler, y_scaler, expected_value = dataPreparation(dataset=dataset, fourier=config[str(i)]["fourier"], window=config[str(i)]["window"], horizon=i, display=n_dias-config[str(i)]["window"])

            y_data = y_scaler.fit_transform(dataset[['NO2']])

            prediction = model.predict(input_data)
            prediction_inv = y_scaler.inverse_transform(prediction)

            predictions.append(round(prediction_inv[0][0], 1))

            #sel_col.write('Día vista ' +  str(i) + ': ' + str(round(prediction_inv[0][0], 2)) +' µg/m³') 
            #sel_col.write('(El valor real fue: ' + str(expected_value[0][0]) + ' µg/m³)')
        

        data = pd.read_csv(archivos[station]["aire"])
        real = data[n_dias-14: n_dias+5]
        real = real.reset_index()
        pred = data[n_dias-1: n_dias+5]
        pred = pred.reset_index()
        predictions.insert(0, real._get_value(13, 'NO2'))
        p = pd.DataFrame(predictions)
        pred['Prediccion'] = p.iloc[0:13].values
        pred = pred.round({'Prediccion':1})
       
        fig, ax = plt.subplots(figsize=(15, 8))
        ax.plot(real['DATE'], real['NO2'], lw= 3, alpha=1, color='dodgerblue', label='Valores reales')
        ax.plot(pred['DATE'], pred['Prediccion'], lw= 3, alpha=1, color='darkorange', label='Predicción')
        ax.axvline(x=pred['DATE'][0], ls=':', lw=3, alpha=0.8, color='grey')
        ax.fill_between(pred['DATE'], real['NO2'][13:], pred['Prediccion'], alpha=0.2, color='darkorange')
        ax.scatter(pred['DATE'], pred['Prediccion'], s=200, lw=3, color='darkorange', marker='o')
        ax.scatter(real['DATE'], real['NO2'], s=200, lw=3, color='dodgerblue', marker='o', fc='w', zorder=5)

        for index in range(len(real['DATE'])):
             ax.text(real['DATE'][index], real['NO2'][index]+1, real['NO2'][index], size=14)
        
        for index in range(1, len(pred['DATE'])):
             ax.text(pred['DATE'][index], pred['Prediccion'][index]+1, pred['Prediccion'][index], size=14)
        
        ax.set_xticklabels(real['DATE'], rotation=70, size=13)
        ax.set_yticklabels(real['NO2'], size=13)
        ax.set_ylabel('NO₂ (µg/m³)', size=13)
        ax.legend(fontsize=14)
        st.pyplot(fig)
        