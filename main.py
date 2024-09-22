from flask import Flask, flash,render_template,request
import numpy as np
import csv
import joblib
import pandas as pd
from plotly.subplots import make_subplots
import plotly.graph_objects as go
import plotly
import  random

app = Flask(__name__)

state_list = ['Andaman and Nicobar Islands', 'Andhra Pradesh', 'Arunachal Pradesh', 'Assam', 'Bihar', 'Chandigarh', 'Chhattisgarh', 'Dadra and Nagar Haveli', 'Goa', 'Gujarat', 'Haryana', 'Himachal Pradesh', 'Jammu and Kashmir ', 'Jharkhand', 'Karnataka', 'Kerala', 'Madhya Pradesh', 'Maharashtra', 'Manipur', 'Meghalaya', 'Mizoram', 'Nagaland', 'Odisha', 'Puducherry', 'Punjab', 'Rajasthan', 'Sikkim', 'Tamil Nadu', 'Telangana ', 'Tripura', 'Uttar Pradesh', 'Uttarakhand', 'West Bengal']
state_dict = {'Andaman and Nicobar Islands':1,'Andhra Pradesh':2,'Arunachal Pradesh':3,'Assam':4,'Bihar':5,'Chandigarh':6,'Chhattisgarh':7,
'Dadra and Nagar Haveli':8,'Goa':9,'Gujarat':10,'Haryana':11,'Himachal Pradesh':12,'Jammu and Kashmir ':13,'Jharkhand':14,'Karnataka':15,
'Kerala':16,'Madhya Pradesh':17,'Maharashtra':18,'Manipur':19,'Meghalaya':20,'Mizoram':21,'Nagaland':22,'Odisha':23,'Puducherry':24,
'Punjab':25,'Rajasthan':26,'Sikkim':27,'Tamil Nadu':28,'Telangana ':29,'Tripura':30,'Uttar Pradesh':31,'Uttarakhand':32,'West Bengal':33}

season_list = ['Kharif','Whole Year','Autumn','Rabi','Winter','Summer']
season_dict = {'Kharif':1, 'Whole Year':2, 'Autumn':3, 'Rabi':4, 'Winter':5, 'Summer':6}

crop_list = ['Arecanut','Other 1 Pulses','Rice', 'Banana', 'Cashewnut', 'Coconut', 'Dry Ginger', 'Sugar Cane', 'Sweet Potato','Topioca', 'Black Pepper', 'Dry Chillies','Other Oilseeds', 'Turmeric', 'Groundnut', 'Maize', 'Moong(Green gan)', 'Urad', 'Sun Flower', 'Bajna', 'Cotton', 'Ragi', 'Tobacco']
crop_dict = {'Arecanut':1,'Other 1 Pulses':2,'Rice':3, 'Banana':4, 'Cashewnut':5, 'Coconut':6, 'Dry Ginger':7, 'Sugar Cane':8, 'Sweet Potato':9,'Topioca':10, 'Black Pepper':11, 'Dry Chillies':12, 'Other Oilseeds':13, 'Turmeric':14, 'Groundnut':15, 'Maize':16, 'Moong(Green gan)':17, 'Urad':18, 'Sun Flower':19, 'Bajna':20, 'Cotton':21, 'Ragi':22, 'Tobacco':23}

soil_list = ['Chalky', 'Clay', 'Loamy', 'Sandy', 'Silty']
soil_dict = {'Chalky':1, 'Clay':2, 'Loamy':3, 'Sandy':4, 'Silty':5}


@app.route('/')
@app.route('/home')
def home():
    return render_template('index.html')
@app.route('/yield')
def newf():
    return render_template('new.html', data = state_list, data1 = crop_list, data2 = soil_list, data3 = season_list)

@app.route('/predict')
def predict():
    return render_template('predict.html', data2 = soil_list)

@app.route('/result',methods=['POST','GET'])
def result():


    model = joblib.load('lightgbm_.pkl')

    classes = ['Paddy', 'Cholam', 'Cumbu', 'Ragi','Cotton', 'Sugarcane','Chilli', 'Pigeon Pea', 
                'Coconut', 'Tobacco', 'Onion', 'Banana','Mangoes', 'Turmeric', 'Groundnut', 'BlackGram', 
                'Maize', 'Tapioca','Tomoto', 'Brinjal', 'Carrot', 'Beans']

    values = []
    if request.method == 'POST':
        values.append(float(request.form.get('nitrogen')))
        values.append(float(request.form.get('phosphorous')))
        values.append(float(request.form.get('potassium')))
        values.append(float(request.form.get('temperature')))
        values.append(float(request.form.get('humidity')))
        values.append(float(request.form.get('ph')))
        values.append(float(request.form.get('rainfall')))

        # answer = model.predict([values])
        predict_pro = model.predict_proba([values])
        list_proba = []
        for i in [-1, -2, -3, -4, -5]:
            list_proba.append(classes[np.argsort(np.max(predict_pro, axis=0))[i]])
        # print(list_proba)
        return render_template('result.html',probab = list_proba)

@app.route('/analysis')
def analysis():
    df = pd.read_csv('data.csv')
    def intractive_plot(df, feature, name):
        colorarr = ['#0592D0','#Cd7f32', '#E97451', '#Bdb76b', '#954535', '#C2b280', '#808000','#C2b280', '#E4d008', '#9acd32', '#Eedc82', '#E4d96f',
               '#32cd32','#39ff14','#00ff7f', '#008080', '#36454f', '#F88379', '#Ff4500', '#Ffb347', '#A94064', '#E75480', '#Ffb6c1', '#E5e4e2',
               '#Faf0e6', '#8c92ac', '#Dbd7d2','#A7a6ba', '#B38b6d']

        df_label = pd.pivot_table(df, index=['label'], aggfunc='mean')
        df_label_feature = df_label.sort_values(by=feature, ascending = False)

        fig = make_subplots(rows = 1, cols = 2)

        top = {

            'y': df_label_feature[feature][:10].sort_values().index,
            'x': df_label_feature[feature][:10].sort_values()
        }
        last = {

            'y': df_label_feature[feature][-10:].sort_values().index,
            'x': df_label_feature[feature][-10:].sort_values()
        }

        fig.add_trace(
            go.Bar(top,
                   name='Least {} Needed'.format(name),
                   marker_color = random.choice(colorarr),
                   orientation = 'h',
                   text = top['x']
                  ),
            row = 1, col = 1
        )
        fig.add_trace(
            go.Bar(last,
                   name='Least {} Needed'.format(name),
                   marker_color = random.choice(colorarr),
                   orientation = 'h',
                   text = top['x']
                  ),
            row = 1, col = 2
        )

        fig.update_traces(texttemplate = '%{text}', textposition = 'inside')
        fig.update_layout(title_text = name,
                          plot_bgcolor = 'white',
                          font_size = 12,
                          font_color = 'black',
                          height = 500
                         )
        fig.update_xaxes(showgrid = False)
        fig.update_yaxes(showgrid = False)
        fig.show()
        
    intractive_plot(df, feature = 'N', name = 'Nitrogen')
    intractive_plot(df, feature = 'P', name = 'Phosphorous')
    intractive_plot(df, feature = 'K', name = 'Potassium')
    intractive_plot(df, feature = 'humidity', name = 'Humidity')
    intractive_plot(df, feature = 'temperature', name = 'Temperature')
    intractive_plot(df, feature = 'ph', name = 'ph')
    intractive_plot(df, feature = 'rainfall', name = 'Rainfall')
    return render_template('predict.html')
@app.route('/predicted', methods = ['POST','GET'])
def predicted():
    if request.method == 'POST':
        data = []
        model = joblib.load('lr.pkl')
        state = request.form.get('state')
        d = state_dict[state]
        data.append(d)
        season = request.form.get('season')
        d1 = season_dict[season]
        data.append(d1)
        crop = request.form.get('crop')
        d2 = crop_dict[crop]
        data.append(d2)
        data.append(request.form.get('area'))

        pred = model.predict([data])
        print(pred)
        return render_template('new.html', msg ='Production is : '+str(pred[0]))


@app.route('/real')
def real():
    return render_template('real.html')

@app.route('/l1')
def l1():
    l1 = [['sandy',68.6,2.2,50,33,60,0,67]]
    return render_template('predict1.html',data=l1,msg='Land 1')

@app.route('/l2')
def l2():
    l2 = [['Chalky',75.6,3,58.3,134,141,0,270]]
    return render_template('predict1.html',data=l2,msg='Land 2')

@app.route('/l3')
def l3():
    l3 = [['Clay',71.4,4,70.8,33,62,0,81]]
    return render_template('predict1.html',data=l3,msg='Land 3')

@app.route('/l4')
def l4():
    l4 = [['Loamy',62.2,4.2,66.6,35,59,0,110]]
    return render_template('predict1.html',data=l4,msg='Land 4')

@app.route('/Paddy-details')
def paddy_detail():
    return render_template('paddy_details.html')

@app.route('/Paddy-disease')
def paddy_disease():
    return render_template('paddy_disease.html')

@app.route('/Paddy-fertilizer')
def paddy_ferti():
    return render_template('paddy_ferti.html')

@app.route('/Cholam-details')
def cholam_detail():
    return render_template('cholam_details.html')

@app.route('/Cholam-disease')
def cholam_disease():
    return render_template('cholam_disease.html')

@app.route('/Cholam-fertilizer')
def cholam_ferti():
    return render_template('cholam_ferti.html')

@app.route('/Cumbu-details')
def cumbu_detail():
    return render_template('cumbu_details.html')

@app.route('/Cumbu-disease')
def cumbu_disease():
    return render_template('cumbu_disease.html')

@app.route('/Cumbu-fertilizer')
def cumbu_ferti():
    return render_template('cumbu_ferti.html')

@app.route('/Ragi-details')
def ragi_detail():
    return render_template('ragi_details.html')

@app.route('/Ragi-disease')
def ragi_disease():
    return render_template('ragi_disease.html')

@app.route('/Ragi-fertilizer')
def ragi_ferti():
    return render_template('ragi_ferti.html')

@app.route('/Cotton-details')
def cotton_detail():
    return render_template('cotton_details.html')

@app.route('/Cotton-disease')
def cotton_disease():
    return render_template('cotton_disease.html')

@app.route('/Cotton-fertilizer')
def cotton_ferti():
    return render_template('cotton_ferti.html')

@app.route('/Sugarcane-details')
def sugarcane_detail():
    return render_template('sugarcane_details.html')

@app.route('/Sugarcane-disease')
def sugarcane_disease():
    return render_template('sugarcane_disease.html')

@app.route('/Sugarcane-fertilizer')
def sugarcane_ferti():
    return render_template('sugarcane_ferti.html')

@app.route('/Chilli-details')
def chilli_detail():
    return render_template('chilli_details.html')

@app.route('/Chilli-disease')
def chilli_disease():
    return render_template('chilli_disease.html')

@app.route('/Chilli-fertilizer')
def chilli_ferti():
    return render_template('chilli_ferti.html')

@app.route('/Pigeon Pea-details')
def pigeon_detail():
    return render_template('pigeon_details.html')

@app.route('/Pigeon Pea-disease')
def pigeon_disease():
    return render_template('pigeon_disease.html')

@app.route('/Pigeon Pea-fertilizer')
def pigeon_ferti():
    return render_template('pigeon_ferti.html')

@app.route('/Coconut-details')
def coconut_detail():
    return render_template('coconut_details.html')

@app.route('/Coconut-disease')
def coconut_disease():
    return render_template('coconut_disease.html')

@app.route('/Coconut-fertilizer')
def coconut_ferti():
    return render_template('coconut_ferti.html')

@app.route('/Tobacco-details')
def tobacco_detail():
    return render_template('tobacco_details.html')

@app.route('/Tobacco-disease')
def tobacco_disease():
    return render_template('tobacco_disease.html')

@app.route('/Tobacco-fertilizer')
def tobacco_ferti():
    return render_template('tobacco_ferti.html')

@app.route('/Onion-details')
def onion_detail():
    return render_template('onion_details.html')

@app.route('/Onion-disease')
def onion_disease():
    return render_template('onion_disease.html')

@app.route('/Onion-fertilizer')
def onion_ferti():
    return render_template('onion_ferti.html')

@app.route('/Banana-details')
def banana_detail():
    return render_template('banana_details.html')

@app.route('/Banana-disease')
def banana_disease():
    return render_template('banana_disease.html')

@app.route('/Banana-fertilizer')
def banana_ferti():
    return render_template('banana_ferti.html')

@app.route('/Mangoes-details')
def mango_detail():
    return render_template('mango_details.html')

@app.route('/Mangoes-disease')
def mango_disease():
    return render_template('mango_disease.html')

@app.route('/Mangoes-fertilizer')
def mango_ferti():
    return render_template('mango_ferti.html')

@app.route('/Turmeric-details')
def termeric_detail():
    return render_template('termeric_details.html')

@app.route('/Turmeric-disease')
def termeric_disease():
    return render_template('termeric_disease.html')

@app.route('/Turmeric-fertilizer')
def termeric_ferti():
    return render_template('termeric_ferti.html')

@app.route('/Groundnut-details')
def ground_detail():
    return render_template('ground_details.html')

@app.route('/Groundnut-disease')
def ground_disease():
    return render_template('ground_disease.html')

@app.route('/Groundnut-fertilizer')
def ground_ferti():
    return render_template('ground_ferti.html')

@app.route('/BlackGram-details')
def black_detail():
    return render_template('black_details.html')

@app.route('/BlackGram-disease')
def black_disease():
    return render_template('black_disease.html')

@app.route('/BlackGram-fertilizer')
def black_ferti():
    return render_template('black_ferti.html')

@app.route('/Maize-details')
def maize_detail():
    return render_template('maize_details.html')

@app.route('/Maize-disease')
def maize_disease():
    return render_template('maize_disease.html')

@app.route('/Maize-fertilizer')
def maize_ferti():
    return render_template('maize_ferti.html')

@app.route('/Tapioca-details')
def topi_detail():
    return render_template('topi_details.html')

@app.route('/Tapioca-disease')
def topi_disease():
    return render_template('topi_disease.html')

@app.route('/Tapioca-fertilizer')
def topi_ferti():
    return render_template('topi_ferti.html')

@app.route('/Tomoto-details')
def tomoto_detail():
    return render_template('tomoto_details.html')

@app.route('/Tomoto-disease')
def tomoto_disease():
    return render_template('tomoto_disease.html')

@app.route('/Tomoto-fertilizer')
def tomoto_ferti():
    return render_template('tomoto_ferti.html')

@app.route('/Brinjal-details')
def brinjal_detail():
    return render_template('brin_details.html')

@app.route('/Brinjal-disease')
def brinjal_disease():
    return render_template('brin_disease.html')

@app.route('/Brinjal-fertilizer')
def brinjal_ferti():
    return render_template('brin_ferti.html')

@app.route('/Carrot-details')
def carrot_detail():
    return render_template('carrot_details.html')

@app.route('/Carrot-disease')
def carrot_disease():
    return render_template('carrot_disease.html')

@app.route('/Carrot-fertilizer')
def carrot_ferti():
    return render_template('carrot_ferti.html')

@app.route('/Beans-details')
def bean_detail():
    return render_template('bean_details.html')

@app.route('/Beans-disease')
def bean_disease():
    return render_template('bean_disease.html')

@app.route('/Beans-fertilizer')
def bean_ferti():
    return render_template('bean_ferti.html')

if __name__ == '__main__':
    app.run(debug=True)