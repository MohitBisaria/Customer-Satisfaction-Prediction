# -*- coding: utf-8 -*-
"""
Created on Sun Aug 21 11:24:18 2022

@author: User
"""
import flask
import joblib
from flask import Flask, jsonify, request
import pandas as pd
import pandas as pd
import numpy as np
import warnings
warnings.filterwarnings("ignore")
import re
import nltk
nltk.download('stopwords')
nltk.download('rslp')
import random
import seaborn as sns
from sklearn.cluster import KMeans
#from sklearn.externals import joblib
import matplotlib.pyplot as plt
#import pyplot
from scipy.stats import randint as sp_randint
from nltk.corpus import stopwords
from nltk.stem import RSLPStemmer
from tqdm import tqdm
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.preprocessing import Normalizer
from scipy.sparse import hstack
from lightgbm import LGBMClassifier
from xgboost import XGBClassifier
from sklearn.linear_model import SGDClassifier,LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import log_loss,accuracy_score, confusion_matrix, f1_score
from sklearn.model_selection import RandomizedSearchCV,GridSearchCV
from sklearn.ensemble import RandomForestClassifier, StackingClassifier, VotingClassifier, AdaBoostClassifier

app = Flask(__name__)


def clean_data(data):
    
    #list_text = []
    
    portuguese_stop = stopwords.words('portuguese') # portugese language stopwords
    #stemmer_1 = RSLPStemmer() # portugese language stemmer
    
    links_alphabet = 'http[s]?://(?:[a-zA-Z]|[0-9]|[$-_@.&+]|[!*\(\),]|(?:%[0-9a-fA-F][0-9a-fA-F]))+' # check for hyperlinks
    numerical_dates = '([0-2][0-9]|(3)[0-1])(\/|\.)(((0)[0-9])|((1)[0-2]))(\/|\.)\d{2,4}' # check for dates
    currency_symbols = '[R]{0,1}\$[ ]{0,}\d+(,|\.)\d+' # check for currency symbols
    
    if data is None:
        return ''
    else:
        if isinstance(data,str):
            data = re.sub('\n', ' ', data) # remove new lines
            data = re.sub(links_alphabet, ' URL ', data) # remove hyperlinks
            data= re.sub(numerical_dates, ' ', data) # remove dates
            data = re.sub(currency_symbols, ' dinheiro ', data) # remove currency symbols
            data = re.sub('[0-9]+', ' numero ', data) # remove digits
            data = re.sub('([nN][ãÃaA][oO]|[ñÑ]| [nN] )', ' negação ', data) # replace no with negative
            data = re.sub('\W', ' ', data) # remove extra whitespaces
            data = re.sub('\s+', ' ', data) # remove extra spaces
            data = re.sub('[ \t]+$', '', data) # remove tabs etc.
            data = ' '.join(e for e in data.split() if e.lower() not in portuguese_stop) # remove stopwords
            #text = ' '.join(stemmer.stem(e.lower()) for e in text.split()) # stemming the words
            #list_text.append(data.lower().strip())
        
    return data
 

def preprocess(data):
  df_olist_orders_data=pd.read_excel(data,sheet_name='olist_orders_data')
  
  df_olist_orders_data.drop(['Unnamed: 0'],axis=1,inplace=True)
  
  df_olist_order_reviews_data=pd.read_excel(data,sheet_name='olist_order_reviews_dataset')
  
  df_olist_order_reviews_data.drop(['Unnamed: 0'],axis=1,inplace=True)
  
  df_olist_customers_data=pd.read_excel(data,sheet_name='olist_customers_data')
  
  df_olist_customers_data.drop(['Unnamed: 0'],axis=1,inplace=True)
  
  df_olist_order_items_dataset=pd.read_excel(data,sheet_name='olist_order_items_dataset')
  
  #df_olist_order_items_dataset.drop(['Unnamed: 0'],axis=1,inplace=True)
  
  df_olist_products_dataset=pd.read_excel(data,sheet_name='olist_products_dataset')
  
  df_olist_products_dataset.drop(['Unnamed: 0'],axis=1,inplace=True)
  
  df_olist_order_payments_dataset=pd.read_excel(data,sheet_name='olist_order_payments_dataset')
  
  df_olist_order_payments_dataset.drop(['Unnamed: 0'],axis=1,inplace=True)
  
  df_product_category_name=pd.read_excel(data,sheet_name='product_category_name')
  
  df_product_category_name.drop(['Unnamed: 0'],axis=1,inplace=True)
  #df_olist_sellers_dataset=pd.read_excel(data,sheet_name='olist_sellers_dataset')
  #df_olist_geolocation_dataset=pd.read_excel(data,sheet_name='olist_geoloction_datset')
  
  df_olist_order_reviews_data=df_olist_order_reviews_data[['order_id','review_score','review_comment_message']]
  
  df_order_review=df_olist_order_reviews_data.merge(df_olist_orders_data,on='order_id')
  df_product_english=pd.merge(df_olist_products_dataset,df_product_category_name,on='product_category_name',how='left')
  df_product_english=df_product_english.drop(labels='product_category_name',axis=1)
  df_product_item=pd.merge(df_olist_order_items_dataset,df_product_english,on='product_id')
  df_product_review=pd.merge(df_order_review,df_product_item,on='order_id')
  df_product_review_payments=pd.merge(df_product_review,df_olist_order_payments_dataset,on='order_id')
  data_final_predict=pd.merge(df_product_review_payments,df_olist_customers_data,on='customer_id')
  
  data_final_predict['review_score']=data_final_predict['review_score'].apply(lambda x:1 if x>3 else 0)
  
  list=['product_name_lenght','product_description_lenght','product_photos_qty','product_weight_g','product_length_cm','product_height_cm','product_width_cm']
  
  res={'product_name_lenght': 52.0,'product_description_lenght': 600.0,'product_photos_qty': 1.0,'product_weight_g': 700.0,'product_length_cm': 25.0,'product_height_cm': 13.0,'product_width_cm': 20.0}
  
  for i in list:
      if data_final_predict[i] is None:
          data_final_predict[i].fillna(res.get(i),inplace=True)
          
          #data_final_predict['product_name_lenght'].fillna(data_final_predict['product_name_lenght'].median(),inplace=True)
          #data_final_predict['product_description_lenght'].fillna(data_final_predict['product_description_lenght'].median(),inplace=True)
          #data_final_predict['product_photos_qty'].fillna(data_final_predict['product_photos_qty'].median(),inplace=True)
          #data_final_predict['product_weight_g'].fillna(data_final_predict['product_weight_g'].median(),inplace=True)
          #data_final_predict['product_length_cm'].fillna(data_final_predict['product_length_cm'].median(),inplace=True)
          #data_final_predict['product_height_cm'].fillna(data_final_predict['product_height_cm'].median(),inplace=True)
          #data_final_predict['product_width_cm'].fillna(data_final_predict['product_width_cm'].median(),inplace=True)
  
  ids=data_final_predict[data_final_predict['order_delivered_customer_date'].isnull() == True].index.values
  vals=data_final_predict.iloc[ids]['order_estimated_delivery_date'].values
  data_final_predict.loc[ids,'order_delivered_customer_date']=vals
  ids=data_final_predict[data_final_predict['order_approved_at'].isnull() == True].index.values
  data_final_predict.loc[ids,'order_approved_at']=data_final_predict.iloc[ids]['order_purchase_timestamp'].values
  data_final_predict.drop(labels='order_delivered_carrier_date',axis=1,inplace=True)
  data_final_predict['review_comment_message']=data_final_predict['review_comment_message'].fillna('no_review')
  #data_final_predict=data_final_predict.dropna()
  
  # converting date to datetime and extracting dates from the datetime columns in the data set
  datetime_columns = ['order_purchase_timestamp','order_approved_at','order_delivered_customer_date','order_estimated_delivery_date']
  for columns in datetime_columns:
      data_final_predict[columns] = pd.to_datetime(data_final_predict[columns].dt.date)
      
  # https://www.kaggle.com/andresionek/predicting-customer-satisfaction
  # calculating estimated delivery time
  data_final_predict['estimated_delivery_time'] = (data_final_predict['order_estimated_delivery_date'] - data_final_predict['order_approved_at']).dt.days

  # calculating actual delivery time
  data_final_predict['actual_delivery_time'] = (data_final_predict['order_delivered_customer_date'] - data_final_predict['order_approved_at']).dt.days

  # calculating diff_in_delivery_time
  data_final_predict['diff_in_delivery_time'] = data_final_predict['estimated_delivery_time'] - data_final_predict['actual_delivery_time']

  # finding if delivery was lare
  data_final_predict['on_time_delivery'] = data_final_predict['order_delivered_customer_date'] < data_final_predict['order_estimated_delivery_date']
  data_final_predict['on_time_delivery'] = data_final_predict['on_time_delivery'].astype('int')

  # finding total order cost
  data_final_predict['total_order_cost'] = data_final_predict['price'] + data_final_predict['freight_value']

  # calculating order freight ratio
  data_final_predict['order_freight_ratio'] = data_final_predict['freight_value']/data_final_predict['price']

  # finding the day of week on which order was made
  data_final_predict['purchase_dayofweek'] = pd.to_datetime(data_final_predict['order_purchase_timestamp']).dt.dayofweek

  # adding is_reviewed where 1 is if review comment is given otherwise 0.
  data_final_predict['is_reviewed'] = (data_final_predict['review_comment_message'] != 'no_review').astype('int')    

  data_final_predict.drop(columns=['order_id', 'order_item_id', 'product_id', 'seller_id','shipping_limit_date','customer_id',
                       'order_purchase_timestamp', 'order_approved_at', 'order_delivered_customer_date', 'customer_state',
                       'order_estimated_delivery_date','customer_unique_id', 'customer_city','customer_zip_code_prefix'],
              axis=1,inplace=True)
  
  # selecting features 
  # numerical features
  numerical = ['price', 'freight_value', 'product_name_lenght','product_description_lenght', 'product_photos_qty','product_weight_g','product_length_cm', 'product_height_cm', 'product_width_cm', 'payment_sequential','payment_installments', 'payment_value','on_time_delivery', 'estimated_delivery_time','actual_delivery_time', 'diff_in_delivery_time', 'purchase_dayofweek','total_order_cost', 'order_freight_ratio','is_reviewed']
  # categorical features
  categorical = ['review_comment_message','product_category_name_english','order_status', 'payment_type']
  
  outlier_numerical=['freight_value','product_name_lenght','product_description_lenght','product_photos_qty','product_weight_g','product_length_cm','product_height_cm','product_width_cm','payment_sequential','payment_installments']
  
  #dictionary=[]
  #dup={}
  #for v in range(0,15):
   #for i in outlier_numerical:
    #du=[]
    #du.append(min(data_final_predict[i]))
    #du.append(max(data_final_predict[i]))
    
    #dictionary.append(du)
    
  #for i in outlier_numerical:
   #for v in dictionary:
    #dup[i]=v
    #dictionary.remove(v)
    #break
    
  dup={'price': [0.85, 6735.0], 'freight_value': [0.0, 409.68], 'product_name_lenght': [5.0, 72.0], 'product_description_lenght': [4.0, 3992.0], 'product_photos_qty': [1.0, 20.0], 'product_weight_g': [0.0, 40425.0], 'product_length_cm': [7.0, 105.0], 'product_height_cm': [2.0, 105.0], 'product_width_cm': [6.0, 118.0], 'payment_sequential': [1.0, 29.0],'payment_installments': [0.0, 24.0]}
  for i in outlier_numerical:
    if (data_final_predict.loc[0,i] < dup.get(i)[0]) or (data_final_predict.loc[0,i] > dup.get(i)[1]):
        return {'please enter valid value of '+ i +' in the range':str(dup[i])}
             
          
  
  # https://www.aclweb.org/anthology/W17-6615

  
  clean_process = clean_data(data_final_predict['review_comment_message'].values)
  data_final_predict['review_comment_message'] = clean_process
  # nao_reveja = no_review in portugese
  data_final_predict['review_comment_message'] = data_final_predict['review_comment_message'].replace({'no_review':'nao_reveja'}) 
  # df_final.to_csv('olist_final.csv',index=False)
  
  # Encoding categorical variable payment_type
  data_final_predict['payment_type'] = data_final_predict['payment_type'].replace({'credit_card':1,'boleto':2,'voucher':3,'debit_card':4})
  
  #Exploring geolocation and payment dataset
  #df_geolocation_payments=pd.concat([df_olist_geolocation_dataset,df_olist_order_payments_dataset],axis=1)
  #data_geolocation_payments=df_geolocation_payments.dropna()
  #data_geolocation_payments['geolocation_state']=data_geolocation_payments['geolocation_state'].replace({'SP':1,'RN':2,'AC':3})
  #data_geolocation_payments['payment_type']=data_geolocation_payments['payment_type'].replace({'credit_card':1,'boleto':2,'voucher':3,'debit_card':4})
  #data_geolocation_payments['geolocation_city']=data_geolocation_payments['geolocation_city'].replace({'sao paulo':1,'são paulo':2,'sao bernardo do campo':3,'jundiaí':4,'taboão da serra':5,'sãopaulo':6,'sp':7})
  #data_geolocation_payments.drop(['order_id'],axis=1,inplace=True)
  #data_geolocation_payments.drop(['geolocation_lat'],axis=1,inplace=True)
  #data_geolocation_payments.drop(['geolocation_lng'],axis=1,inplace=True)
  #data_geolocation_payments = data_geolocation_payments[data_geolocation_payments.payment_type != 'not_defined']
 
  #Sum_of_squared_distances = []
  #V = range(1,15)
  #for k in V:
    #kmeans = KMeans(n_clusters=k)
    #kmeans = kmeans.fit(data_geolocation_payments)
    #Sum_of_squared_distances.append(kmeans.inertia_)
    
  #k_means=KMeans(n_clusters=3)
  #k_means.fit(data_geolocation_payments)
  #clusters=k_means.cluster_centers_
  #y_km=k_means.predict(data_geolocation_payments)
  #df=pd.DataFrame(y_km,columns=['data_geolocation_payments'])
  #data_final=pd.concat([df,data_final_predict],axis=1)
  #data_final=data_final.dropna()
  data_final_predict.drop(['payment_value'],axis=1,inplace=True)
  data_final_predict.drop(['review_score'],axis=1,inplace=True)
  

  
  return data_final_predict
  
 

def create_model():
    fhs_rf_model=joblib.load('random_forest.pkl')
    product_category_name_english_1=joblib.load('product_category2.pkl')
    review_comment_message_1=joblib.load('review_comment2.pkl')
    order_status_1=joblib.load('order_status2.pkl')

    return fhs_rf_model,product_category_name_english_1,review_comment_message_1,order_status_1
    
fhs_rf_model,product_category_name_english_1,review_comment_message_1,order_status_1=create_model()                       
      
    
@app.route('/')
def hello_world():
  return 'Hello World'

@app.route('/index')
def index():
  return flask.render_template('index.html')

@app.route('/predict',methods = ['POST'])
def predict():
    data1=request.files['file']
    y=preprocess(data1)
    
    if type(y) == dict:
      return jsonify(y)
  
    order_status_transform=order_status_1.transform(y.order_status.values)
    product_category_name_english_transform=product_category_name_english_1.transform(y.product_category_name_english.values)
    review_comment_message_1_transform=review_comment_message_1.transform(list(y.review_comment_message.values))
    
    y = y.drop(labels=['review_comment_message','product_category_name_english','order_status'],axis=1)
    
    data_final_concat=hstack((y,order_status_transform,product_category_name_english_transform,review_comment_message_1_transform))
    
    y1=fhs_rf_model.predict(data_final_concat)
    
    if y1 == 1:
        return jsonify({'prediction':'positive'})
    elif y1==0:
        return jsonify({'prediction':'negative'})                   
                        
    #return jsonify({'data':str(y.iloc[0,1])})    
if __name__=='__main__':
    
    app.run(host='0.0.0.0', port=5050,debug=True)    