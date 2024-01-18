import pandas as pd
import numpy as np

data = pd.read_csv('./gen_reversed2.csv')
cols = ['kidneybeans', 'chickpea', 'maize', 'pigeonpeas', 'mungbean',
       'blackgram', 'mothbeans', 'pomegranate', 'jute', 'cotton',
       'lentil', 'muskmelon', 'watermelon', 'coffee', 'rice', 'apple',
       'orange', 'banana', 'coconut', 'papaya', 'grapes', 'mango']
columns = ['7','8','9','10','11','12','13','14','15','16','17','18','19','20','21','22','23','24','25','26','27','28']
dict = {key:val for key , val in zip(columns, cols) }
#print(dict) {'7': 'kidneybeans', '8': 'chickpea', '9': 'maize', '10': 'pigeonpeas', '11': 'mungbean', '12': 'blackgram', '13': 'mothbeans', '14': 'pomegranate', '15': 'jute', '16': 'cotton', '17': 'lentil', '18': 'muskmelon', '19': 'watermelon', '20': 'coffee', '21': 'rice', '22': 'apple', '23': 'orange', '24': 'banana', '25': 'coconut', '26': 'papaya', '27': 'grapes', '28': 'mango'}
data1 = data.iloc[:,7:]
new = []

col_val = data1.max(axis=1)
col_name = data1.idxmax(axis=1)
for i in col_name:
    new.append(dict[i])

new = pd.Series(new, name='label').to_frame()
data3 = data.iloc[:,:7] 
#print(new.head())
data3['label'] = new['label']

#print(new.shape)
#print(data3.shape)
data3.to_csv('./result2.csv', index=False)
print("file saved")
