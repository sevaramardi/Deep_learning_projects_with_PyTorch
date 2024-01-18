# Scale synthetic data to match the value range of the original data
import pandas as pd
import numpy as np
from sklearn.preprocessing import OneHotEncoder

real_data = pd.read_csv('./crop.csv')
gen_data = pd.read_csv('./gen_tf_version3.csv')
encoder = OneHotEncoder()
y_data = encoder.fit_transform(real_data[['label']])

def _df(data):
    df = pd.DataFrame(data)
    for c in range(df.shape[1]):
        mapping = {df.columns[c]: c}
        df = df.rename(columns=mapping)
    return df

x_data = _df(real_data.iloc[:,:-1])
y_data = y_data.toarray()
real_data = np.column_stack((x_data.values, y_data))
real_data = pd.DataFrame(real_data, columns = ['0','1','2','3','4','5','6','7','8','9','10','11','12','13','14','15','16','17','18','19','20','21','22','23','24','25','26','27','28'])
#print(real_data)
#print(type(x_data), type(y_data))
#print(gen_data.columns)
#exit()
for column in real_data.columns:
    min_val = real_data[column].min()
    max_val = real_data[column].max()
    gen_data[column] = gen_data[column] * (max_val - min_val) + min_val

# Save synthetic data to CSV with the same column names and value range as the original data
gen_data.to_csv('./gen_reversed.csv', index=False)
print("file saved")