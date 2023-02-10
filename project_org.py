#!/usr/bin/env python
# coding: utf-8

# In[1]:


import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


# In[2]:


df1=pd.read_csv('datafile1.csv',skiprows=20)
df2=pd.read_csv('datafile2.csv',skiprows=17)


# In[3]:


df1.head()


# In[4]:


df2.head()


# In[5]:


#merging two csvs
df=df=pd.merge(df1,df2,how='left',on=('LAT','LON','YEAR','MO','DY'))
df.head()


# In[6]:


df['MO']=df.MO.astype(str)
df['YEAR']=df.YEAR.astype(str)
df['DY']=df.DY.astype(str)

df['date']=df['YEAR'].str.cat(df['MO'],sep='/')
df['Date']=df['date'].str.cat(df['DY'],sep='/')
df.head()


# In[7]:


#removing lat and lon columns
df.drop(columns=['YEAR','MO','DY','LAT','LON','QV2M','date','WS50M_RANGE', 'WS50M_MIN','WS50M_MAX', 'WS50M','T2MDEW', 'T2MWET',
       'WS10M_MIN', 'WS10M_MAX','WS10M'],axis=1,inplace=True)
df.head()


# In[8]:


df.set_index(['Date'],inplace=True)
df.head()


# In[9]:


df.dtypes


# In[10]:


df.isnull().values.any()


# In[11]:


df.info()


# In[12]:


df.tail()


# # New Section

# In[ ]:


fig,ax=plt.subplots(figsize=(25,25))
for i in range(len(df.columns)):
  plt.subplot(len(df.columns),1,i+1)
  name=df.columns[i]
  plt.plot(df[name])
  plt.title(name,y=0,loc='right')
  plt.yticks([])
plt.show()
fig.tight_layout()


# In[ ]:


plt.figure(figsize=(30,5))
df['PRECTOT'].plot()


# In[ ]:


df.columns


# In[ ]:


df.describe()


# In[13]:


#scale the data
data_train=df.loc[:'2017/12/31',:][[  'PRECTOT', 'RH2M', 'PS', 'T2M_RANGE', 'T2M_MAX', 'T2M_MIN', 'T2M','WS10M_RANGE']]
data_test=df.loc['2018/1/1':'2019/12/31',:][[  'PRECTOT', 'RH2M', 'PS', 'T2M_RANGE',  'T2M_MAX', 'T2M_MIN', 'T2M','WS10M_RANGE']]

from sklearn.preprocessing import MinMaxScaler
scalar=MinMaxScaler()
data_train_scaled=scalar.fit_transform(data_train)
data_test_scaled=scalar.transform(data_test)


# In[14]:


#create training and testing dataset
#splitting data into X and Y for training  
x_train,y_train=[],[]
for i in range(1,len(data_train_scaled)):
  x_train.append(data_train_scaled[i-1])
  y_train.append(data_train_scaled[i])


#splitting data into X and Y for testing
x_test=[]
y_test=[]
for i in range(1,len(data_test_scaled)):
  x_test.append(data_test_scaled[i-1])
  y_test.append(data_test_scaled[i])



  




# In[ ]:





# In[15]:


pd.DataFrame(y_train)


# In[16]:


x_train,y_train=np.array(x_train),np.array(y_train)
x_train.shape,y_train.shape


# In[17]:


x_test=np.array(x_test)
y_test=np.array(y_test)
x_test.shape


# In[27]:


from keras.models import Sequential
from keras.layers import Dense, Activation
from keras.optimizers import Adam
from keras.models import load_model
from keras.layers import Dropout


# In[28]:


model = Sequential()
model.add(Dense(200, input_dim=8, activation='sigmoid'))

model.add(Dense(200, activation='sigmoid'))
model.add(Dropout(0.5))
model.add(Dense(8))


model.summary()


model.compile(optimizer='adam',loss='mse', metrics=['accuracy'])


# In[29]:


history=model.fit(x_train,y_train,validation_split=0.16 ,epochs=3000,batch_size=15,shuffle=False)


# In[30]:


# save model and architecture to single file
model.save("model76.h5")
print("Saved model to disk")



# In[31]:


model = load_model('model76.h5')
model.summary()


# In[32]:


ans=model.predict(x_test,batch_size=15)
ans=scalar.inverse_transform(ans)
actual_value=scalar.inverse_transform(y_test)


# In[33]:


plt.plot(history.history['accuracy'])
plt.plot(history.history['val_accuracy'])
plt.title('Training and Validation accuracy')
plt.xlabel('Epochs')
plt.ylabel('Accuracy')
plt.legend()
plt.show()


# In[34]:


scores = model.evaluate(actual_value,ans,verbose=0)
print("%s: %.2f%%" % (model.metrics_names[1], scores[1]*100))


# In[35]:


print('actual value')
print(' RH2M   PS  T2M_RANGE T2M_MAX T2M_MIN WS10M_RANGE')

print(pd.DataFrame(actual_value))
print('predicted value')
print(' PRECTOT RH2M PS T2M_RANGE T2M_MAX T2M_MIN T2M WS10M_RANGE')

print(pd.DataFrame(ans))


# In[39]:


66
a=input('enter the date of day whose weather you would like to predict ')
print('enter the weather parameters of previous day')
a=[]
print('Precipitation , Relative humidity, Surface pressure , Temperature range, MAX temp ,MIN_temp ,Temperature ,Wind speed at 10M' )
#Precipitation , Relative humidity, Surface pressure , Temperature range, MAX temp ,MIN_temp ,Temperature ,Wind speed at 10M 
for i in range(0,8):
 a.append(float(input()))
a=np.array(a) 
a.shape=(1,8)
a=pd.DataFrame(a) 
print(a)
ans1=model.predict(a)
ans2=scalar.inverse_transform(ans1)
print(ans2)



# In[39]:


scores = model.evaluate(ans1,a,verbose=0)
print("%s: %.2f%%" % (model.metrics_names[1], scores[1]*100))


# In[ ]:




