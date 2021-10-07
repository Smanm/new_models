#!/usr/bin/env python
# coding: utf-8

# In[203]:


from selenium import webdriver
import time
import pandas as pd
import requests
import json
from fbprophet import Prophet


# In[204]:


import warnings;
warnings.simplefilter('ignore')


# In[205]:


path=r'C:\Users\hp\Downloads\chromedriver_win32 (1)\chromedriver.exe'
browser=webdriver.Chrome(executable_path=path)


# In[206]:


browser.get('https://data.covid19india.org/')


# In[207]:


data=browser.find_element_by_xpath('/html/body/div/table[1]/tbody/tr[1]/td[2]/a')


# In[208]:


data.click()


# In[209]:


time.sleep(10)


# In[210]:


r=requests.get('https://data.covid19india.org/v4/min/timeseries.min.json')


# In[211]:


column_name=['dates','total_confirmed','total deceased']
Df=pd.DataFrame(columns=column_name)
Df
column_name=['dates','total_confirmed','total deceased']
D_frame=pd.DataFrame(columns=column_name)
D_frame


# In[212]:


state=['AN','AP','AR','AS','BR','CH','CT','DN','DD','DL','GA','GJ','HR','HP','JK','JH','KA','KL','LD','MP','MH','MN','ML','MZ','NL','OR','PY','PB','RJ','SK','TN','TG','TR','UP','UT','WB']
print(state)


# In[213]:


state_ab=input("ENTER THE STATE ABBREVIATION FROM ABOVE AN/GA/MH ")


# In[214]:


Dat0={}
data0=json.loads(r.text)[state_ab]['dates']['2020-1{}-3{}'.format(2,1)]['total']['confirmed']
Data0=json.loads(r.text)[state_ab]['dates']['2020-1{}-3{}'.format(2,1)]['total']['deceased']
#data1=json.loads(r.text)['KL']['dates']
#print((data))
Dat0[Df.columns[0]]=str(['20201{}3{}'.format(2,1)])
Dat0[Df.columns[1]]=data0
Dat0[Df.columns[2]]=Data0
Df=Df.append(Dat0,ignore_index=True)
for k in range(9):
    for i in range(9):
        try:
            Dat={}
           
            
            data1=json.loads(r.text)[state_ab]['dates']['2021-0{}-0{}'.format(k+1,i+1)]['total']['confirmed']
            Data1=json.loads(r.text)[state_ab]['dates']['2021-0{}-0{}'.format(k+1,i+1)]['total']['deceased']
        #data1=json.loads(r.text)['KL']['dates']
        #print((data))
            Dat[Df.columns[0]]=str('2021/{}/{}'.format(k+1,i+1))
           
            Dat[Df.columns[1]]=data1
            Dat[Df.columns[2]]=Data1
            Df=Df.append(Dat,ignore_index=True)
            D_frame=D_frame.append(Dat,ignore_index=True)
        
        except:
            continue
    for j in range(10):
        try:
            Dat2={}
            data2=json.loads(r.text)[state_ab]['dates']['2021-0{}-1{}'.format(k+1,j)]['total']['confirmed']
            Data2=json.loads(r.text)[state_ab]['dates']['2021-0{}-1{}'.format(k+1,j)]['total']['deceased']
        #data1=json.loads(r.text)['KL']['dates']
        #print((data))
            Dat2[Df.columns[0]]=str('2021/{}/1{}'.format(k+1,j))
           
            Dat2[Df.columns[1]]=data2
            Dat2[Df.columns[2]]=Data2
            
            Df=Df.append(Dat2,ignore_index=True)
            D_frame=D_frame.append(Dat2,ignore_index=True)
        
        except:
            continue  
    for l in range(10):
        try:
            Dat3={}
            data3=json.loads(r.text)[state_ab]['dates']['2021-0{}-2{}'.format(k+1,l)]['total']['confirmed']
            Data3=json.loads(r.text)[state_ab]['dates']['2021-0{}-2{}'.format(k+1,l)]['total']['deceased']
        #data1=json.loads(r.text)['KL']['dates']
        #print((data))
            Dat3[Df.columns[0]]=str('2021/{}/2{}'.format(k+1,l))
           
            Dat3[Df.columns[1]]=data3
            Dat3[Df.columns[2]]=Data3
            Df=Df.append(Dat3,ignore_index=True)
            D_frame=D_frame.append(Dat3,ignore_index=True)
        
        except:
            continue 
    
    for m in range(2):
        try:
            Dat4={}
            data4=json.loads(r.text)[state_ab]['dates']['2021-0{}-3{}'.format(k+1,m)]['total']['confirmed']
            Data4=json.loads(r.text)[state_ab]['dates']['2021-0{}-3{}'.format(k+1,m)]['total']['deceased']
        #data1=json.loads(r.text)['KL']['dates']
        #print((data))
            Dat4[Df.columns[0]]=str('2021/{}/3{}'.format(k+1,m))
           
            Dat4[Df.columns[1]]=data4
            Dat4[Df.columns[2]]=Data4
            Df=Df.append(Dat4,ignore_index=True)
            D_frame=D_frame.append(Dat4,ignore_index=True)
        
        except:
            continue           


# In[215]:


display(Df)


# In[216]:


#n_rows=Df.shape[0]
Df.dtypes


# In[217]:


#pd.set_option('display.max_rows',n_rows)
D_frame=D_frame.rename(columns={"total_confirmed":"active_cases"})
D_frame=D_frame.rename(columns={"total deceased":"daily deaths"})


# In[218]:


D_frame


# In[219]:



#rows=D_frame.shape[0]
#pd.set_option('display.max_rows',rows)


# In[220]:



#D_frame


# In[221]:


D_frame.loc[:,"active_cases"]=D_frame.loc[:,"active_cases"]- Df.loc[:,"total_confirmed"]
D_frame.loc[:,"daily deaths"]=D_frame.loc[:,"daily deaths"]- Df.loc[:,"total deceased"]


# In[222]:


D_frame


# In[223]:


D_frame.dtypes


# In[224]:


active_columns=D_frame[["dates","active_cases"]]
active_df=active_columns.copy()


# In[225]:


death_columns=D_frame[["dates","daily deaths"]]
death_df=death_columns.copy()


# Training model

# In[226]:


active_df.columns=['ds','y']
active_df.dtypes


# In[227]:


death_df.columns=['ds','y']


# In[228]:


active_df['ds']=pd.to_datetime(active_df['ds'],format='%Y/%m/%d')


# In[229]:


death_df['ds']=pd.to_datetime(death_df['ds'],format='%Y/%m/%d')


# In[230]:


active_df


# In[231]:


ml=Prophet(interval_width=0.95,daily_seasonality=True)
model=ml.fit(active_df)


# In[232]:


future=ml.make_future_dataframe(periods=30,freq='D')
forecast=ml.predict(future)


# In[233]:


forecast.tail(30)[['ds','yhat']] # yhat shows the 30 DAYS ACTIVE CASE FORECAST


# In[234]:


death_df


# In[235]:


dl=Prophet(interval_width=0.95,daily_seasonality=True)
model0=dl.fit(death_df)


# In[236]:


future2=dl.make_future_dataframe(periods=30,freq='D')
forecast2=dl.predict(future2)


# In[237]:


forecast2.tail(30)[['ds','yhat']] # yhat shows the 30 DAYS DEATH FORECAST


# In[ ]:




