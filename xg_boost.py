#!/usr/bin/env python
# coding: utf-8

# In[1]:


from selenium import webdriver
import time
import pandas as pd
import requests
import json
from sklearn.model_selection import train_test_split
from xgboost import XGBRegressor
import seaborn as sns


# In[2]:


import warnings;
warnings.simplefilter('ignore')


# In[3]:


path=r'C:\Users\hp\Downloads\chromedriver_win32 (1)\chromedriver.exe'
browser=webdriver.Chrome(executable_path=path)


# In[4]:


browser.get('https://data.covid19india.org/')


# In[5]:


data=browser.find_element_by_xpath('/html/body/div/table[1]/tbody/tr[1]/td[2]/a')


# In[6]:


data.click()


# In[7]:


time.sleep(10)


# In[8]:


r=requests.get('https://data.covid19india.org/v4/min/timeseries.min.json')


# In[9]:


column_name=['dates','total_confirmed','total deceased']
Df=pd.DataFrame(columns=column_name)
Df
column_name=['dates','total_confirmed','total deceased']
D_frame=pd.DataFrame(columns=column_name)
D_frame


# In[10]:


state=['AN','AP','AR','AS','BR','CH','CT','DN','DD','DL','GA','GJ','HR','HP','JK','JH','KA','KL','LD','MP','MH','MN','ML','MZ','NL','OR','PY','PB','RJ','SK','TN','TG','TR','UP','UT','WB']
print(state)


# In[11]:


state_ab=input("ENTER THE STATE ABBREVIATION FROM ABOVE AN/GA/MH ")


# In[12]:


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


# In[13]:


display(Df)


# In[14]:


#n_rows=Df.shape[0]
Df.dtypes


# In[15]:


#pd.set_option('display.max_rows',n_rows)
D_frame=D_frame.rename(columns={"total_confirmed":"active_cases"})
D_frame=D_frame.rename(columns={"total deceased":"daily_deaths"})


# In[16]:


D_frame


# In[17]:



#rows=D_frame.shape[0]
#pd.set_option('display.max_rows',rows)


# In[18]:



#D_frame


# In[19]:


D_frame.loc[:,"active_cases"]=D_frame.loc[:,"active_cases"]- Df.loc[:,"total_confirmed"]
D_frame.loc[:,"daily_deaths"]=D_frame.loc[:,"daily_deaths"]- Df.loc[:,"total deceased"]


# In[20]:


D_frame


# In[21]:


D_frame.dtypes


# In[ ]:


Df["active_cases"]=D_frame[["active_cases"]].copy()


# In[26]:


Df


# In[39]:


Df.drop(labels=["dates","total_confirmed","total deceased"],axis=1,inplace=True)


# In[32]:


Df.dropna(inplace=True)


# In[40]:


Df


# In[41]:


Df["Target_cases"]=Df.active_cases.shift(-1)


# In[ ]:





# In[42]:


Df


# In[43]:


Df.dropna(inplace=True)


# In[44]:


Df


# In[45]:


def train_test_split(dat_frame,per):
    dat_frame=dat_frame.values
    l=int(len(dat_frame)*(1-per))
    return dat_frame[:l],dat_frame[l:]


# In[46]:


train,test=train_test_split(Df,0.2)


# In[47]:


print(len(D_frame))
print(len(train))
print(len(test))


# In[61]:


test


# In[53]:


X1=train[:,:-1]
y1=train[:,-1]


# In[54]:


y1


# In[56]:


reg=XGBRegressor(objective="reg:squarederror",n_estimators=700)
reg.fit(X1,y1)


# In[57]:


predictions=reg.predict(test[:,:-1])


# #deaths_prediction_model

# In[68]:


Df["daily_deaths"]=D_frame[["daily_deaths"]].copy()


# In[70]:


Df.drop(labels=["active_cases","Target_cases"],axis=1,inplace=True)


# In[72]:


Df["Target_deaths"]=Df.daily_deaths.shift(-1)


# In[74]:


Df.dropna(inplace=True)


# In[76]:


train2,test2=train_test_split(Df,0.2)


# In[77]:


X2=train2[:,:-1]
y2=train2[:,-1]


# In[78]:


reg2=XGBRegressor(objective="reg:squarederror",n_estimators=700)
reg2.fit(X2,y2)


# In[79]:


predictions2=reg2.predict(test2[:,:-1])

