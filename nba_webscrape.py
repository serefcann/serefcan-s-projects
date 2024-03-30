#!/usr/bin/env python
# coding: utf-8

# In[1]:


import pandas as pd
import requests
pd.set_option('display.max_columns',None)
import time
import numpy as np


# In[3]:


test_url = 'https://stats.nba.com/stats/leagueLeaders?LeagueID=00&PerMode=PerGame&Scope=S&Season=2019-20&SeasonType=Regular%20Season&StatCategory=PTS'


# In[4]:


r = requests.get(url = test_url).json()


# In[8]:


table_headers = r['resultSet']['headers']


# In[13]:


pd.DataFrame(r['resultSet']['rowSet'], columns = table_headers)


# In[26]:


temp_df1 = pd.DataFrame(r['resultSet']['rowSet'], columns = table_headers)

temp_df2 = pd.DataFrame({'Year': ['2019-20' for i in range(len(temp_df1))],
                         'Season':['Regular%20Season' for i in range(len(temp_df1))]})
temp_df3 = pd.concat([temp_df2,temp_df1], axis=1)
temp_df3


# In[33]:


del temp_df1, temp_df2, temp_df3


# In[59]:


header =['Year','Season_type'] + table_headers
pd.DataFrame(columns=header)


# In[55]:


headers= {
    'Accept': '*/*',
    'Accept-Encoding':'gzip, deflate, br',
    'Accept-Language':'tr-TR,tr;q=0.9,en-US;q=0.8,en;q=0.7',
    'Connection':'keep-alive',
    'Host':'stats.nba.com',
    'Origin':'https://www.nba.com',
    'Referer':'https://www.nba.com/',
    'Sec-Ch-Ua':'"Not A(Brand";v="99", "Opera";v="107", "Chromium";v="121"',
    'Sec-Ch-Ua-Mobile':'?0',
    'Sec-Ch-Ua-Platform':"Windows",
    'Sec-Fetch-Dest':'empty',
    'Sec-Fetch-Mode':'cors',
    'Sec-Fetch-Site':'same-site',
    'User-Agent':'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/121.0.0.0 Safari/537.36 OPR/107.0.0.0'
}


# In[62]:


Seasons =['Regular%20Season','Playoffs']
years = ['2019-20','2020-21','2021-22','2022-23']
df = pd.DataFrame(columns=header)

begin_loop= time.time()

for y in years:
    for s in Seasons:
        api_url = 'https://stats.nba.com/stats/leagueLeaders?LeagueID=00&PerMode=PerGame&Scope=S&Season='+y+'&SeasonType='+s+'&StatCategory=PTS'
        r = requests.get(api_url,headers=headers).json()
        temp_df1 = pd.DataFrame(r['resultSet']['rowSet'], columns = table_headers)

        temp_df2 = pd.DataFrame({'Year': [y for i in range(len(temp_df1))],
                                 'Season_type':[s for i in range(len(temp_df1))]})
        temp_df3 = pd.concat([temp_df2,temp_df1], axis=1)
        df = pd.concat([df,temp_df3], axis=0)
        print(f'finished web scraping  for the {y} {s}')
        lag = np.random.uniform(low=5,high=40)
        print(f'waiting {round(lag,1)} seconds')
        time.sleep(lag)
print(f'process completed. Total run time {time.time()-begin_loop}')


# In[63]:


df


# In[ ]:




