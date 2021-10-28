# -*- coding: utf-8 -*-
"""
Created on Fri Oct  1 17:49:33 2021

@author: Quang
"""

import matplotlib.pyplot as plt
import pandas as pd                        
from pytrends.request import TrendReq
from sklearn.linear_model import LinearRegression
import numpy as np
import seaborn as sns
from colour import Color
import matplotlib as mpl
import matplotlib.colors
from datetime import date

from mpl_toolkits.basemap import Basemap
from matplotlib.patches import Polygon

today = date.today()
today = today.strftime("%Y-%m-%d")


pytrend = TrendReq()




states_dict = {
        'AL': 'Alabama',
        'AR': 'Arkansas',
        'AZ': 'Arizona',
        'CA': 'California',
        'CO': 'Colorado',
        'CT': 'Connecticut',
        'DE': 'Delaware',
        'FL': 'Florida',
        'GA': 'Georgia',
        'IA': 'Iowa',
        'ID': 'Idaho',
        'IL': 'Illinois',
        'IN': 'Indiana',
        'KS': 'Kansas',
        'KY': 'Kentucky',
        'LA': 'Louisiana',
        'MA': 'Massachusetts',
        'MD': 'Maryland',
        'ME': 'Maine',
        'MI': 'Michigan',
        'MN': 'Minnesota',
        'MO': 'Missouri',
        'MS': 'Mississippi',
        'MT': 'Montana',
        'NC': 'North Carolina',
        'ND': 'North Dakota',
        'NE': 'Nebraska',
        'NH': 'New Hampshire',
        'NJ': 'New Jersey',
        'NM': 'New Mexico',
        'NV': 'Nevada',
        'NY': 'New York',
        'OH': 'Ohio',
        'OK': 'Oklahoma',
        'OR': 'Oregon',
        'PA': 'Pennsylvania',
        'RI': 'Rhode Island',
        'SC': 'South Carolina',
        'SD': 'South Dakota',
        'TN': 'Tennessee',
        'TX': 'Texas',
        'UT': 'Utah',
        'VA': 'Virginia',
        'VT': 'Vermont',
        'WA': 'Washington',
        'WI': 'Wisconsin',
        'WV': 'West Virginia',
        'WY': 'Wyoming'
}


pytrend.build_payload(
kw_list=['electric vehicles'],
cat=0,
timeframe='2010-01-01 2021-06-01',
geo='US')
df_all = pytrend.interest_over_time()
df_all.columns=['US','isPartial']
df_all=df_all[['US']]
regression_model = LinearRegression()
regression_model.fit(np.array(range(len(df_all))).reshape(-1, 1), df_all['US'].values)
rate=regression_model.coef_
rate_of_change={'US':rate}

for state in states_dict:
    print(state)
    # name='US'+state
    pytrend.build_payload(
    kw_list=['electric vehicles'],
    cat=0,
    timeframe='2010-01-01 2021-06-01',
    geo='US-'+state)
    data = pytrend.interest_over_time()
    data.columns=[state,'isPartial']
    data=data[[state]]
    df_all=pd.merge(df_all, data, left_index=True, right_index=True)
     
    regression_model = LinearRegression()
    regression_model.fit(np.array(range(len(df_all))).reshape(-1, 1), df_all[state].values)
    rate=regression_model.coef_
    rate_of_change[state]=rate

df_rate=pd.DataFrame(rate_of_change.items(), columns=['State', 'Rate'])
df_rate_state=df_rate[1:]#remove US

df_rate_state['Rate']=df_rate_state['Rate'].astype('float')


fig, ax = plt.subplots(figsize=(20,15))
ax = sns.boxplot(x=df_all.index.year, y=df_all.US, ax=ax)
ax.set_title('Google trend of electric vehicles 2010-2021 in US',size=45)
ax.set_ylabel('Interest',size=35)
ax.set_xlabel('')
plt.xticks(fontsize=25)
plt.yticks(fontsize=25)
plt.savefig('boxplot_trend.png')


interval=list(np.linspace(np.min(df_rate_state['Rate']), np.max(df_rate_state['Rate']), num=6))
interval[0]=-1
interval_name=['< '+format(interval[1],'.3f')]
for i in range(1,len(interval)-2):
    interval_name.append(format(interval[i],'.3f')+' - '+format(interval[i+1],'.3f'))
interval_name.append('>= '+format(interval[-2],'.3f') )  

#scale color from red to green   
# interval=[-1, 0.2, 0.4,0.6,0.8,2]

red = Color("red")
colors_list = list(red.range_to(Color("green"),5))
color_sq=[str(i) for i in colors_list]
#create color scheme usable in map
colors = sns.color_palette(color_sq)

cmap = mpl.colors.ListedColormap(colors)
norm = mpl.colors.BoundaryNorm(np.arange(-0.5,len(colors)), cmap.N) 



new_data, bins = pd.cut(df_rate_state['Rate'], interval, retbins=True, labels=list(range(len(interval)-1)))
# new_data, bins = pd.qcut(df[var], 5, retbins=True, labels=list(range(5))) #qcut equally space bins
color_ton = {}
i=-1
for val in range(len(new_data)):
    i+=1
    color_ton[df_rate_state.iloc[val]['State']]=color_sq[new_data[val+1]]






inv_dict = {v: k for k, v in states_dict.items()}        
        

fig, axs = plt.subplots(1,1,figsize = (60,30),facecolor='w',edgecolor='k')
# create the map
map = Basemap(llcrnrlon=-119,llcrnrlat=22,urcrnrlon=-64,urcrnrlat=49,
        projection='lcc',lat_1=33,lat_2=45,lon_0=-95)

# load the shapefile, use the name 'states'
map.readshapefile('st99_d00', name='states', drawbounds=True)

# collect the state names from the shapefile attributes so we can
# look up the shape obect for a state by it's name
state_names = []
for shape_dict in map.states_info:
    state_names.append(shape_dict['NAME'])

axs  = plt.gca() # get current axes instance
ct=-1
for id in states_dict.values():
    ct+=1
    seg = map.states[state_names.index(id)]
    poly = Polygon(seg, facecolor=color_ton[inv_dict[id]],edgecolor='black')
    axs.add_patch(poly)

sm = plt.cm.ScalarMappable(cmap=cmap, norm=norm)
sm.set_array([])  # only needed for matplotlib < 3.1
cbaxes = fig.add_axes([0.81, 0.20, 0.01, 0.6]) #left-right,up-down,wide,length
# cb = plt.colorbar(ax1, cax = cbaxes)  
cbar=fig.colorbar(sm, cax = cbaxes) #change color bar to horizontal direction if needed
cbar.set_ticks([])#remove colorbar tick lable

axs.spines['top'].set_color('none') #remove the axis and tick
axs.spines['bottom'].set_color('none')
axs.spines['left'].set_color('none')
axs.spines['right'].set_color('none')
axs.set_title('Rate of change electric vehicles trend 2010-2021 for US Contiguous',size=100)
for index, label in enumerate(interval_name): #add text to the color bar
    x = 5
    y = index -0.15
    cbar.ax.text(x, y, label,fontsize=80)

plt.savefig('State_map.png')

