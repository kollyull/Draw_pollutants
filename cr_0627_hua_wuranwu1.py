# -*- coding: utf-8 -*-
"""
Created on Thu Jun 27 18:04:44 2019

@author: Rong Cao
"""


import os
import pandas as pd
import datetime as dt
import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np
from scipy import stats
import matplotlib as mpl
import itertools


def hua_chuizhi(df,ap,labbb,kk):
    yanse = ['red','green','orange','blue','brown','purple','pink','black','rosybrown','cornflowerblue','salmon','aquamarine']
    # cmap_my = mpl.colors.ListedColormap(yanse)
    plt.plot(df[ap],df['Height'],marker='o', markersize=10,label = labbb,c = yanse[kk])
    plt.ylabel('Height (m)')
    
def hua_lashen(df,ap,labbb,kk):
    yanse = ['red','green','orange','blue','brown','purple','pink','black','rosybrown','cornflowerblue','salmon','aquamarine']
    # cmap_my = mpl.colors.ListedColormap(yanse)
    plt.plot(df['Distance'],df[ap],marker='o', markersize=10,label = labbb,c = yanse[kk])
    plt.xlabel('Distance (m)')


def di(df):
    df = df.reset_index(drop = True)
    return(df)


def resample_hour(before,time_name):
    after = before.resample(rule='H', on='datetime').mean()
    after[time_name] = after.index
    return(di(after))


def duibi_huitu(guokong,shouchi,cheng,shouchi_label,my_ylabel,my_title,my_save_name):
    sns.set(font_scale=1.5,font = 'Times New Roman')
    plt.figure(figsize=(10, 6))
    plt.rcParams['font.family'] = 'Times New Roman'
    plt.rcParams['font.size'] = 16
    plt.plot_date(duibi_biaozhun_ap['time'],duibi_biaozhun_ap[guokong],'o-',label = 'Monitoring Station')
    plt.plot_date(duibi_biaozhun_ap['time'],duibi_biaozhun_ap[shouchi]*cheng,'o-',label = shouchi_label)
    plt.xlabel('Date and Time')
    plt.ylabel(my_ylabel)
    plt.legend()
    plt.title(my_title)
    if (len(my_save_name) !=0):
        plt.savefig(results_path+'duibi/'+my_save_name+'.jpg',dpi = 300, bbox_inches='tight')
    plt.show()

def hua_zhuxingtu(mingcheng,biaozhu):
    plt.figure(figsize=(10, 6))
    g = sns.boxplot(y= fenlei_place_3[mingcheng],x=fenlei_place_3['place'],hue=fenlei_place_3['Campaign'],orient="v",saturation=0.75, width=0.4)
    g.set(ylabel = biaozhu,xticklabels = ['Laojianhe','Shuniu'],xlabel = 'Place')
    plt.axhline(y=xiajinxian[mingcheng].mean())




#%%


plt.rcParams['font.size'] = 16

names = locals()


pwd = os.getcwd()
print(pwd)
uav_file_path = pwd+'/../flight/'
data_file_path = pwd+'/../data/'
results_path = pwd+'/../results/'
qixiang_file_path = pwd+'/../data/qixiang/'

data_name = ['9306','8534','bc','trh']
monitor_file = []
#all_exp_record = pd.read_excel(data_file_path+'both_exp_record_0506.xlsx',encoding = 'gbk')
all_exp_record = pd.read_excel(data_file_path+'all_exp_record_0608.xlsx',encoding = 'gbk')

all_exp_record['datetime_begin'] = pd.to_datetime(all_exp_record['datetime_begin'])
all_exp_record['datetime_end'] = pd.to_datetime(all_exp_record['datetime_end'])
all_exp_record['date'] = pd.to_datetime(all_exp_record['date'])

fengsu = pd.read_excel(qixiang_file_path+'54715099999_2018_and_2019.xlsx')
fengsu['datetime(gmt)'] = pd.to_datetime(fengsu['datetime(gmt)'])
fengsu['datetime_china'] = fengsu['datetime(gmt)']+dt.timedelta(hours=8)
fengsu = fengsu[(fengsu['datetime_china']>= pd.datetime(2018,11,1)) & (fengsu['datetime_china']<= pd.datetime(2019,5,1))]

fengsu_2 = pd.read_excel(qixiang_file_path+'weather_20181101_20190501.xlsx')
fengsu_2['datetime'] = pd.to_datetime(fengsu_2['time'])


for i in range(0,len(data_name)):
    monitor_file.append(pd.read_excel(data_file_path+'all_'+data_name[i]+'.xlsx'))
    monitor_file[i]['datetime'] = pd.to_datetime(monitor_file[i]['datetime'])

figure, ax = plt.subplots()
plt.rcParams['font.family'] = 'Times New Roman'
plt.rcParams['font.size'] = 16
figure.suptitle('Remove the abnormal value')
plt.subplot(221)
plt.plot(monitor_file[2]['bc_ir'])
plt.subplot(222)
yichangzhi = monitor_file[2][monitor_file[2]['bc_ir']>1000000].index.tolist()
monitor_file[2].loc[yichangzhi,'bc_ir'] = np.nan
plt.plot(monitor_file[2]['bc_ir'])
plt.subplot(223)
yichangzhi = monitor_file[2][monitor_file[2]['bc_ir']<0].index.tolist()
monitor_file[2].loc[yichangzhi,'bc_ir'] = np.nan
plt.plot(monitor_file[2]['bc_ir'])
plt.subplot(224)
yichangzhi = monitor_file[2][monitor_file[2]['bc_ir']>300000].index.tolist()
monitor_file[2].loc[yichangzhi,'bc_ir'] = np.nan
plt.plot(monitor_file[2]['bc_ir'])
plt.show()

#%%
#去除异常值
monitor_file_all = pd.merge(monitor_file[0],monitor_file[1].iloc[:,2:],on = 'datetime',how = 'outer')
monitor_file_all = pd.merge(monitor_file_all,monitor_file[2].iloc[:,3:],on = 'datetime',how = 'outer')
monitor_file_all = pd.merge(monitor_file_all,monitor_file[3],on = 'datetime',how = 'outer')
#%%
wuranwu_name = ['PM_0.3', 'PM_0.5', 'PM_1', 'PM_3', 'PM_5', 'PM_10',
                'PM1', 'PM2.5', 'RESP', 'PM10', 'TOTAL',
                'bc_ir',
                'temperature', 'RH', 'dew']
wuranwu_ylabel = ['Number concentration (#$/m^3$)','Number concentration (#$/m^3$)','Number concentration (#$/m^3$)',
                  'Number concentration (#$/m^3$)','Number concentration (#$/m^3$)','Number concentration (#$/m^3$)',                 
                  'Concenreation ($\mu g/m^3$)','Concenreation ($\mu g/m^3$)','Concenreation ($\mu g/m^3$)',
                  'Concenreation ($\mu g/m^3$)','Concenreation ($\mu g/m^3$)',
                  'Concenreation ($ng /m^3$)', 
                  'Temperature (°C)', 'RH (%)', 'Dew (°C)'
                  ]
wuranwu_title = ['PM$_{0.3}$ Number Concentration','PM$_{0.5}$ Number Concentration','PM$_{1}$ Number Concentration',
                 'PM$_{3}$ Number Concentration','PM$_{5}$ Number Concentration','PM$_{10}$ Number Concentration',
                 'PM$_{1}$ Mass Concentration','PM$_{2.5}$ Mass Concentration','RESP Mass Concentration',
                 'PM$_{10}$ Mass Concentration','Total PM Mass Concentration',
                 'Black Carbon Mass Concentration',
                 'Temperature','RH','Dew'
                 ]

#%% 去除每一趟的异常值
flight_id = di(all_exp_record['flight'].drop_duplicates())
ap_between_before = pd.DataFrame()
ap_between_after = pd.DataFrame()
sns.set(font = 'Times New Roman')
for k in range(0, len(flight_id)):
    one_fli = di(all_exp_record[all_exp_record['flight'] == flight_id[k]])
    time_bg = one_fli['datetime_begin'].min()
    time_ed = one_fli['datetime_end'].max()
    id_this = all_exp_record.loc[k,'all_id_two']
#    上面这里不对，要把id改成每一趟的id，而不是每一行的，然后选这一趟的第一个的begin和最后一个的end，然后挑这两个之间的数据，2019.6.3
    ap_between = di(monitor_file_all[(monitor_file_all['datetime'] >= time_bg)&(monitor_file_all['datetime'] <= time_ed)])
    ap_between_new = ap_between.copy()
    ap_between_before = ap_between_before.append(ap_between.copy())
    figure, ax = plt.subplots(figsize=(18, 12))
    for j in range(0,len(wuranwu_name)):
        arr = ap_between[wuranwu_name[j]]
        if len(arr.dropna())>0:
            mean = arr.mean()
            sd = arr.std()
            for x in range(0,len(arr)):
                if (arr[x] > mean - 3 * sd) & (arr[x] < mean + 3 * sd):
                    ap_between_new.loc[x,wuranwu_name[j]] = arr[x]
                else:
                    ap_between_new.loc[x,wuranwu_name[j]] = np.nan
        plt.subplot(4,4,j+1)
        plt.plot(ap_between.loc[:,wuranwu_name[j]],'o-',label = 'Original')
        plt.plot(ap_between_new.loc[:,wuranwu_name[j]],'o-',label = 'Drop Abnormal')
        plt.title(wuranwu_title[j])
        plt.legend()
    plt.tight_layout()
#    plt.savefig(results_path+'data_clean/quchuyichang_'+str(k)+'.jpg',dpi = 200)
    ap_between_after = ap_between_after.append(ap_between_new)

#%%
record_and_ap = all_exp_record.copy()
ap_mean_all = pd.DataFrame()
ap_min = pd.DataFrame()
for k in range(0, len(all_exp_record)):
    time_bg = all_exp_record.loc[k,'datetime_begin']
    time_ed = all_exp_record.loc[k,'datetime_end']
    id_this = all_exp_record.loc[k,'all_id_two']
    ap_yihang = ap_between_after[(ap_between_after['datetime'] >= time_bg)&(ap_between_after['datetime'] <= time_ed)]
    if len(ap_yihang) >0:
        ap_yihang_mean = pd.DataFrame(ap_yihang.mean()).T
        ap_yihang_mean['all_id_two'] = id_this
        ap_mean_all = ap_mean_all.append(ap_yihang_mean)
        ap_yihang_min = pd.DataFrame(ap_yihang.min()).T
        ap_yihang_min['all_id_two'] = id_this
        ap_min = ap_min.append(ap_yihang_min)
record_and_ap = pd.merge(record_and_ap,ap_mean_all,on = 'all_id_two', how = 'left')
record_and_ap = pd.merge(record_and_ap,fengsu_2.loc[:,['datetime','wse','wd','wind_direction']],left_on = 'date', right_on = 'datetime',how = 'left')
record_and_ap_min = pd.merge(all_exp_record,ap_min,on='all_id_two', how = 'left')

#%%
wuranwu_name = record_and_ap.columns.values.tolist()
wuranwu_name = wuranwu_name[21:]
wuranwu_name = ['PM_0.3', 'PM_0.5', 'PM_1', 'PM_3', 'PM_5', 'PM_10',
                'PM1', 'PM2.5', 'RESP', 'PM10', 'TOTAL',
                'bc_ir',
                'temperature', 'RH', 'dew',
                'wse', 'wind_direction_y']
wuranwu_ylabel = ['Number concentration (#$/m^3$)','Number concentration (#$/m^3$)','Number concentration (#$/m^3$)',
                  'Number concentration (#$/m^3$)','Number concentration (#$/m^3$)','Number concentration (#$/m^3$)',                 
                  'Concenreation ($\mu g/m^3$)','Concenreation ($\mu g/m^3$)','Concenreation ($\mu g/m^3$)',
                  'Concenreation ($\mu g/m^3$)','Concenreation ($\mu g/m^3$)',
                  'Concenreation ($ng /m^3$)', 
                  'Temperature (°C)', 'RH (%)', 'Dew (°C)',
                  'Wind Level', 'Wind Direction (°)'
                  ]
wuranwu_title = ['PM$_{0.3}$ Number Concentration','PM$_{0.5}$ Number Concentration','PM$_{1}$ Number Concentration',
                 'PM$_{3}$ Number Concentration','PM$_{5}$ Number Concentration','PM$_{10}$ Number Concentration',
                 'PM$_{1}$ Mass Concentration','PM$_{2.5}$ Mass Concentration','RESP Mass Concentration',
                 'PM$_{10}$ Mass Concentration','Total PM Mass Concentration',
                 'Black Carbon Mass Concentration',
                 'Temperature','RH','Dew',
                 'Wind Level','Wind Direction'
                 ] 
#%%
flight_id = di(record_and_ap['flight'].drop_duplicates())
#flight_id = di(flight_id[flight_id!=26])

#%%

record_and_ap_min_day = record_and_ap_min.groupby(record_and_ap_min['date'].dt.date).min()
record_and_ap_min_flight = record_and_ap_min.groupby(record_and_ap_min['flight']).min()
record_and_ap_qu_back = record_and_ap.iloc[:,21:36]
record_and_ap_qu_back['flight'] = record_and_ap['flight']
record_and_ap_paixu = di(record_and_ap.sort_values(by = ['flight'],ascending = True))
record_and_ap_qu_back = di(record_and_ap_qu_back.sort_values(by = ['flight'],ascending = True))
record_and_ap_qu_back_jieguo = pd.DataFrame()
for i in range(0,len(flight_id)):
    print(str(i)+': '+str(flight_id[i]))
    record_and_ap_qu_back_jieguo = record_and_ap_qu_back_jieguo.append(di(record_and_ap_qu_back[record_and_ap_qu_back['flight']==flight_id[i]].iloc[:,0:15] - 
                                                                       record_and_ap_min_flight.iloc[i,21:36]))
record_and_ap_qu_back_jieguo = di(record_and_ap_qu_back_jieguo)
record_and_ap_qu_back_jieguo['all_id_two'] = record_and_ap_paixu['all_id_two']
record_and_ap_qu_back_jieguo =  pd.merge(all_exp_record,record_and_ap_qu_back_jieguo,on='all_id_two', how = 'left')
#%%
wuranwu_name = ['PM_0.3', 'PM_0.5', 'PM_1', 'PM_3', 'PM_5', 'PM_10',
                'PM1', 'PM2.5',  'PM10', 
                'bc_ir',
                'temperature', 'RH', 'dew']
wuranwu_ylabel = ['Number concentration (#$/m^3$)','Number concentration (#$/m^3$)','Number concentration (#$/m^3$)',
                  'Number concentration (#$/m^3$)','Number concentration (#$/m^3$)','Number concentration (#$/m^3$)',                 
                  'Concenreation ($\mu g/m^3$)','Concenreation ($\mu g/m^3$)','Concenreation ($\mu g/m^3$)',
                  'Concenreation ($ng /m^3$)', 
                  'Temperature (°C)', 'RH (%)', 'Dew (°C)',
                  ]
wuranwu_title = ['PM$_{0.3}$ Number Concentration','PM$_{0.5}$ Number Concentration','PM$_{1}$ Number Concentration',
                 'PM$_{3}$ Number Concentration','PM$_{5}$ Number Concentration','PM$_{10}$ Number Concentration',
                 'PM$_{1}$ Mass Concentration','PM$_{2.5}$ Mass Concentration','PM$_{10}$ Mass Concentration',
                 'Black Carbon Mass Concentration',
                 'Temperature','RH','Dew',
                 ] 


#%%
flight_id = di(flight_id[flight_id!=26])


wuranwu_name = ['PM_0.3', 'PM_0.5', 'PM_1', 'PM_3', 'PM_5', 'PM_10']
wuranwu_ylabel = ['Number concentration (#$/m^3$)','Number concentration (#$/m^3$)','Number concentration (#$/m^3$)',
                  'Number concentration (#$/m^3$)','Number concentration (#$/m^3$)','Number concentration (#$/m^3$)']
wuranwu_title = ['PM$_{0.3}$ Number Concentration','PM$_{0.5}$ Number Concentration','PM$_{1}$ Number Concentration',
                 'PM$_{3}$ Number Concentration','PM$_{5}$ Number Concentration','PM$_{10}$ Number Concentration']
wuranwu_title_new = ['(a)','(b)','(c)','(d)','(e)','(f)']


fig, ax = plt.subplots(figsize=(18, 8))
sns.set(font_scale=1,font = 'Times New Roman')
for j in range(0,len(wuranwu_name)):
    plt.subplot(2,3,j+1)
    legend_label = []
    for i in range(0,len(flight_id)):
        one_id = di(record_and_ap_qu_back_jieguo[record_and_ap_qu_back_jieguo['flight'] == flight_id[i]])
        if (one_id['mode'][0] == 'lashen') & (len(one_id[wuranwu_name[j]].dropna()) == len(one_id)):
            if (one_id['Place'][0] == 'Laojianhe'):
                print(one_id['Place'][0] +' ' + str(one_id['datetime_begin'][0].date()) + ' wind: '+one_id['wind_direction'][0] +' '+str(one_id['wind_level'][0]))
                if (one_id['date'][0] < dt.datetime(2019,2,1)):
                    plt.plot(one_id['Distance'],one_id[wuranwu_name[j]],'o-',label = one_id['Place'][0][0] +' ' + str(one_id['datetime_begin'][0].date()) +' '+ str(one_id['datetime_begin'][0].hour))
                else:
                    plt.plot(one_id['Distance'],one_id[wuranwu_name[j]],'^-',label = one_id['Place'][0][0] +' ' + str(one_id['datetime_begin'][0].date()) +' '+ str(one_id['datetime_begin'][0].hour))        
            else:
                if (one_id['date'][0] < dt.datetime(2019,2,1)):
                    plt.plot(one_id['Distance'],one_id[wuranwu_name[j]],'o--',label = one_id['Place'][0][0] +' ' + str(one_id['datetime_begin'][0].date()) +' '+ str(one_id['datetime_begin'][0].hour))
                else:
                    plt.plot(one_id['Distance'],one_id[wuranwu_name[j]],'^--',label = one_id['Place'][0][0] +' ' + str(one_id['datetime_begin'][0].date()) +' '+ str(one_id['datetime_begin'][0].hour))
    plt.xlabel('Distance (m)')
    plt.ylabel(wuranwu_ylabel[j])
    plt.title(wuranwu_title_new[j])
#    plt.title(wuranwu_title[j])
    plt.axvline(x=0,color = 'black',ls = '--')
    plt.legend()
    print(len(legend_label))
plt.tight_layout()
#plt.savefig(results_path+'fig_py/'+'fig_0707_PNC_danwei.jpg',dpi = 300,bbox_inches = 'tight')


#fig, ax = plt.subplots(figsize=(12, 14))
#sns.set(font_scale=1,font = 'Times New Roman')
#for j in range(0,len(wuranwu_name)):
#    plt.subplot(2,3,j+1)
#    legend_label = []
#    for i in range(0,len(flight_id)):
#        one_id = di(record_and_ap_qu_back_jieguo[record_and_ap_qu_back_jieguo['flight'] == flight_id[i]])
#        if (one_id['mode'][0] == 'chuizhi') & (len(one_id[wuranwu_name[j]].dropna()) == len(one_id)):
#            print(one_id['Place'][0] +' ' + str(one_id['datetime_begin'][0].date()) + ' wind: '+one_id['wind_direction'][0]+' '+str(one_id['wind_level'][0]))
#            if (one_id['Place'][0] == 'Laojianhe'):
#                if (one_id['date'][0] < dt.datetime(2019,2,1)):
#                    plt.plot(one_id[wuranwu_name[j]],one_id['Height'],'o-',label = one_id['Place'][0] +' ' + str(one_id['datetime_begin'][0].date()))
#                else:
#                    plt.plot(one_id[wuranwu_name[j]],one_id['Height'],'^-',label = one_id['Place'][0] +' ' + str(one_id['datetime_begin'][0].date()))
#            else:
#                if (one_id['date'][0] < dt.datetime(2019,2,1)):
#                    plt.plot(one_id[wuranwu_name[j]],one_id['Height'],'o--',label = one_id['Place'][0] +' ' + str(one_id['datetime_begin'][0].date()))
#                else:
#                    plt.plot(one_id[wuranwu_name[j]],one_id['Height'],'^--',label = one_id['Place'][0] +' ' + str(one_id['datetime_begin'][0].date()))
#    plt.ylabel('Height (m)')
#    plt.xlabel(wuranwu_ylabel[j])
#    plt.title(wuranwu_title_new[j])
##    plt.axvline(x=0,color = 'black',ls = '--')
#    plt.legend()
#    print(len(legend_label))
#plt.tight_layout()
##plt.savefig(results_path+'fig_py/'+'fig_0701_PNC_chuizhi.jpg',dpi = 300,bbox_inches = 'tight')


#%%
#record_and_ap_qu_back_jieguo.to_excel(data_file_path+'daochu_0701_record_and_ap_quchu_back.xlsx')
part_1_laojianhe_53 = record_and_ap_qu_back_jieguo[(record_and_ap_qu_back_jieguo['flight'] == 53)]
part_1_laojianhe_54 = record_and_ap_qu_back_jieguo[(record_and_ap_qu_back_jieguo['flight'] == 54)]


#%%分析老减河上下风向
gai_record_and_ap_qu_back_jieguo = pd.read_excel(data_file_path+'daochu_0701_record_and_ap_quchu_back_xiugai.xlsx')

flight_id = di(gai_record_and_ap_qu_back_jieguo['flight'].drop_duplicates())
flight_id = di(flight_id[flight_id!=26])


fig, ax = plt.subplots(figsize=(18, 8))
sns.set(font_scale=1,font = 'Times New Roman')
for j in range(0,len(wuranwu_name)):
    plt.subplot(2,3,j+1)
    legend_label = []
    for i in range(0,len(flight_id)):
        one_id = di(gai_record_and_ap_qu_back_jieguo[gai_record_and_ap_qu_back_jieguo['flight'] == flight_id[i]])
        if (one_id['mode'][0] == 'lashen') & (len(one_id[wuranwu_name[j]].dropna()) == len(one_id)):
            print(one_id['Place'][0] +' ' + str(one_id['datetime_begin'][0].date()) + ' wind: '+one_id['wind_direction'][0] +' '+str(one_id['wind_level'][0]))
            if (one_id['Place'][0] == 'Laojianhe'):
                print(one_id['Place'][0] +' ' + str(one_id['datetime_begin'][0].date()) + ' wind: '+one_id['wind_direction'][0] +' '+str(one_id['wind_level'][0]))
                if (one_id['date'][0] < dt.datetime(2019,2,1)):
                    plt.plot(one_id['Distance'],one_id[wuranwu_name[j]],'o-',label = one_id['Place'][0][0] +' ' + str(one_id['datetime_begin'][0].date()) +' '+ str(one_id['datetime_begin'][0].hour))
                else:
                    plt.plot(one_id['Distance'],one_id[wuranwu_name[j]],'^-',label = one_id['Place'][0][0] +' ' + str(one_id['datetime_begin'][0].date()) +' '+ str(one_id['datetime_begin'][0].hour))        
            else:
                if (one_id['date'][0] < dt.datetime(2019,2,1)):
                    plt.plot(one_id['Distance'],one_id[wuranwu_name[j]],'o--',label = one_id['Place'][0][0] +' ' + str(one_id['datetime_begin'][0].date()) +' '+ str(one_id['datetime_begin'][0].hour))
                else:
                    plt.plot(one_id['Distance'],one_id[wuranwu_name[j]],'^--',label = one_id['Place'][0][0] +' ' + str(one_id['datetime_begin'][0].date()) +' '+ str(one_id['datetime_begin'][0].hour))
    plt.xlabel('Distance (m)')
    plt.ylabel(wuranwu_ylabel[j])
    plt.title(wuranwu_title_new[j])
    plt.axvline(x=0,color = 'black',ls = '--')
    plt.legend()
    print(len(legend_label))
plt.tight_layout()
#plt.savefig(results_path+'fig_py/'+'fig_0718_PNC_danwei_mac_new.eps',dpi = 300,bbox_inches = 'tight')



#fig, ax = plt.subplots(figsize=(12, 14))
#sns.set(font_scale=1,font = 'Times New Roman')
#for j in range(0,len(wuranwu_name)):
#    plt.subplot(2,3,j+1)
#    legend_label = []
#    for i in range(0,len(flight_id)):
#        one_id = di(gai_record_and_ap_qu_back_jieguo[gai_record_and_ap_qu_back_jieguo['flight'] == flight_id[i]])
#        if (one_id['mode'][0] == 'chuizhi') & (len(one_id[wuranwu_name[j]].dropna()) == len(one_id)):
#            print(one_id['Place'][0] +' ' + str(one_id['datetime_begin'][0].date()) + ' wind: '+one_id['wind_direction'][0]+' '+str(one_id['wind_level'][0]))
#            if (one_id['Place'][0] == 'Laojianhe'):
#                if (one_id['date'][0] < dt.datetime(2019,2,1)):
#                    plt.plot(one_id[wuranwu_name[j]],one_id['Height'],'o-',label = one_id['Place'][0] +' ' + str(one_id['datetime_begin'][0].date()))
#                else:
#                    plt.plot(one_id[wuranwu_name[j]],one_id['Height'],'^-',label = one_id['Place'][0] +' ' + str(one_id['datetime_begin'][0].date()))
#            else:
#                if (one_id['date'][0] < dt.datetime(2019,2,1)):
#                    plt.plot(one_id[wuranwu_name[j]],one_id['Height'],'o--',label = one_id['Place'][0] +' ' + str(one_id['datetime_begin'][0].date()))
#                else:
#                    plt.plot(one_id[wuranwu_name[j]],one_id['Height'],'^--',label = one_id['Place'][0] +' ' + str(one_id['datetime_begin'][0].date()))
#    plt.ylabel('Height (m)')
#    plt.xlabel(wuranwu_ylabel[j])
#    plt.title(wuranwu_title_new[j])
##    plt.axvline(x=0,color = 'black',ls = '--')
#    plt.legend()
#    print(len(legend_label))
#plt.tight_layout()
##plt.savefig(results_path+'fig_py/'+'fig_0701_PNC_chuizhi.jpg',dpi = 300,bbox_inches = 'tight')

#%%
#j=0
#fig = plt.figure(figsize=(18, 8))
##f = plt.figure()
##f, axes = plt.subplots(2, 3)
#tips = gai_record_and_ap_qu_back_jieguo[gai_record_and_ap_qu_back_jieguo["mode"]=="lashen"]
#sns.set(font_scale=1,font = 'Times New Roman')
#
#for j in range(0,6):
##    fig.add_subplot(2, 3, j+1)
##    plt.subplot(2,3,j+1)
#    axm = fig.add_subplot(2,3,j+1)
#    sns.catplot(x="Place", y=wuranwu_name[j], showfliers=False,
#                hue = "Campaign",kind="box", data=tips,ax = axm,width = 0.4);
#    axm.set_ylabel(wuranwu_ylabel[j])
##    plt.title(wuranwu_title_new[j])
#plt.show()
#
#j=0
#fig = plt.figure(figsize=(18, 8))
#sns.catplot(x="Distance", y="bc_ir", showfliers=False,
#            hue = "Place",col = 'Campaign',kind="point", data=tips,markers=["^", "o"], linestyles=["-", "--"],capsize=.2,height = 10);

#%%接下来是PMC和BC
wuranwu_name = ['PM1', 'PM2.5',  'PM10','pm1_25','pm_25_10',
                'bc_ir']
wuranwu_ylabel = ['Concenreation ($mg/m^3$)','Concenreation ($mg/m^3$)','Concenreation ($mg/m^3$)','PM$_1$ / PM$_{2.5}$ Ratio (%)','PM$_{2.5}$ / PM$_{10}$ Ratio (%)',
                  'Black Carbon Mass Concentration']
wuranwu_title = ['PM$_{1}$ Mass Concentration','PM$_{2.5}$ Mass Concentration','PM$_{10}$ Mass Concentration',
                 'Concenreation ($ng /m^3$)']
gai_record_and_ap_qu_back_jieguo = pd.read_excel(data_file_path+'daochu_0704_record_and_ap_quchu_back_xiugai.xlsx')
fig, ax = plt.subplots(figsize=(18, 8))
sns.set(font_scale=1.5,font = 'Times New Roman')
all_one_id_pm_ratio = pd.DataFrame()
for j in range(0,len(wuranwu_name)):
    plt.subplot(2,3,j+1)
    legend_label = []
    for i in range(0,len(flight_id)):
        one_id = di(gai_record_and_ap_qu_back_jieguo[gai_record_and_ap_qu_back_jieguo['flight'] == flight_id[i]])
        one_id['PM1'] = one_id['PM1']*1000
        one_id['PM2.5'] = one_id['PM2.5']*1000 +one_id['PM1']
        one_id['PM10'] = one_id['PM10']*1000 +one_id['PM2.5']
        one_id['pm1_25'] = one_id['PM1']/one_id['PM2.5']
        one_id['pm_25_10'] = one_id['PM2.5']/one_id['PM10']
        
        if (one_id['mode'][0] == 'lashen') & (len(one_id[wuranwu_name[j]].dropna()) == len(one_id)):
            print(one_id['Place'][0] +' ' + str(one_id['datetime_begin'][0].date()) + ' wind: '+one_id['wind_direction'][0] +' '+str(one_id['wind_level'][0]))
            print(one_id)
            
            if (one_id['Place'][0] == 'Laojianhe'):
                print(one_id['Place'][0] +' ' + str(one_id['datetime_begin'][0].date()) + ' wind: '+one_id['wind_direction'][0] +' '+str(one_id['wind_level'][0]))
                if (one_id['date'][0] < dt.datetime(2019,2,1)):
                    plt.plot(one_id['Distance'],one_id[wuranwu_name[j]],'o-',label = one_id['Place'][0][0] +' ' + str(one_id['datetime_begin'][0].date()) +' '+ str(one_id['datetime_begin'][0].hour))
                else:
                    plt.plot(one_id['Distance'],one_id[wuranwu_name[j]],'^-',label = one_id['Place'][0][0] +' ' + str(one_id['datetime_begin'][0].date()) +' '+ str(one_id['datetime_begin'][0].hour))        
            else:
                if (one_id['date'][0] < dt.datetime(2019,2,1)):
                    plt.plot(one_id['Distance'],one_id[wuranwu_name[j]],'o--',label = one_id['Place'][0][0] +' ' + str(one_id['datetime_begin'][0].date()) +' '+ str(one_id['datetime_begin'][0].hour))
                else:
                    plt.plot(one_id['Distance'],one_id[wuranwu_name[j]],'^--',label = one_id['Place'][0][0] +' ' + str(one_id['datetime_begin'][0].date()) +' '+ str(one_id['datetime_begin'][0].hour))
    plt.xlabel('Distance (m)')
    plt.ylabel(wuranwu_ylabel[j])
    plt.title(wuranwu_title_new[j])
    plt.axvline(x=0,color = 'black',ls = '--')
    plt.legend()
    print(len(legend_label))
plt.tight_layout()

#plt.savefig(results_path+'fig_py/'+'fig_0718_PMC_BC_mac.eps',dpi = 300,bbox_inches = 'tight')

#%% 黑炭+温湿度
wuranwu_name = ['bc_ir','temperature', 'RH']
wuranwu_ylabel = ['Concenreation ($ng /m^3$)','Temperature (°C)', 'RH (%)']
wuranwu_title = ['Black Carbon Mass Concentration','Temperature','RH'] 
gai_record_and_ap_qu_back_jieguo = pd.read_excel(data_file_path+'daochu_0701_record_and_ap_quchu_back_xiugai.xlsx')
flight_id = di(gai_record_and_ap_qu_back_jieguo['flight'].drop_duplicates())
flight_id = di(flight_id[flight_id!=26])

hua_trh = gai_record_and_ap_qu_back_jieguo.copy()
hua_trh['temperature'] = record_and_ap_qu_back['temperature']
hua_trh['RH'] = record_and_ap_qu_back['RH']

fig, ax = plt.subplots(figsize=(18, 8))
sns.set(font_scale=1,font = 'Times New Roman')
for j in range(0,len(wuranwu_name)):
    plt.subplot(2,3,j+1)
    legend_label = []
    for i in range(0,len(flight_id)):
        one_id = di(hua_trh[hua_trh['flight'] == flight_id[i]])
        if (one_id['mode'][0] == 'lashen') & (len(one_id[wuranwu_name[j]].dropna()) == len(one_id)):
            print(one_id['Place'][0] +' ' + str(one_id['datetime_begin'][0].date()) + ' wind: '+one_id['wind_direction'][0] +' '+str(one_id['wind_level'][0]))
            if (one_id['Place'][0] == 'Laojianhe'):
                print(one_id['Place'][0] +' ' + str(one_id['datetime_begin'][0].date()) + ' wind: '+one_id['wind_direction'][0] +' '+str(one_id['wind_level'][0]))
                if (one_id['date'][0] < dt.datetime(2019,2,1)):
                    plt.plot(one_id['Distance'],one_id[wuranwu_name[j]],'o-',label = one_id['Place'][0][0] +' ' + str(one_id['datetime_begin'][0].date()) +' '+ str(one_id['datetime_begin'][0].hour))
                else:
                    plt.plot(one_id['Distance'],one_id[wuranwu_name[j]],'^-',label = one_id['Place'][0][0] +' ' + str(one_id['datetime_begin'][0].date()) +' '+ str(one_id['datetime_begin'][0].hour))        
            else:
                if (one_id['date'][0] < dt.datetime(2019,2,1)):
                    plt.plot(one_id['Distance'],one_id[wuranwu_name[j]],'o--',label = one_id['Place'][0][0] +' ' + str(one_id['datetime_begin'][0].date()) +' '+ str(one_id['datetime_begin'][0].hour))
                else:
                    plt.plot(one_id['Distance'],one_id[wuranwu_name[j]],'^--',label = one_id['Place'][0][0] +' ' + str(one_id['datetime_begin'][0].date()) +' '+ str(one_id['datetime_begin'][0].hour))
    plt.xlabel('Distance (m)')
    plt.ylabel(wuranwu_ylabel[j])
    plt.title(wuranwu_title_new[j])
    plt.axvline(x=0,color = 'black',ls = '--')
    plt.legend()
    print(len(legend_label))
plt.tight_layout()
#plt.savefig(results_path+'fig_py/'+'fig_0704_BC_shuiping_new_label.jpg',dpi = 300,bbox_inches = 'tight')




#%% 单独黑炭
wuranwu_name = ['bc_ir']
wuranwu_ylabel = ['Concenreation ($ng /m^3$)']
wuranwu_title = ['Black Carbon Mass Concentration'] 
gai_record_and_ap_qu_back_jieguo = pd.read_excel(data_file_path+'daochu_0704_record_and_ap_quchu_back_xiugai.xlsx')
flight_id = di(gai_record_and_ap_qu_back_jieguo['flight'].drop_duplicates())
flight_id = di(flight_id[flight_id!=26])

hua_trh = gai_record_and_ap_qu_back_jieguo.copy()

plt.rcParams['font.size'] = 16
fig, ax = plt.subplots(figsize=(18, 6))
sns.set(font_scale=1.5,font = 'Times New Roman')
for j in range(0,len(wuranwu_name)):
    plt.subplot(1,2,j+1)
    legend_label = []
    for i in range(0,len(flight_id)):
        one_id = di(hua_trh[hua_trh['flight'] == flight_id[i]])
        if (one_id['mode'][0] == 'lashen') & (len(one_id[wuranwu_name[j]].dropna()) == len(one_id)):
            print(one_id['Place'][0] +' ' + str(one_id['datetime_begin'][0].date()) + ' wind: '+one_id['wind_direction'][0] +' '+str(one_id['wind_level'][0]))
            if (one_id['Place'][0] == 'Laojianhe'):
                print(one_id['Place'][0] +' ' + str(one_id['datetime_begin'][0].date()) + ' wind: '+one_id['wind_direction'][0] +' '+str(one_id['wind_level'][0]))
                if (one_id['date'][0] < dt.datetime(2019,2,1)):
                    plt.plot(one_id['Distance'],one_id[wuranwu_name[j]],'o-',label = one_id['Place'][0][0] +' ' + str(one_id['datetime_begin'][0].date()) +' '+ str(one_id['datetime_begin'][0].hour))
                else:
                    plt.plot(one_id['Distance'],one_id[wuranwu_name[j]],'^-',label = one_id['Place'][0][0] +' ' + str(one_id['datetime_begin'][0].date()) +' '+ str(one_id['datetime_begin'][0].hour))        
            else:
                if (one_id['date'][0] < dt.datetime(2019,2,1)):
                    plt.plot(one_id['Distance'],one_id[wuranwu_name[j]],'o--',label = one_id['Place'][0][0] +' ' + str(one_id['datetime_begin'][0].date()) +' '+ str(one_id['datetime_begin'][0].hour))
                else:
                    plt.plot(one_id['Distance'],one_id[wuranwu_name[j]],'^--',label = one_id['Place'][0][0] +' ' + str(one_id['datetime_begin'][0].date()) +' '+ str(one_id['datetime_begin'][0].hour))
    plt.xlabel('Distance (m)')
    plt.ylabel(wuranwu_ylabel[j])
#    plt.title(wuranwu_title_new[j])
    plt.axvline(x=0,color = 'black',ls = '--')
    plt.legend()
    print(len(legend_label))
plt.tight_layout()
#plt.savefig(results_path+'fig_py/'+'fig_0704_BC_only_shuiping_new_label.jpg',dpi = 300,bbox_inches = 'tight')

#%%

record_and_ap_0726 = pd.read_excel(data_file_path+'daochu_0726_record_and_ap_yuanshi.xlsx')


group_by_date = record_and_ap_0726.groupby([record_and_ap_0726["date"].dt.date]).mean()

#group_by_campaign = record_and_ap_0726.groupby([record_and_ap_0726["campaign"]]).mean()
#group_by_campaign_std = record_and_ap.groupby([record_and_ap["Campaign"]]).std()
#group_by_campaign_min = record_and_ap.groupby([record_and_ap["Campaign"]]).min()
#group_by_campaign_max = record_and_ap.groupby([record_and_ap["Campaign"]]).max()


group_by_distance = record_and_ap.groupby(record_and_ap["Distance"]).mean()


#%%
record_and_ap_0726 = pd.read_excel(data_file_path+'daochu_0726_record_and_ap_yuanshi.xlsx')
group_by_campaign_L = record_and_ap_0726[record_and_ap_0726["Place"]=="Laojianhe"].describe()
group_by_campaign_S = record_and_ap_0726[record_and_ap_0726["Place"]=="Shuniu"].describe()
group_by_campaign = record_and_ap_0726.describe()

group_by_campaign_winter = record_and_ap_0726[record_and_ap_0726["campaign"]=="winter"].describe()
group_by_campaign_spring = record_and_ap_0726[record_and_ap_0726["campaign"]=="spring"].describe()



#%% 一起画
wuranwu_name = ['PM_0.3', 'PM_0.5', 'PM_1', 'PM_3', 'PM_5', 'PM_10',
                'PM1', 'PM2.5',  'PM10','pm1_25','pm_25_10',
                'bc_ir']
wuranwu_ylabel = ['Number concentration (#$/m^3$)','Number concentration (#$/m^3$)','Number concentration (#$/m^3$)',
                  'Number concentration (#$/m^3$)','Number concentration (#$/m^3$)','Number concentration (#$/m^3$)',
                  'Concenreation ($\mu g/m^3$)','Concenreation ($\mu g/m^3$)','Concenreation ($\mu g/m^3$)',
                  'PM$_1$ / PM$_{2.5}$ Ratio (%)','PM$_{2.5}$ / PM$_{10}$ Ratio (%)',
                  'Black Carbon Mass Concentration']
wuranwu_title_new = ['(a)','(b)','(c)','(d)','(e)','(f)',
                     '(g)','(h)','(i)','(j)','(k)',
                     '(l)']
wuranwu_ylabel_1 = ['(#$/m^3$)','(#$/m^3$)','(#$/m^3$)','(#$/m^3$)','(#$/m^3$)','(#$/m^3$)',
                  '($\mu g/m^3$)','($\mu g/m^3$)','($\mu g/m^3$)',
                  '(%)','(%)',
                  '($ng /m^3$)'
                  ]
wuranwu_legend = ['N$_{0.3}$','N$_{0.5}$','N$_{1}$','N$_{3}$','N$_{5}$','N$_{10}$',
                  'PM$_{1}$','PM$_{2.5}$','PM$_{10}$','PM$_{1}$/PM$_{2.5}$ Ratio','PM$_{2.5}$/PM$_{10}$ Ratio',
                  'BC']
wuranwu_title = ['PM$_{0.3}$ Number Concentration','PM$_{0.5}$ Number Concentration','PM$_{1}$ Number Concentration',
                 'PM$_{3}$ Number Concentration','PM$_{5}$ Number Concentration','PM$_{10}$ Number Concentration',
                 'PM$_{1}$ Mass Concentration','PM$_{2.5}$ Mass Concentration','PM$_{10}$ Mass Concentration',
                 'Black Carbon Mass Concentration',
                 ]
wuranwu_ylabel = ['Number concentration (#$/m^3$)','Number concentration (#$/m^3$)','Number concentration (#$/m^3$)',
                  'Number concentration (#$/m^3$)','Number concentration (#$/m^3$)','Number concentration (#$/m^3$)',
                  'Mass Concenreation ($\mu g/m^3$)','Mass Concenreation ($\mu g/m^3$)','Mass Concenreation ($\mu g/m^3$)',
                  'Ratio (%)','Ratio (%)',
                  'Concenreation ($ng /m^3$)'
                  ]
gai_record_and_ap_qu_back_jieguo = pd.read_excel(data_file_path+'daochu_0704_record_and_ap_quchu_back_xiugai.xlsx')
fig, ax = plt.subplots(figsize=(18, 16))
sns.set(font_scale=1.3,font = 'Times New Roman')
all_one_id_pm_ratio = pd.DataFrame()
ms=10
for j in range(0,len(wuranwu_name)):
    plt.subplot(4,3,j+1)
    legend_label = []
    for i in range(0,len(flight_id)):
        one_id = di(gai_record_and_ap_qu_back_jieguo[gai_record_and_ap_qu_back_jieguo['flight'] == flight_id[i]])
#        one_id['PM1'] = one_id['PM1']*1000
#        one_id['PM2.5'] = one_id['PM2.5']*1000 +one_id['PM1']
#        one_id['PM10'] = one_id['PM10']*1000 +one_id['PM2.5']
#        one_id['pm1_25'] = one_id['PM1']/one_id['PM2.5']
#        one_id['pm_25_10'] = one_id['PM2.5']/one_id['PM10']
        
        one_id['PM1'] = one_id['PM1']*1000
        one_id['PM2.5'] = one_id['PM2.5']*1000
        one_id['PM10'] = one_id['PM10']*1000
        one_id['pm1_25'] = one_id['PM1']/(one_id['PM2.5']+one_id['PM1']/2)
        one_id['pm_25_10'] = (one_id['PM2.5']+one_id['PM1']/2)/one_id['PM10']
        if (one_id['mode'][0] == 'lashen') & (len(one_id[wuranwu_name[j]].dropna()) >= 3):
#            print(one_id['Place'][0] +' ' + str(one_id['datetime_begin'][0].date()) + ' wind: '+one_id['wind_direction'][0] +' '+str(one_id['wind_level'][0]))
#            print(one_id)
            one_id_hua = one_id.copy()
            fuzhi_1 = one_id_hua[wuranwu_name[j]].dropna().index.tolist()
            one_id_hua = one_id_hua.loc[fuzhi_1,['date','Place','datetime_begin','Distance',wuranwu_name[j]]] 

            if (one_id['Place'][0] == 'Laojianhe'):
#                print(one_id_hua['Place'][0] +' ' + str(one_id_hua['datetime_begin'][0].date()) + ' wind: '+one_id_hua['wind_direction'][0] +' '+str(one_id_hua['wind_level'][0]))
                if (one_id_hua['date'][0] < dt.datetime(2019,2,1)):
                    plt.plot(one_id_hua['Distance'],one_id_hua[wuranwu_name[j]],'-', marker = 'o',markersize=ms, label = one_id_hua['Place'][0][0] +' ' + str(one_id_hua['datetime_begin'][0].date()) +' '+ str(one_id_hua['datetime_begin'][0].hour))
                else:
                    plt.plot(one_id_hua['Distance'],one_id_hua[wuranwu_name[j]],'-',marker = '^',markersize=ms,label = one_id_hua['Place'][0][0] +' ' + str(one_id_hua['datetime_begin'][0].date()) +' '+ str(one_id_hua['datetime_begin'][0].hour))        
            else:
                if (one_id['date'][0] < dt.datetime(2019,2,1)):
                    plt.plot(one_id_hua['Distance'],one_id_hua[wuranwu_name[j]],'--',marker = 'o',markersize=ms,label = one_id_hua['Place'][0][0] +' ' + str(one_id_hua['datetime_begin'][0].date()) +' '+ str(one_id_hua['datetime_begin'][0].hour))
                else:
                    plt.plot(one_id_hua['Distance'],one_id_hua[wuranwu_name[j]],'--',marker = '^',markersize=ms, label = one_id_hua['Place'][0][0] +' ' + str(one_id_hua['datetime_begin'][0].date()) +' '+ str(one_id_hua['datetime_begin'][0].hour))
#            one_id_hua = di(one_id.dropna())
    plt.xlabel('Distance (m)')
    plt.ylabel(wuranwu_ylabel[j])
    plt.title(wuranwu_legend[j])
    plt.axvline(x=0,color = 'black',ls = '--')
    plt.legend()
    print(len(legend_label))
plt.tight_layout()

#plt.savefig(results_path+'fig_py/'+'fig_2020_0124_all_shuiping_label.svg',dpi =    300,bbox_inches = 'tight')

#%%
#record_and_ap.to_excel(data_file_path+'data_0727_record_and_ap_original.xlsx')






#%%
names = locals()


pwd = os.getcwd()
print(pwd)
uav_file_path = pwd+'/flight/'
data_file_path = pwd+'/data/'
results_path = pwd+'/results/'
qixiang_file_path = pwd+'/data/qixiang/'

#%%

data_name = ['9306','8534','bc','trh']
monitor_file = []
#all_exp_record = pd.read_excel(data_file_path+'both_exp_record_0506.xlsx',encoding = 'gbk')
all_exp_record = pd.read_excel(data_file_path+'all_exp_record_0522.xlsx',encoding = 'gbk')

all_exp_record['datetime_begin'] = pd.to_datetime(all_exp_record['datetime_begin'])
all_exp_record['datetime_end'] = pd.to_datetime(all_exp_record['datetime_end'])


for i in range(0,len(data_name)):
    monitor_file.append(pd.read_excel(data_file_path+'all_'+data_name[i]+'.xlsx'))
    monitor_file[i]['datetime'] = pd.to_datetime(monitor_file[i]['datetime'])

figure, ax = plt.subplots()
plt.rcParams['font.family'] = 'Times New Roman'
plt.rcParams['font.size'] = 16
figure.suptitle('Remove the abnormal value')
plt.subplot(221)
plt.plot(monitor_file[2]['bc_ir'])
plt.subplot(222)
yichangzhi = monitor_file[2][monitor_file[2]['bc_ir']>1000000].index.tolist()
monitor_file[2].loc[yichangzhi,'bc_ir'] = np.nan
plt.plot(monitor_file[2]['bc_ir'])
plt.subplot(223)
yichangzhi = monitor_file[2][monitor_file[2]['bc_ir']<0].index.tolist()
monitor_file[2].loc[yichangzhi,'bc_ir'] = np.nan
plt.plot(monitor_file[2]['bc_ir'])
plt.subplot(224)
yichangzhi = monitor_file[2][monitor_file[2]['bc_ir']>300000].index.tolist()
monitor_file[2].loc[yichangzhi,'bc_ir'] = np.nan
plt.plot(monitor_file[2]['bc_ir'])
plt.show()

all_9306 = monitor_file[0]
all_8534 = monitor_file[1]
all_bc = monitor_file[2]
all_trh =monitor_file[3] 



#%%
flight_id = [1,2,3]
place = 'Laojianhe'
for i in range(0,len(flight_id)):
    names[place +'_'+ str(flight_id[i])] = all_exp_record[all_exp_record['flight'] == flight_id[i]]
    names[place +'_'+ str(flight_id[i])] = names[place +'_'+ str(flight_id[i])].reset_index(drop=True)

for m in range(0,len(flight_id)):
    for i in range(len(data_name)):
        mean_all = pd.DataFrame()
        for j in range(len(names[place +'_'+ str(flight_id[m])])):
            names[place +'_'+ data_name[i]] = names['all_' + data_name[i]][(names['all_' + data_name[i]]['datetime'] >=
                  names[place +'_' + str(flight_id[m])]['datetime_begin'][j]) & (names['all_' + data_name[i]]['datetime'] <=
                  names[place +'_' + str(flight_id[m])]['datetime_end'][j])]
            names[place +'_' + data_name[i]] = names[place +'_' + data_name[i]].reset_index(drop=True)
            mean_yige = pd.DataFrame(names[place +'_' + data_name[i]].mean()).T
            mean_all = mean_all.append(mean_yige)
            mean_all = mean_all.reset_index(drop = True)
        names[place +'_' + str(flight_id[m])] = pd.merge(names[place +'_' + str(flight_id[m])], mean_all, left_index=True, right_index=True)


#%%
record_and_ap = all_exp_record.copy()
for i in range(0,len(monitor_file)):
#i = 0
    ap_mean_all = pd.DataFrame()
    for k in range(0, len(all_exp_record)):
#    for k in range(176, 190):
        time_bg = all_exp_record.loc[k,'datetime_begin']
        time_ed = all_exp_record.loc[k,'datetime_end']
        id_this = all_exp_record.loc[k,'all_id_two']
        ap_yihang = monitor_file[i][(monitor_file[i]['datetime'] >= time_bg)&(monitor_file[i]['datetime'] <= time_ed)]
        if len(ap_yihang) >0:
            ap_yihang_mean = pd.DataFrame(ap_yihang.mean()).T
            ap_yihang_mean['all_id_two'] = id_this
            ap_mean_all = ap_mean_all.append(ap_yihang_mean)
    record_and_ap = pd.merge(record_and_ap,ap_mean_all,on = 'all_id_two', how = 'left')



#%%
sns.set()
#plt.plot(record_and_ap['temperature_x'],record_and_ap['temperature_y'],'o')
plt.figure(figsize=(16, 10))
plt.rcParams['font.family'] = 'Times New Roman'
#plt.rcParams['font.size'] = 5
z1 = np.polyfit(record_and_ap['temperature_local'], record_and_ap['temperature'], 1) # 用3次多项式拟合
p1 = np.poly1d(z1)
print(p1) # 在屏幕上打印拟合多项式
#plt.subplots_adjust(top=0.9)
sns.set(font_scale=2,font = 'Times New Roman')
g = sns.jointplot(x=record_and_ap['temperature_local'],y=record_and_ap['temperature'],kind = 'reg',size = 10)
g.fig.suptitle('(a)') # can also get the figure from plt.gcf()
g.set_axis_labels("Local Station Temperature ($°C$)", "Portable Devices Temperature ($°C$)");
g = g.annotate(stats.pearsonr)
plt.text(-9,20.5,'y = 0.9581 x + 0.408')

#plt.savefig(results_path+'duibi/duibi_wendu_0506_mac'+'.jpg', dpi=300,bbox_inches='tight')
plt.show()

z1 = np.polyfit(record_and_ap['RH_local'], record_and_ap['RH'], 1) # 用1次多项式拟合
p1 = np.poly1d(z1)
print(p1) # 在屏幕上打印拟合多项式
#plt.subplots_adjust(top=0.9)
plt.figure(figsize=(10, 6))
g = sns.jointplot(x=record_and_ap['RH_local'],y=record_and_ap['RH'],kind = 'reg',size = 10)
g.fig.suptitle('(b)') # can also get the figure from plt.gcf()
g = g.annotate(stats.pearsonr)
g.set_axis_labels("Local Station RH (%)", "Portable Devices RH (%)");
#for windows
#plt.text(35,67,'y = 0.8681 x + 9.389')
# for mac
plt.text(8,65,'y = 0.8681 x + 9.389')

#plt.savefig(results_path+'duibi/duibi_rh_0506_mac'+'.jpg', dpi=300,bbox_inches='tight')


#%% 对比pm
filepath_mac = '/Users/rc/Desktop/crr_all/2018_12_30_德州实验/cr'
filepath_windows = 'F:/crr/cloud/crr_all/2018_12_30_德州实验/cr'

filepath_path = filepath_windows

file_name = ["/8534.xlsx","/biaozhun.xlsx",'/hobo_wai.xlsx','/bc_1.xlsx','/bc_2_1331.xlsx']

i=0
duibi_8534 = pd.read_excel(filepath_path+file_name[i])
duibi_8534['datetime'] = pd.to_datetime(duibi_8534['time'])
duibi_re_8534 = resample_hour(duibi_8534,'datetime_8534')

i=1
duibi_biaozhun = pd.read_excel(filepath_path+file_name[i])

i=2
duibi_hobo = pd.read_excel(filepath_path+file_name[i])
duibi_hobo['datetime'] = pd.to_datetime(duibi_hobo['time'])
duibi_re_hobo = resample_hour(duibi_hobo,'datetime_hobo')

i=3
duibi_bc_1276 = pd.read_excel(filepath_path+file_name[i])
duibi_bc_1276['datetime'] = pd.to_datetime(duibi_bc_1276['time'])
duibi_bc_re_1276 = resample_hour(duibi_bc_1276,'datetime_bc_1276')

i=4
duibi_bc_1331 = pd.read_excel(filepath_path+file_name[i])
duibi_bc_1331['datetime'] = pd.to_datetime(duibi_bc_1331['time'])
duibi_bc_re_1331 = resample_hour(duibi_bc_1331,'datetime_bc_1331')




#%%

duibi_biaozhun_ap = pd.merge(duibi_biaozhun,duibi_re_8534,left_on = 'time',right_on = 'datetime_8534',how = 'left')
duibi_biaozhun_ap = pd.merge(duibi_biaozhun_ap,duibi_re_hobo,left_on = 'time',right_on = 'datetime_hobo',how = 'left')
duibi_biaozhun_ap = pd.merge(duibi_biaozhun_ap,duibi_bc_re_1276,left_on = 'time',right_on = 'datetime_bc_1276',how = 'left')
duibi_biaozhun_ap = pd.merge(duibi_biaozhun_ap,duibi_bc_re_1331,left_on = 'time',right_on = 'datetime_bc_1331',how = 'left')
duibi_biaozhun_ap.columns.values.tolist()
#duibi_biaozhun_ap.to_excel(data_file_path+'duibi_biaozhun_ap.xlsx')

#%%
duibi_biaozhun_lie = pd.DataFrame()
duibi_biaozhun_lie['PM_2_5'] = duibi_biaozhun_ap['PM2.5_x']
duibi_biaozhun_lie['PM_10'] = duibi_biaozhun_ap['PM10_x']
duibi_biaozhun_lie['temperature'] = duibi_biaozhun_ap['temperature']
duibi_biaozhun_lie['RH'] = duibi_biaozhun_ap['rh_x']
duibi_biaozhun_lie['BC1'] = duibi_biaozhun_ap['bc']
duibi_biaozhun_lie['BC2'] = duibi_biaozhun_ap['bc']

duibi_biaozhun_lie['station'] = 0
duibi_biaozhun_lie_2 = pd.DataFrame()
duibi_biaozhun_lie_2['PM_2_5']= duibi_biaozhun_ap['PM2.5_y']*1000
duibi_biaozhun_lie_2['PM_10']= duibi_biaozhun_ap['PM10_y']*1000
duibi_biaozhun_lie_2['temperature'] = duibi_biaozhun_ap['temp']
duibi_biaozhun_lie_2['RH'] = duibi_biaozhun_ap['rh_y']
duibi_biaozhun_lie_2['BC1'] = duibi_biaozhun_ap['conc_x']
duibi_biaozhun_lie_2['BC2'] = duibi_biaozhun_ap['conc_y']


duibi_biaozhun_lie_2['station'] = 1
duibi_hebing = duibi_biaozhun_lie.append(duibi_biaozhun_lie_2)

#duibi_hebing.to_excel(data_file_path+'duibi_hebing.xlsx')


#%%
sns.set(font_scale=1.5,font = 'Times New Roman')
plt.figure(figsize=(10, 6))
plt.rcParams['font.family'] = 'Times New Roman'
plt.rcParams['font.size'] = 16
plt.plot_date(duibi_biaozhun_ap['time'],duibi_biaozhun_ap['bc'],'o-',label = 'Monitoring station')
plt.plot_date(duibi_biaozhun_ap['time'],duibi_biaozhun_ap['conc_x']*1,'o-',label = 'microAeth AE51 Before')
plt.plot_date(duibi_biaozhun_ap['time'][1:],duibi_biaozhun_ap['conc_x'][0:len(duibi_biaozhun_ap)-1]*1,'o-',label = 'microAeth AE51 After')
plt.xlabel('Date and Time')
plt.ylabel('Black Carbon ($ng/m^3$)')
plt.legend()
plt.title('Black Carbon Comparison')
#plt.savefig(results_path+'duibi/duibi_pm25_0509_win'+'.jpg',dpi = 300, bbox_inches='tight')
plt.show()

#%% def duibi_huitu(guokong,shouchi,cheng,shouchi_label,my_ylabel,my_title,my_save_name):

#duibi_huitu('PM2.5_x','PM2.5_y',1000,'TSI DustTrak 8534','PM$_{2.5}$ Concentration ($\mu g/m^3$)','PM$_{2.5}$ Comparison','')
duibi_huitu('PM10_x','PM10_y',1000,'TSI DustTrak 8534','PM$_{10}$ Concentration ($\mu g/m^3$)','PM$_{10}$ Comparison','')
#duibi_huitu('temperature','temp',1,'Onset HOBO U12-013','Temperature ($°C$)','Temperature Comparison','duibi_hobo_0510_win')
#duibi_huitu('rh_x','rh_y',1,'Onset HOBO U12-013','Relative Humidity (%)','Relative Humidity Comparison','duibi_hobo_rh_0510_win')

#duibi_huitu('bc','conc_x',1,'microAeth AE51','Black Carbon ($ng/m^3$)','Black Carbon Comparison','duibi_hobo_bc_0511_win')
#%%
#z1 = np.polyfit(duibi_biaozhun_ap['PM2.5_x'], duibi_biaozhun_ap['PM2.5_y']*1000, 1) 
#p1 = np.poly1d(z1)
#print(p1) # 在屏幕上打印拟合多项式
#
#
#plt.figure(figsize=(10, 6))
#g = sns.jointplot(no_8534_24_guokong_10[0],no_8534_24_guokong_10['PM10']/1000,size=8,kind = 'reg')
#g.set_axis_labels("8534 PM$_{10}$","Monitoring station");
#plt.text(0.125,0.15,'y= 0.5204 x + 0.041')
#plt.show()

g = sns.jointplot(duibi_biaozhun_ap['temperature'],duibi_biaozhun_ap['temp'],height=8,kind = 'reg')
g = g.annotate(stats.pearsonr)
g.set_axis_labels("Monitoring station ($°C$)","Onset HOBO U12-013 ($°C$)");
#plt.title('Temperature Comparison')

g = sns.jointplot(duibi_biaozhun_ap['rh_x'],duibi_biaozhun_ap['rh_y'],height=8,kind = 'reg')
g = g.annotate(stats.pearsonr)
g.set_axis_labels("Monitoring station (%)","Onset HOBO U12-013 (%)");
#plt.title('Relative Humidity Comparison')

g = sns.jointplot(duibi_biaozhun_ap['PM2.5_x'],duibi_biaozhun_ap['PM2.5_y']*1000,height=8,kind = 'reg')
g = g.annotate(stats.pearsonr)
g.set_axis_labels("Monitoring station ($\mu g/m^3$)","TSI DustTrak 8534 ($\mu g/m^3$)");

g = sns.jointplot(duibi_biaozhun_ap['PM10_x'],duibi_biaozhun_ap['PM10_y']*1000,height=8,kind = 'reg')
g = g.annotate(stats.pearsonr)
g.set_axis_labels("Monitoring station ($\mu g/m^3$)","TSI DustTrak 8534 ($\mu g/m^3$)");

g = sns.jointplot(duibi_biaozhun_ap['bc'][1:],duibi_biaozhun_ap['conc_x'][0:len(duibi_biaozhun_ap)-1],height=8,kind = 'reg')
#plt.text(5000,6800,'y = 0.598x + 1152.6')
g = g.annotate(stats.pearsonr)
g.set_axis_labels("Monitoring station ($ng/m^3$)","microAeth AE51 ($ng/m^3$)");

g = sns.jointplot(duibi_biaozhun_ap['bc'][1:],duibi_biaozhun_ap['conc_y'][0:len(duibi_biaozhun_ap)-1],height=8,kind = 'reg')
g = g.annotate(stats.pearsonr)
g.set_axis_labels("Monitoring station ($ng/m^3$)","microAeth AE51 2 ($ng/m^3$)");



#%%
#laojianhe = record_and_ap[record_and_ap['place'] == 'laojianhe']
#laojianhe['place_lei'] = 0
#shuniu = record_and_ap[record_and_ap['place'] == 'shuniu']
#shuniu['place_lei'] = 1
#idd_1 = laojianhe.flight.unique() 
#idd_2 = shuniu.flight.unique() 

#fenlei_place = [laojianhe['bc_ir'],shuniu['bc_ir']]
#fenlei_place_2 =laojianhe.append(shuniu)
fenlei_place_3 = pd.read_excel(data_file_path+'fenlei_place.xlsx')
laojianhe = fenlei_place_3[fenlei_place_3['place'] == 'laojianhe']
shuniu = fenlei_place_3[fenlei_place_3['place'] == 'shuniu']
xiajinxian = pd.read_excel(data_file_path+'xiajinxian.xlsx')

#laojianhe.to_excel(data_file_path+'laojianhe.xlsx')
#shuniu.to_excel(data_file_path+'shuniu.xlsx')
#xiajinxian.to_excel(data_file_path+'xiajinxian.xlsx')

#fenlei_place_2.to_excel(data_file_path+'fenlei_place.xlsx')
c = "red"
#ax = plt.subplot(1,1,1)  
#ax.boxplot(fenlei_place,bootstrap=5000,whis=1.5,meanline=True,flierprops=dict(color=c, markeredgecolor=c))  
#plt.figure(figsize=(10, 6))
#g = sns.boxplot(y= fenlei_place_3['PM_0.3'],x=fenlei_place_3['place'],hue=fenlei_place_3['Campaign'],orient="v",saturation=0.75, width=0.4)
#g.set(ylabel = 'PM$_{0.3}$ Concentration',xticklabels = ['Laojianhe','Shuniu'],xlabel = 'Place')
#plt.axhline(y=xiajinxian['PM_0.3'].mean())


#def hua_zhuxingtu(mingcheng,biaozhu):
hua_zhuxingtu('PM_0.3','PM$_{0.3}$ Concentration ($\mu g/m^3$)')
hua_zhuxingtu('PM_0.5','PM$_{0.5}$ Concentration ($\mu g/m^3$)')
hua_zhuxingtu('PM_1','PM$_{1}$ Concentration ($\mu g/m^3$)')
hua_zhuxingtu('PM_3','PM$_{3}$ Concentration ($\mu g/m^3$)')
hua_zhuxingtu('PM_5','PM$_{5}$ Concentration ($\mu g/m^3$)')
hua_zhuxingtu('PM_10','PM$_{10}$ Concentration ($\mu g/m^3$)')

hua_zhuxingtu('bc_ir','Blank Carbon Concentration ($ng/m^3$)')

#hua_zhuxingtu('wind_level_local','Wind Concentration (level)')

#plt.figure(figsize=(10, 6))
#g = sns.boxplot(y= fenlei_place_3['wind_level_local'],x=fenlei_place_3['place'],hue=fenlei_place_3['Campaign'],orient="v",saturation=0.75, width=0.4)
#g.set(ylabel = 'Wind Concentration (level)',xticklabels = ['Laojianhe','Shuniu'],xlabel = 'Place')


plt.figure(figsize=(10, 6))
g = sns.lineplot(x="Campaign", y="wind_level_local",
             hue="place", 
             data=fenlei_place_3)
g.set(ylabel = 'Wind Concentration (level)')



#%%

fenlei_place_3 = pd.read_excel(data_file_path+'fenlei_place.xlsx')
fenlei_place_4 = fenlei_place_3[fenlei_place_3['mode']=='lashen']

laojianhe = fenlei_place_4[fenlei_place_4['place'] == 'laojianhe']
shuniu = fenlei_place_4[fenlei_place_4['place'] == 'shuniu']


plt.figure(figsize=(10, 6))
g = sns.lineplot(x="Distance", y="PM_0.3",hue="place",data=fenlei_place_4)
g.set(ylabel ='PM$_{0.3}$ Concentration ($\mu g/m^3$)')


plt.figure(figsize=(10, 6))
g = sns.lineplot(x="Distance", y="PM_0.5",hue="place",data=fenlei_place_4)
g.set(ylabel ='PM$_{0.5}$ Concentration ($\mu g/m^3$)')

plt.figure(figsize=(10, 6))
g = sns.lineplot(x="Distance", y="PM_1",hue="place",data=fenlei_place_4)
g.set(ylabel ='PM$_{1}$ Concentration ($\mu g/m^3$)')

plt.figure(figsize=(10, 6))
g = sns.lineplot(x="Distance", y="bc_ir",hue="place",data=fenlei_place_4)
g.set(ylabel ='BC Concentration ($\mu g/m^3$)')
#%%

plt.figure(figsize=(10, 6))
g = sns.lineplot(x="Distance", y="PM_0.3",hue="Campaign",data=fenlei_place_4,style="place")
g.set(ylabel ='PM$_{0.3}$ Concentration ($\mu g/m^3$)')

#plt.figure(figsize=(10, 6))
#g = sns.lineplot(x="Distance", y="PM_0.3",hue="Campaign",data=laojianhe,style="Campaign")
#g.set(ylabel ='Laojianhe PM$_{0.3}$ Concentration ($\mu g/m^3$)')

#plt.figure(figsize=(10, 6))
#g = sns.lineplot(x="Distance", y="PM_0.3",hue="Campaign",data=shuniu)
#g.set(ylabel ='Shuniu PM$_{0.3}$ Concentration ($\mu g/m^3$)')


#%%
#sns.jointplot(laojianhe['PM_0.3'],shuniu['PM_0.3']*1000,height=8,kind = 'reg')
plt.figure(figsize=(20, 12))
sns.set(font_scale=2,font = 'Times New Roman')
g = sns.boxplot(y= fenlei_place_3['PM_0.3']*1000,x=fenlei_place_3['Distance'],hue=fenlei_place_3['Campaign'],orient="v",saturation=0.75, width=0.4)
g.set(ylabel = 'PM$_{0.3}$ Concentration')



#%%


place = pd.DataFrame(all_exp_record['Place'])
place = place.drop_duplicates()
place = di(place)

mean_id = pd.DataFrame()
mean_place = pd.DataFrame()
for i in range(0,len(place)):
#i=0
    print(place.iloc[i,0])
    huatu_place_one = record_and_ap[record_and_ap['Place'] == place.iloc[i,0]]
    huatu_place_one_id = di(huatu_place_one['flight'].drop_duplicates())
    mean_place = mean_place.append(pd.DataFrame((huatu_place_one.iloc[:,12:]).mean()).T)
    for k in range(0,len(huatu_place_one_id)):
        huatu_place_one_id_one = di(huatu_place_one[huatu_place_one['flight'] == huatu_place_one_id[k]])
        mean_id = mean_id.append(pd.DataFrame((huatu_place_one_id_one.iloc[:,12:]).mean()).T)
wuranwu_name = record_and_ap.columns.values.tolist()
wuranwu_name = wuranwu_name[21:]

wuranwu_ylabel = ['Number concentration (#$/m^3$)','Number concentration (#$/m^3$)','Number concentration (#$/m^3$)',
                  'Number concentration (#$/m^3$)','Number concentration (#$/m^3$)','Number concentration (#$/m^3$)',                 
                  'Concenreation ($\mu g/m^3$)','Concenreation ($\mu g/m^3$)','Concenreation ($\mu g/m^3$)',
                  'Concenreation ($\mu g/m^3$)','Concenreation ($\mu g/m^3$)',
                  'Concenreation ($ng /m^3$)', 
                  'Temperature (°C)', 'RH (%)', 'Dew (°C)'
                  ]


#%%

#yanse = ['red','green','orange','blue','brown','purple']
#
#for j in range(0,len(wuranwu_name)):
##j = 0
#    for i in range(0,len(place)):
#    #i=0    
#        huatu_place_one = record_and_ap[record_and_ap['Place'] == place.iloc[i,0]]
#        huatu_place_one_id = di(huatu_place_one['flight'].drop_duplicates())
#        #k = 0
#        fig = plt.figure(figsize=(8, 12))
#        ax1 = fig.add_subplot(111)#需要通过加子图的方式实现
#        plt.rcParams['font.size'] = 20
#        plt.rcParams['font.family'] = 'Times New Roman'
#        for k in range(0,len(huatu_place_one_id)):
#            huatu_place_one_id_one = di(huatu_place_one[huatu_place_one['flight'] == huatu_place_one_id[k]])
#            if (huatu_place_one_id_one.loc[0,'mode'] == 'chuizhi')&(str(huatu_place_one_id_one.loc[0,wuranwu_name[j]]) != 'nan'):
#                hua_chuizhi(huatu_place_one_id_one,wuranwu_name[j],huatu_place_one_id_one.loc[0,'datetime_begin'],k)
#        plt.xlabel(wuranwu_ylabel[j])
#        plt.title(place.iloc[i,0]+" "+wuranwu_name[j])
#        box = ax1.get_position()
#        ax1.set_position([box.x0, box.y0, box.width* 0.5 , box.height])
#        ax1.legend(loc='center left', bbox_to_anchor=(1, 0.15),ncol=1)
#        # plt.savefig(results_path+'chuizhi/'+place.iloc[i,0]+"_"+wuranwu_name[j]+'_chiuzhi'+'.jpg', dpi=300,bbox_inches='tight')
#        plt.show()
#        
#        fig = plt.figure(figsize=(12, 8))
#        ax1 = fig.add_subplot(111)#需要通过加子图的方式实现
#        plt.rcParams['font.size'] = 20
#        plt.rcParams['font.family'] = 'Times New Roman'
#        for k in range(0,len(huatu_place_one_id)):
#            huatu_place_one_id_one = di(huatu_place_one[huatu_place_one['flight'] == huatu_place_one_id[k]])
#            if (huatu_place_one_id_one.loc[0,'mode'] == 'shuiping')&(str(huatu_place_one_id_one.loc[0,wuranwu_name[j]]) != 'nan'):
#                hua_lashen(huatu_place_one_id_one,wuranwu_name[j],huatu_place_one_id_one.loc[0,'datetime_begin'],k)
#        plt.ylabel(wuranwu_ylabel[j])
#        plt.title(place.iloc[i,0]+" "+wuranwu_name[j])
#        box = ax1.get_position()
#        ax1.set_position([box.x0, box.y0, box.width* 0.5 , box.height])
#        ax1.legend(loc='center left', bbox_to_anchor=(1, 0.15),ncol=1)
#        # plt.savefig(results_path+'lashen/'+place.iloc[i,0]+"_"+wuranwu_name[j]+'_shuiping'+'.jpg', dpi=300,bbox_inches='tight')
#        plt.show()

#%% 画风速风向
plt.figure(figsize=(12, 6))
fengsu = pd.read_excel(qixiang_file_path+'54715099999_2018_and_2019.xlsx')
fengsu['datetime(gmt)'] = pd.to_datetime(fengsu['datetime(gmt)'])
fengsu['datetime_china'] = fengsu['datetime(gmt)']+dt.timedelta(hours=8)
fengsu = fengsu[(fengsu['datetime_china']>= pd.datetime(2018,11,1)) & (fengsu['datetime_china']<= pd.datetime(2019,5,1))]
plt.plot_date(fengsu['datetime_china'],fengsu['wind_direction'],'o-')


#%%


flight_id = di(record_and_ap['flight'].drop_duplicates())

i=0

record_and_ap[record_and_ap['flight'] == flight_id[i]]



















    


