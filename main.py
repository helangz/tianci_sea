import pandas as pd
from gensim.models import Word2Vec
import time
import numpy as np
import os
import warnings
import datetime
import lightgbm as lgb
from sklearn.model_selection import StratifiedKFold
from sklearn import metrics
import warnings
pd.set_option('display.max_columns', 100)
warnings.filterwarnings('ignore')

train_root='tcdata/hy_round2_train_20200225'
train_path=[train_root+'/'+file for file in os.listdir(train_root)]
files_list=[pd.read_csv(path,encoding='utf-8') for path in train_path]
train=pd.concat(files_list)
#test
test_root='tcdata/hy_round2_testB_20200312'
test_path=[test_root+'/'+file for file in os.listdir(test_root)]
files_list=[pd.read_csv(path,encoding='utf-8') for path in test_path]
test=pd.concat(files_list)

data=train.append(test)
data=data.rename(columns={'渔船ID':'ship','lat':'x','lon':'y','速度':'v','方向':'d'})
def get_deta(datai,num=1):
    datai['time']=pd.to_datetime(datai['time'], format='%m%d %H:%M:%S')
    datai=datai.sort_values(by='time').reset_index(drop=True)
    datai['deta_v']=datai['v']-datai['v'].shift(num)    
    datai['deta_x']=datai['x']-datai['x'].shift(num)
    datai['deta_y']=datai['y']-datai['y'].shift(num)
    datai['deta_d']=datai['d']-datai['d'].shift(num)       
    datai['deta_t']=(datai['time']-datai['time'].shift(num)).dt.total_seconds()
    return datai
group_list=[datai for name,datai in data.groupby('ship')]
group_list_map=list(map(get_deta,group_list))
data=pd.concat(group_list_map)
for feat in ['x','y','v','d']:
    data[f'deta_{feat}_t']=data[f'deta_{feat}'].apply(lambda x:abs(x))*100/data['deta_t']
data['deta_xy_t']=data['deta_x'].apply(lambda x:abs(x))+data['deta_y'].apply(lambda x:abs(x))
#计算时间相关的特征
data['time'] = pd.to_datetime(data['time'], format='%m%d %H:%M:%S')
data['date'] = data['time'].dt.date
data['hour'] =data['time'].dt.hour
data['weekday'] =data['time'].dt.weekday
data.to_csv('data/deta_data.csv',encoding='utf-8',index=None)


# ### 基础统计特征

data=pd.read_csv('data/deta_data.csv',encoding='utf-8')
data_label=data.drop_duplicates('ship')[['x','y','v','d','type','ship']].reset_index(drop=True)
def group_feature(df, key, target, aggs):   
    agg_dict = {}
    for ag in aggs:
        agg_dict[f'{target}_{ag}'] = ag
    t = df.groupby(key)[target].agg(agg_dict).reset_index()
    return t
#基础特征
t = group_feature(data, 'ship','x',['max','min','mean','std','skew'])
data_label= pd.merge(data_label, t, on='ship', how='left')
t = group_feature(data, 'ship','y',['max','min','mean','std','skew'])
data_label= pd.merge(data_label, t, on='ship', how='left')
t = group_feature(data, 'ship','v',['max','min','mean','std','skew'])
data_label= pd.merge(data_label, t, on='ship', how='left')
t = group_feature(data, 'ship','d',['max','mean','std','skew'])
data_label= pd.merge(data_label, t, on='ship', how='left')
#求极值
t = group_feature(data, 'ship','x',['count'])
data_label = pd.merge(data_label, t, on='ship', how='left')
data_label['v_max_v_min']=data_label['v_max'] -data_label['v_min']
data_label['x_max_x_min'] = data_label['x_max'] -data_label['x_min']
data_label['y_max_y_min'] = data_label['y_max'] - data_label['y_min']
data_label['y_max_x_min'] = data_label['y_max'] - data_label['x_min']
data_label['x_max_y_min'] = data_label['x_max'] - data_label['y_min']
data_label['slope'] =data_label['y_max_y_min'] / np.where(data_label['x_max_x_min']==0, 0.001, data_label['x_max_x_min'])
data_label['area'] = data_label['x_max_x_min'] *data_label['y_max_y_min']

#deta的基础统计特征
for feat in ['x','y','v','d']:
    t = group_feature(data, 'ship',f'deta_{feat}_t',['max','mean','std'])
    data_label= pd.merge(data_label,t, on='ship', how='left')
#deta的高阶特征
data['deta_x|y']=data['deta_x'].apply(lambda x:abs(x))/data['deta_y'].apply(lambda x:abs(x))
data['deta_x_y']=data['deta_x'].apply(lambda x:abs(x))*data['deta_y'].apply(lambda x:abs(x))
data['deta_xy']=data['deta_x'].apply(lambda x:abs(x))+data['deta_y'].apply(lambda x:abs(x))


nuns={np.NaN:0,np.inf:0}
data=data.replace(nuns)
for feat in  ['deta_x|y','deta_x_y','deta_xy']:
    t = group_feature(data, 'ship',feat,['max','mean'])
    data_label= pd.merge(data_label,t, on='ship', how='left')
#大于零的概率
t=data.groupby('ship')['deta_x|y'].agg({'deta_x|y_prob0': lambda x: len(x[x>0])/len(x)}).reset_index()
data_label= pd.merge(data_label,t, on='ship', how='left')

    
#极差与均值之比
data_label['x_max_x_min|mean']=data_label['x_max_x_min']/data_label['x_mean']
data_label['y_max_y_min|mean']=data_label['y_max_y_min']/data_label['y_mean']
data_label['v_max_v_min|mean']=data_label['v_max_v_min']/data_label['v_mean']
#均值之比
data_label['y_mean|x_mean']=data_label['y_mean']/data_label['x_mean']
data_label['x_max_x_min|y_max_y_min']=data_label['x_max_x_min']/data_label['y_max_y_min']
data_label['x_max_x_min|v_mean']=data_label['x_max_x_min']/data_label['v_mean']
data_label['y_max_y_min|v_mean']=data_label['y_max_y_min']/data_label['v_mean']

cov_list=[datai['x'].cov(datai['y']) for name,datai in data.groupby('ship')]
data_label['cov_list']=cov_list
#与时间相关的特征
mode_hour = data.groupby('ship')['hour'].agg(lambda x:x.value_counts().index[0]).to_dict()
data_label['mode_hour'] =data_label['ship'].map(mode_hour)
t = group_feature(data, 'ship','hour',['max','min'])
data_label = pd.merge(data_label, t, on='ship', how='left')
hour_nunique = data.groupby('ship')['hour'].nunique().to_dict()
date_nunique = data.groupby('ship')['date'].nunique().to_dict()
data_label['hour_nunique'] = data_label['ship'].map(hour_nunique)
data_label['date_nunique'] = data_label['ship'].map(date_nunique)
#分位数
for feat in ['x','y','v']:
    for num in [0.1,0.2,0.8,0.9]: 
        t = data.groupby('ship')[feat].agg({f'{feat}_quanti_{num}':lambda x:x.quantile(num)}).reset_index()
        data_label = pd.merge(data_label, t, on='ship', how='left')
#分位数均值之比    
    for num in [0.1,0.2,0.8,0.9]:
        data_label[f'{feat}_{num}|mean']=data_label[f'{feat}_quanti_{num}']/data_label[f'{feat}_mean']


# ### 速度和方向差的分段特征

# In[39]:


#按方向的分段差特征
data['deta_d']=data['deta_d'].abs()
data['deta_x_t']=data['deta_x_t'].abs()
data['deta_y']=data['deta_y_t'].abs()
data['deta_v_t']=data['deta_v_t'].abs()
groupd0=pd.concat([datai[datai['deta_d']==0] for name,datai in data.groupby('ship')])
groupd1=pd.concat([datai[(datai['deta_d']>0) & (datai['deta_d']<10)] for name,datai in data.groupby('ship')])
groupd2=pd.concat([datai[datai['deta_d']>10] for name,datai in data.groupby('ship')])
for i,group in enumerate([groupd0,groupd1,groupd2]):
    #v均值,detax,y均值，deta_v均值
    for feat in ['deta_x_t','deta_y_t','deta_v_t','v']:
        t=group.groupby('ship')[feat].agg({f'{feat}_detad{i}_mean':'mean'}).reset_index()
        data_label= pd.merge(data_label, t, on='ship', how='left')
#按速度的分段特征
datav0=pd.concat([datai[datai['v']<datai['v'].quantile(0.3)] for name,datai in data.groupby('ship')])
datav1=pd.concat([datai[datai['v']>datai['v'].quantile(0.7)] for name,datai in data.groupby('ship')])
for i,dataz in enumerate([datav0,datav1]):
    t = dataz.groupby('ship')['v'].agg({f'v{i}_mean':'mean'}).reset_index()                         
    data_label= pd.merge(data_label, t, on='ship', how='left')
    t = dataz.groupby('ship')['x'].agg({f'x{i}_max_mean':lambda x:x.max()-x.min()}).reset_index()                         
    data_label= pd.merge(data_label, t, on='ship', how='left')
    t = dataz.groupby('ship')['y'].agg({f'y{i}_max_mean':lambda x:x.max()-x.min()}).reset_index()                         
    data_label= pd.merge(data_label, t, on='ship', how='left')   
    t = dataz.groupby('ship')['deta_d_t'].agg({f'deta_d{i}_mean':lambda x:x.abs().mean()}).reset_index()                         
    data_label= pd.merge(data_label, t, on='ship', how='left') 


# ### 港口和众数的特征

#data=pd.read_csv('data/deta_data.csv',encoding='utf-8')
#data_label=pd.read_csv('data/data_label_basic0.csv',encoding='utf-8')
for f in ['x','y']:         
    #提取每艘船停留最多x,y位置
    t=data.groupby('ship')[f].agg({'stay_pos_{}'.format(f):lambda x : x.value_counts().index[0]}).reset_index()
    data_label= pd.merge(data_label, t, on='ship', how='left')
    #行驶最远与停泊点的距离
    data_label['stay_pos_dis_{}'.format(f)] = data_label['{}_max'.format(f)] - data_label['stay_pos_{}'.format(f)]
    if f == 'y':
        data_label['stay_pos_dis_line'] = (data_label['stay_pos_dis_x']**2 + data_label['stay_pos_dis_y']**2)**0.5
        print('stay_pos_dis_line')    
   #停泊点和港口距离 
    if f == 'x':
        data_label['stay_base_{}'.format(f)] = data_label['stay_pos_{}'.format(f)] -data['x'].mode()[0]
    if f == 'y':
        data_label['stay_base_{}'.format(f)] = data_label['stay_pos_{}'.format(f)] -data['y'].mode()[0]
        data_label['stay_base_area'] = data_label['stay_base_x'].abs()*data_label['stay_base_y'].abs()
        data_label['stay_base_distance'] = (data_label['stay_base_x']**2 + data_label['stay_base_y']**2)**0.5
        print('stay_base_distance')        
    #停留情况的比例
    print('stay_ratiao_y')
    if f == 'y':
        t=data.groupby(['ship'])['y'].apply(lambda x : x.value_counts().iloc[0])/ data.groupby(['ship'])['y'].count()
        t.name= 'stay_ratiao_y'
        data_label= pd.merge(data_label, t, on='ship', how='left')
data_label.to_csv('data/data_label_basic.csv',encoding='utf-8',index=None)
data=pd.read_csv('data/deta_data.csv',encoding='utf-8')
data_label=data.drop_duplicates('ship')[['type','ship']].reset_index(drop=True)

#排列熵特征
import itertools
from math import factorial
def _embed(x, order=3, delay=1):
    N = len(x)
    Y = np.empty((order, N - (order - 1) * delay))
    for i in range(order):
        Y[i] = x[i * delay:i * delay + Y.shape[1]]
    return Y.T
def permutation_entropy(time_series, order=3, delay=1, normalize=False):
    x = np.array(time_series)
    hashmult = np.power(order, np.arange(order))
    sorted_idx = _embed(x, order=order, delay=delay).argsort(kind='quicksort')
    hashval = (np.multiply(sorted_idx, hashmult)).sum(1)
    _, c = np.unique(hashval, return_counts=True)
    p = np.true_divide(c, c.sum())
    pe = -np.multiply(p, np.log2(p)).sum()
    if normalize:
        pe /= np.log2(factorial(order))
    return pe
data['x_trans']=data['x'].apply(lambda x:int(x/0.01))
data['y_trans']=data['y'].apply(lambda x:int(x/0.01))
data['d_trans']=data['d'].apply(lambda x:int(x/3))
data['v_trans']=data['v']
for name in ['x','y','v','d']:
    t = data.groupby('ship')[name+'_trans'].agg({f'{name}_pentropy':permutation_entropy}).reset_index()
    data_label= pd.merge(data_label, t, on='ship', how='left')
data_label.to_csv('data/pentropy.csv',encoding='utf-8',index=None)

#embedding特征
def emb(data, f2):
    emb_size =5
    tmp = data.groupby('ship', as_index=False)[f2].agg({'{}_list'.format( f2): list})
    sentences = tmp['{}_list'.format(f2)].values.tolist()
    del tmp['{}_list'.format( f2)]
    for i in range(len(sentences)):
        sentences[i] = [str(x) for x in sentences[i]]
    model = Word2Vec(sentences, size=emb_size, window=5, min_count=3, sg=0, hs=1, seed=2019)
    emb_matrix = []
    for seq in sentences:
        vec = []
        for w in seq:
            if w in model:
                vec.append(model[w])
        if len(vec) > 0:
            emb_matrix.append(np.mean(vec, axis=0))
        else:
            emb_matrix.append([0] * emb_size)
    emb_matrix = np.array(emb_matrix)
    for i in range(emb_size):
        tmp['{}_emb_{}'.format(f2, i)] = emb_matrix[:, i]
    del model, emb_matrix, sentences
    return tmp
data=pd.read_csv('data/deta_data.csv',encoding='utf-8')
data_label=data.drop_duplicates('ship').reset_index(drop=True)[['type','ship']]
data['x']=data['x'].apply(lambda x:int(x/0.01))
data['y']=data['y'].apply(lambda x:int(x/0.01))
data['d']=data['d'].apply(lambda x:int(x/3))
data['v']=data['v']
for feat in ['x','y','v','d']:
    t=emb(data,feat)
    data_label=pd.merge(data_label,t, on='ship', how='left')
data_label.to_csv('data/embding.csv',encoding='utf-8',index=None)


#降维特征
from sklearn.manifold import TSNE
data=pd.read_csv('./data/deta_data.csv',encoding='utf-8')
#标准化处理xyv
from sklearn.preprocessing import scale
for feat in ['x','y','v','d']:
    data[f'{feat}_scale']=scale(data[feat],with_mean=True,with_std=True,copy=True)
data_label=data.drop_duplicates('ship').reset_index(drop=True)[['type','ship']]
def get_one_hot(dataj,feat):
    one_hots=np.zeros((len(dataj['ship'].unique()),len(dataj[feat].unique())))
    dic=dict(zip(dataj[feat].unique(),range(len(dataj[feat].unique()))))
    group_list=[datai for name,datai in dataj.groupby('ship')]
    for i,datai in enumerate(group_list):
        tmp_dic=datai[feat].value_counts(1).to_dict()
        for key in tmp_dic:
            label=dic[key]
            one_hots[i,label]=tmp_dic[key]
    return one_hots
def get_label(data_label,feat,num,n,state):
    datai=pd.DataFrame()
    datai['ship']=data['ship']
    datai[feat]=data[feat].apply(lambda x:int(x/num+0.5*num))
    one_hots=get_one_hot(datai,feat)
    tsne=TSNE(n_components=n,random_state=state)
    one_hots=tsne.fit_transform(one_hots)
    for i in range(len(one_hots[0])):
        data_label[f'label_{feat}_{i}']=one_hots[:,i]
    return data_label
num=0.03
dataxy=pd.DataFrame()
dataxy['ship']=data['ship']
dataxy['x']=data['x_scale']
dataxy['y']=data['y_scale']
dataxy['x']=dataxy['x'].apply(lambda x:int(x/num+0.5*num))
dataxy['y']=dataxy['y'].apply(lambda x:int(x/num))
dataxy['xy']=dataxy['x'].map(str)+'_'+dataxy['y'].map(str)
print(len(dataxy['xy'].unique()))
tsne=TSNE(n_components=3,random_state=5)
one_hots=get_one_hot(dataxy,'xy')
one_hots=tsne.fit_transform(one_hots)
for i in range(len(one_hots[0])):
    data_label[f'label_xy_{i}']=one_hots[:,i]
for feat in ['x','y']:
    print('****************start_{feat}_______*********')
    data_label=get_label(data_label,feat,0.005,n=3,state=5)
data_label=get_label(data_label,'v',0.01,n=1,state=5)
print('end_v')
data_label=get_label(data_label,'d',2,n=1,state=5)
print('end_d')
t=data.groupby('ship')['x'].agg({'x_median':'median'}).reset_index()
data_label= pd.merge(data_label, t, on='ship', how='left')
t=data.groupby('ship')['y'].agg({'y_median':'median'}).reset_index()
data_label= pd.merge(data_label, t, on='ship', how='left')
interval=30
allx=data_label
allx['x_median'] = pd.qcut(allx['x_median'].values,interval).codes
allx['y_median'] = pd.qcut(allx['y_median'].values,interval).codes
allx=data_label
grid_col = 'x_median'+'y_median'+'_grid'
allx[grid_col] = allx[ 'x_median'].map(str)+ allx['y_median'].map(str)
less_5 = allx[grid_col].value_counts()
less_5 = less_5[less_5<=5].index
allx[grid_col] = allx[grid_col].replace(less_5,[999999]*len(less_5))
less_10 = allx[grid_col].value_counts()
less_10 = less_10[(less_10<=10) &(less_10>5)].index
allx[grid_col] = allx[grid_col].replace(less_10,[99999]*len(less_10))
dimension = TSNE(n_components =3,n_iter = 300,random_state=15)
a = dimension.fit_transform(pd.get_dummies(allx[grid_col]).values)
new_name = [grid_col+'_tsne'+str(i) for i in range(3)]
a = pd.DataFrame(a,columns=new_name)
for col in a.columns:
    data_label[col]=a[col]
data_label=data_label.drop(['x_median','y_median','x_mediany_median_grid'],axis=1)
data_label.to_csv('./data/tsne.csv',encoding='utf-8',index=None)

#特征读取
data_label=pd.read_csv('data/data_label_basic.csv',encoding='utf-8')
emb=pd.read_csv('data/embding.csv',encoding='utf-8').iloc[:,2:]
tsne=pd.read_csv('./data/tsne.csv',encoding='utf-8').iloc[:,2:]
for col in emb:
    data_label[col]=emb[col]
pentropy=pd.read_csv('data/pentropy.csv',encoding='utf-8').iloc[:,2:]
for col in pentropy:
    data_label[col]=pentropy[col]    
for col in tsne:
    data_label[col]=tsne[col]

train_label=data_label[~data_label['type'].isnull()]
test_label=data_label[data_label['type'].isnull()]
type_map = dict(zip(train_label['type'].unique(), np.arange(3)))
type_map_rev = {v:k for k,v in type_map.items()}
train_label['type'] = train_label['type'].map(type_map)
#type_map_rev={0: '拖网', 1: '围网', 2: '刺网'}
features = [x for x in train_label.columns if x not in ['ship','type','time','diff_time']]
target = 'type'

#模型
params = {
    'num_boost_round': 100,
    'boosting_type': 'gbdt',
    'objective': 'multiclass',
    'num_class': 3,
    'early_stopping_rounds': 50,
}
pred_list=[]
for state in range(40,45):
    fold = StratifiedKFold(n_splits=5, shuffle=True, random_state=state )

    X = train_label[features].copy()
    y = train_label[target]
    models = []
    pred = np.zeros((len(test_label),3))
    oof = np.zeros((len(X), 3))
    for index, (train_idx, val_idx) in enumerate(fold.split(X, y)):


        train_set = lgb.Dataset(X.iloc[train_idx], y.iloc[train_idx])
        val_set = lgb.Dataset(X.iloc[val_idx], y.iloc[val_idx])

        model = lgb.train(params, train_set, valid_sets=[train_set, val_set], verbose_eval=100)
        models.append(model)
        val_pred = model.predict(X.iloc[val_idx])
        oof[val_idx] = val_pred
        val_y = y.iloc[val_idx]
        val_pred = np.argmax(val_pred, axis=1)
        print(index, 'val f1', metrics.f1_score(val_y, val_pred, average='macro'))
        # 0.8695539641133697
        # 0.8866211724839532

        test_pred = model.predict(test_label[features])
        pred += test_pred/5

    pred_list.append(pred)

pred=sum(pred_list)/len(pred_list)
pred = np.argmax(pred, axis=1)
sub = test_label[['ship']]
sub['pred'] = pred
print(sub['pred'].value_counts())
sub['pred'] = sub['pred'].map(type_map_rev)
sub.to_csv('result.csv', header=None, index=False)