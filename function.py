import pickle
import csv
import pickle
import pandas as pd
from sklearn.model_selection import RandomizedSearchCV
import numpy as np
from sklearn.linear_model import LogisticRegression as LR
from sklearn.model_selection import KFold,ParameterSampler
import xgboost
from sklearn.preprocessing import OneHotEncoder
from collections import Iterable
from random import shuffle
pd.set_option("display.max_column",10)
q=[0,0.01,0.05,0.1,0.2,0.3,0.4,0.5,0.6,0.7,0.8,0.9,0.95,0.99,1]

def read_df(df_list):
    ans=[]
    for name in df_list:
        ans.append(pd.read_csv(name))
    return pd.concat(ans,ignore_index=True)

def score(pre,label):
    """
    计算得分（在全部label数上取平均，在已预测的样本上取平均），召回率（在全部label上统计，在已预测的样本上统计）
    命中时的名次，命中时的得分（如果有得分的话）
    :param pre: dict of list
    :param label: list of tuple (user_id , item_id)
    :return:
    """
    recall=0.0
    ans=0.0
    num=0
    num1=0
    shoot={}
    shoot1 = {}#命中的得分
    for user_id,item_id in label.items():
        num+=1
        if(pre.get(user_id) is None):
            continue
        _pre=pre[user_id]
        if(len(_pre)!=0):
            num1+=1
        for i,item in zip(range(30),_pre):
            df=None
            if(isinstance(item,tuple)):
                e=item[0]
                df=item[1]
            else:
                e=item
            if(e==item_id):
                ans+=1/(i+1)
                recall+=1
                shoot[user_id]=i
                shoot1[user_id]=df
                break
    return ans/num,ans/num1,recall/num,recall/num1,shoot,shoot1

def save(data,file_name):
    """
    保存数据
    :param data:
    :param file_name:
    :return:
    """
    with open(file_name,"wb") as f:
        pickle.dump(data,f)

def load(file_name):
    """
    从pkl里读取数据
    :param data:
    :param file_name:
    :return:
    """
    with open(file_name,"rb") as f:
        data=pickle.load(f)
    return data

def split_time(dt,silence=True):
    """
    分割时间的字段
    :param dt:
    :return:
    """
    ans={"year":[],"month":[],"day":[],"hour":[],"min":[],"sec":[]}
    for i,e in enumerate(dt.values):
        if(not silence and i%1e5==0):
            print(f"{i}  ({i/dt.shape[0]})")
        # ans["year"].append(e[:4])
        # ans["month"].append(e[5:7])
        ans["day"].append(e[8:10])
        ans["hour"].append(e[11:13])
        ans["min"].append(e[14:16])
        ans["sec"].append(e[-2:])
    return ans


def score(pre,label,n=30):
    """
    :param pre: dict of list
    :param label: list of tuple (user_id , item_id)
    :return:
    """
    recall=0.0
    ans=0.0
    num=0
    num1=0
    shoot={}
    shoot1 = {}#命中的得分
    for user_id,item_id in label.items():
        num+=1
        if(pre.get(user_id) is None):
            continue
        _pre=pre[user_id]
        if(len(_pre)!=0):
            num1+=1
        for i,item in zip(range(n),_pre):
            df=None
            if(isinstance(item,Iterable)):
                e=item[0]
                df=item[1]
            else:
                e=item
            if(e==item_id):
                ans+=1/(i+1)
                recall+=1
                shoot[user_id]=i
                shoot1[user_id]=df
                break
    if num1!=0:
         return ans / num, ans / num1, recall / num, recall / num1, shoot, shoot1
    else:
        return 0,0,0,0,0,0


def multi_col(df,col_list):
    """
    把df中的多列作为字符串拼接，给pandas.isin做功能扩展,df[multi_col(df , cols).isin(multi_col(df1 , cols))]
    :param df:
    :param col_list:
    :return:
    """
    for col in col_list:
        try:
            ans=ans+"_"+df[col].astype("str")
        except:
            ans=df[col].astype("str")
    if(len(ans.iloc[0].split("_"))==len(col_list)):
        return ans
    else:
        print("some thing error")
        return None


def read_map_from_file(file_name):
    """
    读取两列，形成dict，主要给计算score时提供true_label
    :param file_name:
    :return:
    """
    lst1 = {}
    with open(file_name, 'r') as f:
        file = csv.reader(f)
        print(next(file))
        for e in file:
            e[0] = int(e[0])
            e[1] = int(e[1])
            lst1[e[0]]=e[1]
    return lst1

def read_data(fn="yy"):
    X_train=load(f"offline/{fn}_train_sample.pkl")
    X_val=load(f"offline/{fn}_val_sample.pkl")
    lb=pd.read_csv(f"offline/{fn}_val_label.csv").rename(columns={"buyer_admin_id":"user_id"})
    label=read_map_from_df(lb[["user_id","item_id"]])

    y_train=X_train[["user_id","item_id","label"]]
    X_train=X_train.drop(columns=["user_id","item_id","label","score_from_0_0"])
    y_val=X_val[["user_id","item_id","label"]]
    X_val=X_val.drop(columns=["user_id","item_id","label","score_from_0_0"])
    return X_train,y_train,X_val,y_val,label

def read_map_from_df(df):
    """
    同read_map_from_file, 从dataframe中读取
    :param df:
    :return:
    """
    lst1 = {}
    for e in df.values:
        lst1[e[0]]=e[1]
    return lst1

def _day_ppl(data):
    assert isinstance(data,pd.DataFrame)
    if ("buyer_admin_id" in data.columns):
        data = data.rename(columns={"buyer_admin_id": "user_id"})
    df = data.drop_duplicates(["user_id", "item_id"]).groupby(["day", "item_id"]).size().rename("day_ppl").reset_index()
    df = df.sort_values("day_ppl", ascending=False)
    group = df.groupby("day")["item_id"]
    ans = {}
    for day, item in group:
        ans[day] = list(item.values[:100])
    return ans

def get_day_ppl(data,test,ans=None,n=30):
    assert isinstance(data, pd.DataFrame)
    assert isinstance(test, pd.DataFrame)
    day_ppl=_day_ppl(data)
    if(ans is None):
        ans={}
    if ("buyer_admin_id" in test.columns):
        test = test.rename(columns={"buyer_admin_id": "user_id"})
    if ("buyer_admin_id" in data.columns):
        data = data.rename(columns={"buyer_admin_id": "user_id"})
    test=test.sort_values("day",ascending=False)
    for uid,day in test.drop_duplicates("user_id")[["user_id","day"]].values:
        t=day_ppl.get(day)
        if(t is None):
            t=[]
        try:
            ans[uid]+= t[:n]
        except:
            ans[uid] = t[:n]
    return ans


def get_last_item(df,ans=None):
    """
    推荐每个用户的复购商品
    如果ans为None，则新建dict存结果，如果ans不为None，则给ans中的答案继续添加
    :param df:
    :param ans:
    :return:
    """
    assert isinstance(df,pd.DataFrame)
    df=df.sort_values("log_time",ascending=False)
    if("buyer_admin_id" in df.columns):
        df=df.rename(columns={"buyer_admin_id":"user_id"})
    df=df.drop_duplicates(["user_id","item_id"])
    group=df.groupby("user_id")
    if(ans is None):
        ans={}
    for uid,e in group:
        try:
            ans[uid]+= list(e["item_id"].values)
        except:
            ans[uid]= list(e["item_id"].values)
    return ans

def merge_ans(ans_list):
    """
    合并多个ans,如果key相同，则内容合并
    :param ans_list:
    :return:
    """
    ans0={}
    for ans in ans_list:
        for k in ans:
            try:
                ans0[k]+=ans[k]
            except:
                ans0[k] = ans[k]
    return ans0

def split_dataset(df,frac=1/6,random_state=0):
    """
    按照用户切分数据集
    :param df:
    :param frac:
    :return:
    """
    assert isinstance(df,pd.DataFrame)
    user_id = df["buyer_admin_id"].drop_duplicates()
    user_id_a=user_id.sample(frac=frac,random_state=random_state)
    user_id_b=set(user_id)-set(user_id_a)
    return df[df["buyer_admin_id"].isin(user_id_a)].copy(),df[df["buyer_admin_id"].isin(user_id_b)].copy()

def get_len_each_ans(ans):
    """
    返回ans中对每个用户的预测数目
    :param ans:
    :return:
    """
    return [len(ans[e]) for e in ans]

def check_ans_len(ans,n=30):
    """
    检查预测是否满足提交数目要求，默认30
    :param ans:
    :param n:
    :return:
    """
    for e in get_len_each_ans(ans):
        if(e!=30):
            return False
    return True

def to_submit(ans,file_name="to_submit.csv",has_rate=False,n_user=9844,baseline="提交/baseline.csv"):
    """
    保存预测结果为csv文件，可直接提交。如果ans中有得分，令has_rate为True即可
    内置了检查格式，长度，用户id等
    :param ans:
    :param file_name:
    :param has_rate:
    :return:
    """
    if(check_ans_len(ans)):
        if(has_rate):
            ans=get_ans_filed(ans,0)
        t = pd.DataFrame(ans).T
        # if(t.shape[0]!=n_user):
        #     print("用户数量不对")
        #     return None
        base=pd.read_csv(baseline,header=None,index_col=0)
        if(set(t.index)!=set(base.index)):
            print("用户列表不对")
            return dict(pre_user=set(t.index),base_user=set(base.index))
        t.to_csv(file_name, header=False)
    else:
        print("预测结果不满足数量条件，请检查对每个用户的预测数量")


def get_ans_filed(ans,n=0):
    """
    获取ans中的第n个字段，通常第0个字段是item_id，第1个字段是分数
    :param ans:
    :param n:
    :return:
    """
    ans_new = {}
    for k in ans:
        ans_new[k] = [e[n] for e in ans[k]]
    return ans_new

def reducemem(df):
    for i,(col,dtype) in enumerate(zip(df.columns,df.dtypes)):
        if(i%5==0):
            print(i,col,dtype)
        if(dtype in ["int8","int16","int32","int64"]):
            mx=df[col].max()
            # mn=df[col].min()
            if(mx<np.iinfo(np.int8).max):
                df[col]=df[col].astype(np.int8)
            elif(mx<np.iinfo(np.int16).max):
                df[col]=df[col].astype(np.int16)
            elif(mx<np.iinfo(np.int32).max):
                df[col]=df[col].astype(np.int32)
        elif(dtype in ["float16","float32","float64"]):
            mx = df[col].max()
            # mn=df[col].min()
            if (mx < np.finfo(np.float16).max):
                df[col] = df[col].astype(np.float16)
            elif (mx < np.finfo(np.float32).max):
                df[col] = df[col].astype(np.float32)
        else:
            print("mismath type",col,dtype)
    return df


def get_sample(recall_list,has_rate=None,n=100):
    print(n)
    """
    从各个召回ans中整理出分类器所需要的样本，has_rate默认全False，has_rate应该和recall_list长度相同
    :param recall_list:
    :param has_rate:
    :return:
    """
    if(has_rate is None):
        has_rate=[False for _ in recall_list]
    all_sample=pd.DataFrame(columns=["user_id","item_id"])
    try:
        for i, (lst, rate) in enumerate(zip(recall_list, has_rate)):
            ans = []
            if (rate):
                for k in lst:  # k用户id
                    for e, iii in zip(lst[k], range(n)):  # e是商品id和打分
                        ans.append([k] + list(e))
                ans = pd.DataFrame(ans)
                ans.columns = ["user_id", "item_id"] + ["score_from_" + str(i) + "_" + str(j) for j in
                                                        range(ans.shape[1] - 2)]
                ans = ans.drop_duplicates(["user_id", "item_id"])
                all_sample = pd.merge(all_sample, ans, how="outer", on=["user_id", "item_id"])
            else:
                for k in lst:  # 用户id
                    for e, iii in zip(lst[k], range(n)):  # 商品id
                        ans.append([k, e])
                all_sample = pd.merge(all_sample, pd.DataFrame(ans, columns=["user_id", "item_id"]).drop_duplicates(),
                                      how="outer", on=["user_id", "item_id"])
    except TypeError:
        print("ans_num is :",i)
        print("user id: ",k)
        print("ans: ",lst[k])
        raise ValueError
    return all_sample.drop_duplicates(["user_id","item_id"])

def buquan(ans,fn="yy",all_user=[]):
    df=pd.read_csv(f"{fn}.csv")
    size = df.drop_duplicates(["buyer_admin_id", "item_id"]).groupby("item_id").size().sort_values(ascending=False)
    for user_id in all_user:
        if(ans.get(user_id) is None):
            ans[user_id] = list(size[:30].index)
        else:
            if (len(ans[user_id]) < 30):
                ans[user_id] = ans[user_id] + list(size[:(30 - len(ans[user_id]))].index)
            else:
                ans[user_id]=ans[user_id][:30]
    return ans

if __name__=="__main__":
    print("fine")