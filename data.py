"""
准备线下验证集和label等,存到当前目录下
"""

import pandas as pd
from function import split_dataset
import os

def pipe_data():
    train=pd.read_csv("Antai_AE_round2_train_20190813.zip")
    test=pd.read_csv("Antai_AE_round2_test_20190813.zip")
    print(train.shape,test.shape)
    item_attr=pd.read_csv("Antai_AE_round2_item_attr_20190813.zip")
    train=train.merge(item_attr,how="left",on="item_id")
    test=test.merge(item_attr,how="left",on="item_id")
    print(train.shape,test.shape)
    train=train.drop(train[(train["buy_flag"]==0) & (train["irank"]==1)].index)
    xx=train[train["country_id"]=="xx"]
    yy=train[train["country_id"]=="yy"]
    zz=train[train["country_id"]=="zz"]
    test_yy=test[test["country_id"]=="yy"]
    test_zz=test[test["country_id"]=="zz"]
    xx.to_csv("xx.csv",index=False)
    yy.to_csv("yy.csv",index=False)
    zz.to_csv("zz.csv",index=False)
    test_yy.to_csv("test_yy.csv",index=False)
    test_zz.to_csv("test_zz.csv",index=False)

    #已经保证了每个数据集中的用户的irank1记录为购买
    xx=pd.read_csv("xx.csv")
    yy=pd.read_csv("yy.csv")
    zz=pd.read_csv("zz.csv")
    print(xx.shape)

    item=yy["item_id"].unique().tolist()+zz["item_id"].unique().tolist()
    user=xx[xx["item_id"].isin(item)]["buyer_admin_id"].unique()
    xx=xx[xx["buyer_admin_id"].isin(user)]
    print(xx.shape)

    print("all read")
    for df,name in zip([xx,yy,zz],["xx","yy","zz"]):

        print(f"write {name}")
        user = df[df["irank"]==1]['buyer_admin_id'].unique()
        df=df[df["buyer_admin_id"].isin(user)]
        #保证每个用户都有irank1的记录

        #保证用户至少有两条记录，进而保证train和label的用户数目一致
        user_size=df.groupby("buyer_admin_id").size()
        user=user_size[user_size>1].index

        # if(name=="xx"):#对xx国删去活跃度过低和活跃度过高的用户
        #     user = user_size[user_size <= user_size.quantile(0.999)].index
        df = df[df["buyer_admin_id"].isin(user)]
        df[df["irank"]==1].to_csv(f"{name}_clean_label.csv",index=False,header=True)
        df[df["irank"] != 1].to_csv(f"{name}_clean.csv", index=False, header=True)
