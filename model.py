from cand_func import *
from sklearn.model_selection import ParameterSampler,ParameterGrid
from scipy.stats import randint,uniform
import seaborn as sns
import random
import xgboost as xgb
import heapq
import time
import os
from collections import Iterable,Iterator
import matplotlib.pyplot as plt
import gc


class Node(list):
    def __lt__(self, other):
        assert isinstance(other,Node)
        return self[1][0]<other[1][0]

    def __eq__(self, other):
        assert isinstance(other, Node)
        return self[1][0]== other[1][0]

    def __gt__(self, other):
        assert isinstance(other, Node)
        return self[1][0] > other[1][0]


def kfold_average(sample,test,label,feat,params=None,fn="yy",n_jobs=4,recall_num=30,k=4,res_name="res_yy.pkl"):
    """

    :param sample: 训练样本是哪些
    :param test: 测试样本有哪些
    :param label: 用于给score函数评测的map类型的label
    :param feat: 特征有哪些
    :return: 返回对训练集和测试集的预测结果
    """
    user = sample["user_id"].unique()
    test_ans=test[["user_id","item_id"]].copy()
    test_ans["prob"]=0
    test_ans["status"] = 0
    kf = KFold(n_splits=k, shuffle=True, random_state=2019)
    if(params is None):
        params=dict()
    print("params: ",params)
    timea=time.time()
    sssc= []
    impts=pd.DataFrame()
    all_res={}
    for jj,(uid_train, uid_val) in enumerate(kf.split(user)):
        model = xgb.XGBClassifier(n_jobs=n_jobs, **params)
        gc.collect()
        train_user = user[uid_train]
        val_user = user[uid_val]

        X_train = sample[sample["user_id"].isin(train_user)]
        y_train = X_train["label"]

        X_val = sample[sample["user_id"].isin(val_user)]
        y_val = X_val["label"]

        model.fit(X_train[feat], y_train)#训练
        impts[f"fold_{jj}"]=model.feature_importances_.copy()#存特征重要性
        pre = model.predict_proba(X_val[feat])[:, 1]#预测
        ans = X_val[["user_id", "item_id"]].copy()#整理
        ans["pre"] = pre
        group = ans.groupby(["user_id"])
        res = {}
        for i, e in group:
            res[i] = e.nlargest(recall_num, "pre", keep="all")["item_id"].values.tolist()[:recall_num]#对每个用户只保留recallnum个结果
        all_res.update(res)#记录这一折用户的结果
        if (label is not None):
            lb = {}
            for e in val_user:
                lb[e] = label[e]
            sc0 = score(res, lb,n=recall_num)[:-2]
            print(f"top {recall_num}:",sc0)
            sssc.append(sc0)
        pre = model.predict_proba(test[feat])[:, 1]  # 预测
        test_ans["prob"]+=pre
        del model
        gc.collect()
    sccc=np.mean(sssc, axis=0)
    print(f"average for top {recall_num}: ", sccc)
    print("time: ",time.time()-timea)
    test_ans0=test_ans.sort_values("prob",ascending=False).groupby("user_id").apply(lambda x: x["item_id"].values[:recall_num].tolist())
    test_ans0=dict(test_ans0)#用于提交的格式
    print("mean item num predict for test",np.mean(get_len_each_ans(test_ans0)))
    # save(all_res, f"res_offline_{fn}.pkl")
    # save(test_ans0,res_name)
    impts.index = feat
    # impts.to_csv(f"feat_importance_{fn}.csv")  # 存特征重要性
    return all_res,test_ans0,sccc,impts

def predict(sample,feat,test,res_name=None,best=None,buq=True,fn="zz",n_jobs=4):
    """

    :param sample: 训练样本
    :param feat: 特征
    :param test: 测试样本
    :param best: 最好的n个参数，iterable，最后结果为多个模型平均
    :param res_name: 结果存成什么文件名
    :param buq: 是否补全
    :param fn: 用哪个国家的热门进行补全,以及补全all_user取自哪个国家
    :return: 返回结果
    """
    if(res_name is None):
        res_name=f"predict4{fn}"
    if(best is None):
        return None
    if(isinstance(best,ParameterSampler)):
        pass
    elif(isinstance(best[0],Node)):
        best=[params[0] for params in best if isinstance(params[0],dict)]

    ans = test[["user_id", "item_id"]].copy()
    ans["pre"] = 0

    for ii,params in enumerate(best):
        print(ii,"========================")
        print(params)
        model = xgb.XGBClassifier(n_jobs=n_jobs, **params)
        model.fit(sample[feat], sample["label"])
        pre = model.predict_proba(test[feat])[:, 1]
        ans["pre"] += pre
    group = ans.groupby(["user_id"])
    res = {}
    for i, e in group:
        res[i] = e.nlargest(30, "pre", keep="all")["item_id"].values.tolist()[:30]

    all_user = pd.read_csv(f"test_{fn}.csv")["buyer_admin_id"].unique()
    if(buq):
        res = buquan(res, fn, all_user)
    save(res, res_name)
    return res

def pipe_model(n_run=15):
    feat=None

    params_lll = [[{
        "n_estimators": m,
        "tree_method": "gpu_hist",
        "max_depth": n
    } for m,n in zip(np.random.randint(400,600,n_run),np.random.randint(4,6,n_run))],
        [{
            "n_estimators": m,
            "tree_method": "gpu_hist",
            "max_depth": n
        } for m,n in zip(np.random.randint(400,600,n_run),np.random.randint(4,6,n_run))],
    ]
    for fn,params_list in zip(["yy","zz"],params_lll):
        gc.collect()
        sample = pd.read_csv(f"{fn}_sample_feat.csv", index_col=0)
        data=read_df([f"test_{fn}.csv",f"{fn}_clean.csv"]).rename(columns={"buyer_admin_id":"user_id"}).sort_values("log_time",ascending=False)#线下数据
        assert isinstance(data,pd.DataFrame)
        t=data[data["buy_flag"]==0].drop_duplicates(["user_id","item_id"]).groupby("item_id").size()
        sample["notbuy_size"]=sample["item_id"].map(t)#加了一个特征
        label = pd.read_csv(f"{fn}_clean_label.csv")
        label = read_map_from_df(label[["buyer_admin_id", "item_id"]])
        print("sample shape", sample.shape)
        if(feat is None):
            feat = []
            for e in sample.columns:
                if (e not in ["user_id", "item_id", "label", "buyer_admin_id", "score_from_1_0", "score_from_2_0",
                              "score_from_3_0", "score_from_4_0", "diff_price", "diff_last_day_median",
                              "cate_id_1","cate_id_2","huitou_item","huitou_store"]):
                    feat.append(e)
        print("len of feat", len(feat))
        print(feat)
        test = pd.read_csv(fn + "_sample_test_feat.csv", index_col=0)
        test["notbuy_size"] = test["item_id"].map(t)#加了一个特征
        del data,t
        sample=sample[["user_id", "item_id", "label"]+feat]
        test = test[["user_id", "item_id"] + feat]
        gc.collect()
        bst=[0,0,0,0]
        for ii,params in enumerate(params_list):
            print(ii)
            ans_train,ans_test,best,impts=kfold_average(sample, test, label, feat,params=params, fn=fn, n_jobs=4,
                          recall_num=30,res_name=f"res_{fn}.pkl")
            if(best[0]>bst[0]):
                save(ans_test,f"to_sub_{fn}.pkl")
                save(ans_train,f"offline_{fn}.pkl")
                save([params,best], f"best_{fn}_now.pkl")
                bst=best
                impts.to_csv(f"feat_importance_{fn}.csv", index=False)  # 存特征重要性

    res_yy=load(f"to_sub_yy.pkl")
    res_zz=load(f"to_sub_zz.pkl")
    ans=merge_ans([res_yy,res_zz])
    to_submit(ans,"to_submit.csv")


if __name__ == '__main__':
    print("fine")