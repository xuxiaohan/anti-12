from function import *
import gc

def get_feat(fn,offline,statistic_list,result="result.csv"):
    item_attr=pd.read_csv("Antai_AE_round2_item_attr_20190813.zip").rename(columns={"buyer_admin_id":"user_id"})

    #sample=recall_sample(fn,offline)
    # sample=get_sample([ans],[True])
    if(offline):
        #sample=load(f"{fn}_sample.pkl")
        sample = pd.read_csv(f"{fn}_sample.csv")
    else:
        # sample = load(f"{fn}_sample_test.pkl")
        sample = pd.read_csv(f"{fn}_sample_test.csv")

    feat=[f"score_from_{i}_0" for i in range(5,9)]+[f"score_from_0_1"]+[f"score_from_{i}_1" for i in range(15,19)]
    #此时feat中已经包含了：
    #是否被该用户购买过（排名）
    #三个usercf的得分信息
    #商品热度在当天的占比
    #商品热度在该商店的占比
    #商品热度在该商店当天的占比
    #商品热度在种类的占比
    #商品热度在该种类当天的占比

    if(offline):
        sample=sample[['user_id', 'item_id',"label"]+feat]
    else:
        sample = sample[['user_id', 'item_id'] + feat]
    gc.collect()
    user_id=sample["user_id"].unique()
    item_id=sample["item_id"].unique()

    #抽取2000个样本做测试
    # sample=sample[sample["user_id"].isin(user_id[:2000])]

    print(f"sample shape is {sample.shape}, user num is {user_id.shape[0]}, item num is {item_id.shape[0]}")
    sample = sample.merge(item_attr, on="item_id", how="left")
    #抽取特征
    data=read_df(statistic_list).rename(columns={"buyer_admin_id":"user_id"}).sort_values("log_time",ascending=False)#线下数据
    assert isinstance(data,pd.DataFrame)
    data["day"]=data["log_time"].str[8:10].astype("int")
    data1=data[data["buy_flag"]==1]
    print("data shape",data.shape,data1.shape)
    user_last_act=data.drop_duplicates("user_id")[["user_id","item_id","cate_id","store_id","day","item_price"]]
    user_last_buy = data1.drop_duplicates("user_id")[["user_id","item_id","cate_id","store_id","log_time","day","item_price"]]
    user_day = data[["user_id", "day"]].drop_duplicates("user_id").set_index("user_id")["day"]
    sample=sample.merge(user_last_act,how="left",on="user_id",suffixes=["","_lastact"])
    sample = sample.merge(user_last_buy, how="left", on="user_id", suffixes=["", "_lastbuy"])

    user_cate_last_price=data.drop_duplicates(["user_id","cate_id"])[["user_id","cate_id","item_price"]].rename(columns={"item_price":"user_cate_last_price"})
    user_store_last_price = data.drop_duplicates(["user_id", "store_id"])[["user_id", "store_id", "item_price"]].rename(
        columns={"item_price": "user_store_last_price"})

    print("sample shape",sample.shape)
    #用户侧
    feat=feat+["user_act"]
    #用户活跃度
    user_act = data[data["buy_flag"]==1].drop_duplicates(["user_id","item_id","day"]).groupby("user_id").size()
    sample["user_act"] = sample["user_id"].map(user_act)
    del user_act


    # #####
    ###userday
    sample["user_day"]=sample["user_id"].map(user_day)
    ###和对该cate的最后一次行为的差价
    sample=sample.merge(user_cate_last_price,how="left",on=["user_id","cate_id"])
    sample = sample.merge(user_store_last_price,how="left",on=["user_id","store_id"])
    sample["diff_last_cate_price"]=sample["item_price"]-sample["user_cate_last_price"]
    sample["diff_last_store_price"] = sample["item_price"] - sample["user_store_last_price"]
    feat+=["diff_last_cate_price","diff_last_store_price"]

    print("len of feat",len(feat))

    #某商品的回头客数量
    t=data1.drop_duplicates(["user_id","item_id","day"]).groupby(["user_id","item_id"]).size()
    t=t[t>1].reset_index()#定义大于1次购买才成为回头客
    t=t.groupby("item_id").size()#item级别的回头客
    sample["huitou_item"]=sample["item_id"].map(t)
    feat.append("huitou_item")

    #某店铺的回头客数量
    t = data1.drop_duplicates(["user_id", "item_id", "day"]).groupby(["user_id","store_id", "item_id"]).size()
    t = t[t > 1].reset_index()  # 定义大于1次购买才成为回头客
    t = t.groupby("store_id").size()  # item级别的回头客
    sample["huitou_store"] = sample["store_id"].map(t)
    feat.append("huitou_store")
    print("len of feat", len(feat))

    t=data.drop_duplicates(["user_id","item_id"])[["user_id","item_id","buy_flag"]]
    sample=sample.merge(t,on=["user_id","item_id"],how="left")
    sample["buy_flag"]=sample["buy_flag"].fillna(0.5)
    feat.append("buy_flag")

    #该商品所在的种类有多少商品，所在的店有多少商品
    t=item_attr.groupby("cate_id").size()
    sample["cate_size"]=sample["cate_id"].map(t)
    t = item_attr.groupby("store_id").size()
    sample["store_size"] = sample["store_id"].map(t)
    print("len of feat", len(feat))
    #####
    feat+=["cate_size","store_size"]

    #这个店的这类商品被多少人喜欢
    t=data1.drop_duplicates(["user_id","item_id"]).groupby(["store_id","cate_id"]).size().rename("store_cate_buy_num")
    sample = sample.merge(t, how="left", on=["store_id", "cate_id"])
    feat.append("store_cate_buy_num")
    print("len of feat", len(feat))
    # 这个商品在这个商店的这个种类的销售占比
    t = (data1.groupby(["item_id","cate_id","store_id"]).size()/data1.groupby(["cate_id","store_id"]).size()).rename("item_in_cate_store_rate")
    t=t.droplevel([0,1])
    sample = sample.merge(t, how="left", on="item_id")
    feat.append("item_in_cate_store_rate")

    #这个商品在这个商店的这个种类的价格升序排序
    #t=item_attr.sort_values("item_price").groupby(["store_id","cate_id"]).apply(lambda x:x.reset_index(drop=True).index)
    #参考某个score
    print("len of feat", len(feat))

    ###该种类商品价格的标准差
    t = item_attr.groupby("cate_id")["item_price"].std()
    sample["cate_price_std"] = sample["cate_id"].map(t)
    feat += ["user_day", "cate_price_std"]

    # #该店有多少该种类的商品
    t = item_attr.groupby(["store_id", "cate_id"]).size().rename("store_cate_item_num")
    sample = sample.merge(t, how="left", on=["store_id", "cate_id"])
    feat.append("store_cate_item_num")
    # #该店有多少种类的商品
    t = item_attr.drop_duplicates(["store_id", "cate_id"]).groupby("store_id").size().rename("store_cate_num")
    sample = sample.merge(t, how="left", on="store_id")
    feat.append("store_cate_num")
    # #该种类有多大比例的商品是在该店的有的
    sample["item_rate_cate_store"] = sample["store_cate_item_num"] / sample["cate_size"]
    feat.append("item_rate_cate_store")
    # #该种类的商品在多少店有
    t = item_attr.drop_duplicates(["store_id", "cate_id"]).groupby("cate_id").size().rename("cate_store_num")
    sample = sample.merge(t, how="left", on="cate_id")
    feat.append("cate_store_num")
    print("len of feat", len(feat))
    #商品侧
    #商品被多少人购买过
    ans=data1.drop_duplicates(["user_id","item_id"]).groupby("item_id").size()
    sample["item_buy_ppl"]=sample["item_id"].map(ans).fillna(0)
    feat.append("item_buy_ppl")

    #和最后一次记录或最后一次购买的差价
    sample["diff_price1"]=sample["item_price"]-sample["item_price_lastbuy"]
    # sample["diff_price"] = sample["item_price"] - sample["item_price_lastact"]#没啥用
    feat+=["diff_price1"]
    print("len of feat", len(feat))
    ### 和最后一次记录或最后一次购买的差价与最后一次价格的比值
    # sample["diff_price1_rate"] = sample["diff_price1"]/sample["item_price_lastbuy"]
    # sample["diff_price_rate"] = sample["diff_price"]/sample["item_price_lastact"]
    # feat += ["diff_price1_rate", "diff_price_rate"]

    #和最后一次记录或最后一次购买是否在同一商店或同一种类
    sample["is_same_store_buy"]=(sample["store_id"]==sample["store_id_lastbuy"]).astype("int")
    sample["is_same_store_act"] = (sample["store_id"] == sample["store_id_lastact"]).astype("int")
    sample["is_same_cate_buy"] = (sample["cate_id"] == sample["cate_id_lastbuy"]).astype("int")
    sample["is_same_cate_act"] = (sample["cate_id"] == sample["cate_id_lastact"]).astype("int")
    feat += ["is_same_store_buy","is_same_store_act","is_same_cate_buy","is_same_cate_act"]
    print("len of feat", len(feat))
    #该商品和用户对该种类商品查看过的最低价的差距
    d=data.dropna().sort_values("item_price").drop_duplicates(["user_id","cate_id"]).set_index(["user_id","cate_id"])["item_price"]
    d=d.rename("lowest_price")
    sample=pd.merge(sample,d,how="left",on=["user_id","cate_id"])
    sample["diff_lowest_price"]=sample["item_price"]-sample["lowest_price"]
    feat.append("diff_lowest_price")
    print("len of feat", len(feat))
    # 该商品和用户对该商店商品查看过的最低价的差距
    d = data.dropna().sort_values("item_price").drop_duplicates(["user_id", "store_id"]).set_index(["user_id", "store_id"])[
        "item_price"]
    d = d.rename("lowest_price_store")
    sample = pd.merge(sample, d, how="left", on=["user_id", "store_id"])
    sample["diff_lowest_price_store"] = sample["item_price"] - sample["lowest_price_store"]
    feat.append("diff_lowest_price_store")
    print("len of feat", len(feat))
    # 该商品和用户对该种类商品买过的最低价的差距
    d = \
    data1.dropna().sort_values("item_price").drop_duplicates(["user_id", "cate_id"]).set_index(["user_id", "cate_id"])[
        "item_price"]
    d = d.rename("lowest_price1")
    sample = pd.merge(sample, d, how="left", on=["user_id", "cate_id"])
    sample["diff_lowest_price1"] = sample["item_price"] - sample["lowest_price1"]
    feat.append("diff_lowest_price1")
    print("len of feat", len(feat))
    # 该商品和用户对该商店商品买过的最低价的差距
    d = \
    data1.dropna().sort_values("item_price").drop_duplicates(["user_id", "store_id"]).set_index(["user_id", "store_id"])[
        "item_price"]
    d = d.rename("lowest_price_store1")
    sample = pd.merge(sample, d, how="left", on=["user_id", "store_id"])
    sample["diff_lowest_price_store1"] = sample["item_price"] - sample["lowest_price_store1"]
    feat.append("diff_lowest_price_store1")

    print("len of feat", len(feat))

 #=======================

    #该商品和用户对该种类商品查看过的最高的差距
    d=data.dropna().sort_values("item_price",ascending=False).drop_duplicates(["user_id","cate_id"]).set_index(["user_id","cate_id"])["item_price"]
    d=d.rename("highest_price")
    sample=pd.merge(sample,d,how="left",on=["user_id","cate_id"])
    sample["diff_highest_price"]=sample["item_price"]-sample["highest_price"]
    feat.append("diff_highest_price")
    print("len of feat", len(feat))
    # 该商品和用户对该商店商品查看过的最高价的差距
    d = data.dropna().sort_values("item_price",ascending=False).drop_duplicates(["user_id", "store_id"]).set_index(["user_id", "store_id"])[
        "item_price"]
    d = d.rename("highest_price_store")
    sample = pd.merge(sample, d, how="left", on=["user_id", "store_id"])
    sample["diff_highest_price_store"] = sample["item_price"] - sample["highest_price_store"]
    feat.append("diff_highest_price_store")
    print("len of feat", len(feat))
    # 该商品和用户对该种类商品买过的最高价的差距
    d = \
    data1.dropna().sort_values("item_price",ascending=False).drop_duplicates(["user_id", "cate_id"]).set_index(["user_id", "cate_id"])[
        "item_price"]
    d = d.rename("highest_price1")
    sample = pd.merge(sample, d, how="left", on=["user_id", "cate_id"])
    sample["diff_highest_price1"] = sample["item_price"] - sample["highest_price1"]
    feat.append("diff_highest_price1")
    print("len of feat", len(feat))
    # 该商品和用户对该商店商品买过的最高价的差距
    d = \
    data1.dropna().sort_values("item_price",ascending=False).drop_duplicates(["user_id", "store_id"]).set_index(["user_id", "store_id"])[
        "item_price"]
    d = d.rename("highest_price_store1")
    sample = pd.merge(sample, d, how="left", on=["user_id", "store_id"])
    sample["diff_highest_price_store1"] = sample["item_price"] - sample["highest_price_store1"]
    feat.append("diff_highest_price_store1")

# =======================

    print("len of feat", len(feat))


    df=data[data["user_id"].isin(user_id)].copy()
    df[f"user_day"]=df["user_id"].map(user_day)
    df1=df[df["buy_flag"]==1]
    #商品所在种类/商店在用户最后一天的行为/购买次数
    #在最后三天的行为/购买次数
    for i in [0,1,2]:
        for col in ["cate_id","store_id"]:
            print(f"{i}_{col}")
            d = df[df["day"] == (df["user_day"] - i)].groupby(["user_id",col]).size()  # 获得用户倒数第i天的数据
            d1 = df1[df1["day"] == (df1["user_day"] - i)].groupby(["user_id",col]).size()  # 获得用户倒数第i天的数据
            sample=pd.merge(sample,d.rename(f"{col}_{i}"),how="left",on=["user_id",col])
            sample[f"{col}_{i}"]=sample[f"{col}_{i}"].fillna(0)
    for col in ["cate_id", "store_id"]:
        sample[f"{col}_1"]=sample[f"{col}_1"]+sample[f"{col}_0"]
        sample[f"{col}_2"] = sample[f"{col}_1"] + sample[f"{col}_2"]

    feat+=['cate_id_0', 'store_id_0', 'cate_id_1', 'store_id_1', 'cate_id_2', 'store_id_2']
    ###该商品价格和用户最后一天行为/购买的平均价的差价  #用最低价会不会更好？ 或者最高价
    d = df[df["day"] == (df["user_day"])].groupby(["user_id"])["item_price"].min()  # 获得用户倒数第i天的数据
    sample["diff_last_day_max"]=sample["user_id"].map(d)-sample["item_price"]
    d = df[df["day"] == (df["user_day"])].groupby(["user_id"])["item_price"].max()
    sample["diff_last_day_min"] = sample["user_id"].map(d) - sample["item_price"]
    feat+=["diff_last_day_max","diff_last_day_min"]
    print("len of feat", len(feat))
    # for i in range(1,9):
    #     sample[f"score_from_{i}_0"]=sample[f"score_from_{i}_0"].fillna(0)
    # sample["user_act"]=sample["user_act"].fillna(0)

    # sample[feat]=sample[feat]/(sample[feat].max()-sample[feat].min())
    print(sample["user_id"].nunique())
    #save(sample,result)
    print("save sample as csv format file")
    print("befor reduce:",sample.memory_usage().sum()/1000**2,"MB")
    sample=reducemem(sample)
    print("after reduce:", sample.memory_usage().sum() / 1000 ** 2, "MB")
    if(offline):
        sample[["user_id","item_id","label"]+feat].to_csv(result)
    else:
        sample[["user_id", "item_id"] + feat].to_csv(result)
    return feat


def preprocess(fn,offline):
    if(offline):
        statistic_list = [f"test_{fn}.csv",f"{fn}_clean.csv"]

        print("do for train set of",fn)
        get_feat(fn=fn,offline=True,statistic_list=statistic_list,result=f"{fn}_sample_feat.csv")
    else:
        print(f"do for test set of {fn}")
        statistic_list = [fn+".csv", "test_" + fn + ".csv"]
        get_feat(fn=fn, offline=False, statistic_list=statistic_list, result=f"{fn}_sample_test_feat.csv")

#if(__name__=="__main__"):
def pipe_preprocessing():
    """
    fn in ["xx","yy","zz"]
    offline in [True,False]
    """

    for status in [True,False]:
        for fn in ["yy","zz"]:
            preprocess(fn, status)

if __name__ == '__main__':
    pipe_preprocessing()