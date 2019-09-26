from cand_func import *
import heapq
import time

class Node:
    def __init__(self,a,b):
        self.a=a
        self.b=b
    def __lt__(self, other):
        assert isinstance(other,Node)
        return self.b<other.b
    def __eq__(self, other):
        assert isinstance(other, Node)
        return self.b == other.b
    def __repr__(self):
        return str((self.a,self.b))

    def __iter__(self):
        return iter([self.a,self.b])

def get_k_neighbor_user(data, k=3,p=0.5,time=True):
    user_sim = {}
    user_size = {}
    df = data[data["buy_flag"] == 1].copy()
    # assert isinstance(df,pd.DataFrame)

    n_item=df["item_id"].nunique()
    item_buy_table = df.groupby("item_id").apply(lambda x: set(x["buyer_admin_id"].values))
    assert isinstance(item_buy_table, pd.Series)
    print("consider time:",time)
    if(time):
        df_ = df.drop_duplicates(["buyer_admin_id", "item_id"])[["buyer_admin_id", "item_id", "day"]].set_index(
            ["buyer_admin_id", "item_id"])
        df_ = df_["day"]
        df_1 = df.drop_duplicates(["buyer_admin_id"])[["buyer_admin_id", "day"]].set_index("buyer_admin_id")
        df_1 = df_1["day"]
        for (iid, user_list) in zip(item_buy_table.index, item_buy_table):
            for uida in user_list:
                try:
                    user_size[uida] += 1
                except KeyError:
                    user_size[uida] = 1
                for uidb in user_list:
                    temp=1/np.log(2+np.abs(df_.loc[uida,iid]-df_1[uida]) +np.abs(df_.loc[uidb,iid]-df_1[uidb]) )
                    #temp=1
                    try:
                        user_sim[uida, uidb] += temp
                    except KeyError:
                        user_sim[uida, uidb] = temp
    else:
        for (iid, user_list) in zip(item_buy_table.index, item_buy_table):
            # if (iii % 1000 == 0):
            #     print(f"{iii}  ({iii / n_item})")
            for uida in user_list:
                try:
                    user_size[uida] += 1
                except KeyError:
                    user_size[uida] = 1
                for uidb in user_list:
                    #temp=1/np.log(2+np.abs(df_.loc[uida,iid]-df_1[uida]) +np.abs(df_.loc[uidb,iid]-df_1[uidb]) )
                    try:
                        user_sim[uida, uidb] += 1
                    except KeyError:
                        user_sim[uida, uidb] = 1
    # 获得用户共现矩阵user_sim和用户购买商品数量user_size

    for (k1, k2) in user_sim:
        user_sim[k1, k2] = user_sim[k1, k2] / (user_size[k1]**0.5 * user_size[k2]**p)

    user_k_neighbor = {}
    for (k1, k2), v in user_sim.items():
        if (k1 == k2):
            continue
        if (user_k_neighbor.get(k1) is None):
            user_k_neighbor[k1] = []
        if (user_k_neighbor.get(k2) is None):
            user_k_neighbor[k2] = []
        # 如果邻居还不够，就直接push
        if (len(user_k_neighbor[k1]) < k):
            heapq.heappush(user_k_neighbor[k1], Node(k2, v))
        else:
            t = user_k_neighbor[k1][0]  # 堆顶是最小的元素
            if (v > t.b):
                heapq.heappushpop(user_k_neighbor[k1], Node(k2, v))

        # if (len(user_k_neighbor[k2]) < k):
        #     heapq.heappush(user_k_neighbor[k2], Node(k1, v))
        # else:
        #     t = user_k_neighbor[k2][0]  # 堆顶是最小的元素
        #     if (v > t.b):
        #         heapq.heappushpop(user_k_neighbor[k2], Node(k1, v))
    return user_k_neighbor

def get_k_neighbor_item(data, k=3,p=0.5):
    item_sim = {}
    item_size = {}
    df = data[data["buy_flag"] == 1].copy()
    item_buy_table = df.groupby("buyer_admin_id").apply(lambda x: set(x["item_id"].values))#每个用户都买了哪些商品
    assert isinstance(item_buy_table, pd.Series)
    for uid, item_list in zip(item_buy_table.index, item_buy_table):
        #用户uid，和对应的购买列表itemlist
        for iida in item_list:
            try:
                item_size[iida] += 1#统计每个商品的被多少人买过
            except KeyError:
                item_size[iida] = 1
            for iidb in item_list:
                try:
                    item_sim[iida, iidb] += 1
                except KeyError:
                    item_sim[iida, iidb] = 1
    # 获得商品共现矩阵item_sim和商品被购买商品数量item_size

    for (k1, k2) in item_sim:
        item_sim[k1, k2] = item_sim[k1, k2] / (item_size[k1]**p * item_size[k2]**0.5)

    item_k_neighbor = {}
    for (k1, k2), v in item_sim.items():
        if (k1 == k2):
            continue
            # item_k_neighbor[k1], Node(k2, 1)
            # item_k_neighbor[k2], Node(k1, 1)
        if (item_k_neighbor.get(k1) is None):
            item_k_neighbor[k1] = []
        if (item_k_neighbor.get(k2) is None):
            item_k_neighbor[k2] = []
        # 如果邻居还不够，就直接push
        if (len(item_k_neighbor[k1]) < k):
            heapq.heappush(item_k_neighbor[k1], Node(k2, v))
        else:
            t = item_k_neighbor[k1][0]  # 堆顶是最小的元素
            if (v > t.b):
                heapq.heappushpop(item_k_neighbor[k1], Node(k2, v))
    return item_k_neighbor

def get_rate(data,k=30):

    df=data[data["buy_flag"]==1][["buyer_admin_id","item_id","day"]].drop_duplicates()
    rate={}
    for uid,iid,day in df.values:
        try:
            rate[uid, iid] += 1
        except KeyError:
            rate[uid, iid] = 1

    rate_k_top = {}
    for (k1, k2), v in rate.items():
        if (rate_k_top.get(k1) is None):
            rate_k_top[k1] = []
        # 如果邻居还不够，就直接push
        if (len(rate_k_top[k1]) < k):
            heapq.heappush(rate_k_top[k1], Node(k2, v))
        else:
            t = rate_k_top[k1][0]  # 堆顶是最小的元素
            if (v > t.b):
                heapq.heappushpop(rate_k_top[k1], Node(k2, v))
    return rate_k_top

def predict(user_neighbor,item_neighbor,rate,test_user,n=30,silence_user=None):
    ans={}
    for uid in test_user:
        #uid为目标用户
        if(user_neighbor.get(uid) is None):
            #如果没有邻居，goon
            ans[uid]=[]
            continue
        cand={}
        for neib,sim in user_neighbor[uid]:
        # 相邻的用户neib，相似度sim
            if(silence_user is not None and neib in silence_user):
                continue
            if(rate.get(neib) is not None):
                for iid,v in rate[neib]:
                #该用户喜欢的商品iid，和打分v
                    try:
                        cand[iid]+= sim*v
                    except KeyError:
                        cand[iid] = sim * v
                    #先保存这个商品
                    if(item_neighbor.get(iid) is None):
                        #如果这个商品没有邻居，直接下一个
                        continue
                    for iids,vs in item_neighbor[iid]:
                        #这个商品为iids，与iid商品的相似性为vs
                        try:
                            cand[iids]+= sim*v*vs
                        except KeyError:
                            cand[iids] = sim*v*vs
        cand = list(cand.items())
        cand.sort(key=lambda x:x[1],reverse=True)
        cand=[[e[0],e[1],ii] for ii,e in enumerate(cand)]
        ans[uid]=cand[:n]
    return ans


def usercf(data,test_user=None,k_neighbor=400,k_top_item=100,n=100,silence_user=None,p=0.5,tm=True):
    """
    data应该有 buyer_admin_id和 item_id 、log_time
    :param data:
    :param test_user:
    :param k_neighbor:
    :param k_top_item:
    :param n:
    :return:
    """
    time_start=time.time()
    data = data.sort_values("log_time",ascending=False)
    data["day"] = data["log_time"].str[8:10].astype("int")
    #每个用户每天只能对每个商品贡献一次
    data=data.drop_duplicates(["buyer_admin_id","item_id","day"])

    print("cal neighbor of user")
    user_neighbor = get_k_neighbor_user(data, k=k_neighbor,p=p,time=tm)
    print("cal neighbor of item")
    item_neighbor= get_k_neighbor_item(data, k=k_neighbor,p=0.5)
    print("cal rate")
    rate = get_rate(data,k=k_top_item)
    if(test_user is None):
        test_user=data.buyer_admin_id.unique()
    ans=predict(user_neighbor=user_neighbor,item_neighbor=item_neighbor, rate=rate, test_user=test_user,n=n,silence_user=silence_user)
    print(f"time: {time.time()-time_start}")
    return ans,user_neighbor


def mydrop(df,fn,lim=None,get_user=False):
    if(lim is None):
        return df
    user_size=df[df["country_id"]==fn].groupby("buyer_admin_id").size()
    if (get_user):
        user = user_size[(user_size < user_size.quantile(lim[0])) | (user_size > user_size.quantile(lim[1]))].index
        return user
    user = user_size[(user_size >= user_size.quantile(lim[0])) & (user_size <= user_size.quantile(lim[1]))].index
    df=df[(df["country_id"]!=fn) | (df["buyer_admin_id"].isin(user))]
    return df


#if(__name__=="__main__"):
def pipe_usercf():
    #对于其他国家过度活跃的用户
    #对于没有购买记录的用户，无法使用usercf进行推荐，也无法使用复购进行推荐，根据数据集特点，已查看的
    p_map=dict([
        (("xx","yy"),1),
        (("xx", "zz"),1.1),
        (("yy", "yy"),1.04),
        (("yy", "zz"),1.2),
        (("zz", "yy"),1.06667),
        (("zz", "zz"),1.42)
    ])
    #测试数据集部分
    for fn0 in ["xx","yy","zz"]:
    #for fn0 in [ "yy", "zz"]:
        for fn in ["yy","zz"]:
            tm=True
            print(f"from_{fn0}_to_{fn}_test")
            data=read_df([f"{fn0}_clean.csv",f"{fn0}_clean_label.csv",f"test_{fn}.csv"])
            if(fn0=="xx"):
                # tm=False
                data = mydrop(df=data, fn=fn0, lim=[0, 0.999])

            test_user=pd.read_csv(f"test_{fn}.csv")["buyer_admin_id"].unique()
            ans,user_neighbor=usercf(data=data,test_user=test_user,n=100,p=p_map[fn0,fn],tm=tm)
            save(ans, f"from_{fn0}_to_{fn}_test.pkl")
            #save(user_neighbor, f"user_neighbor_from_{fn0}_to_{fn}_test.pkl")

    #训练数据集部分
    #xx对yy和zz
    for fn in ["yy","zz"]:
        print(f"from_xx_to_{fn}")
        data = read_df([f"xx_clean.csv", f"xx_clean_label.csv", f"{fn}_clean.csv"])
        data = mydrop(df=data, fn="xx", lim=[0, 0.999])
        test_user = pd.read_csv(f"{fn}_clean.csv")["buyer_admin_id"].unique()
        test_label = pd.read_csv(f"{fn}_clean_label.csv")  ###预测谁
        test_label = read_map_from_df(test_label[["buyer_admin_id", "item_id"]])
        ans,user_neighbor = usercf(data=data, test_user=test_user, n=100,p=p_map["xx",fn],tm=False)
        print(score(ans, test_label)[:-2])
        save(ans, f"from_xx_to_{fn}.pkl")
        #save(user_neighbor, f"user_neighbor_from_xx_to_{fn}.pkl")

    print(f"from_yy_to_zz")
    data = read_df([f"yy_clean.csv",f"yy_clean_label.csv", f"zz_clean.csv"])
    #data = mydrop(df=data, fn="yy", lim=[0, 0.999])
    test_user = pd.read_csv(f"zz_clean.csv")["buyer_admin_id"].unique()
    test_label = pd.read_csv(f"zz_clean_label.csv")  ###预测谁
    test_label=read_map_from_df(test_label[["buyer_admin_id","item_id"]])
    ans,user_neighbor = usercf(data=data, test_user=test_user, n=100,p=p_map["yy","zz"])
    print(score(ans,test_label)[:-2])
    save(ans, f"from_yy_to_zz.pkl")
    #save(user_neighbor, f"user_neighbor_from_yy_to_zz.pkl")

    print(f"from_zz_to_yy")
    data = read_df([ f"yy_clean.csv",f"zz_clean.csv", f"zz_clean_label.csv"]) # 有什么数据
    #data=mydrop(df=data,fn="zz",lim=[0,0.998])
    test_user = pd.read_csv(f"yy_clean.csv")["buyer_admin_id"].unique()#预测谁
    test_label = pd.read_csv(f"yy_clean_label.csv") #答案是什么
    test_label = read_map_from_df(test_label[["buyer_admin_id", "item_id"]])
    ans,user_neighbor = usercf(data=data, test_user=test_user, n=100,p=p_map["zz","yy"])
    print(score(ans, test_label)[:-2])
    save(ans, f"from_zz_to_yy.pkl")
    #save(user_neighbor, f"user_neighbor_from_zz_to_yy.pkl")

    print(f"from_zz_to_zz")
    data = read_df([ f"zz_clean.csv"])
    #user=mydrop(df=data,fn="zz",lim=[0,0.99],get_user=True)
    test_user = pd.read_csv(f"zz_clean.csv")["buyer_admin_id"].unique()
    test_label = pd.read_csv(f"zz_clean_label.csv")  ###预测谁
    test_label=read_map_from_df(test_label[["buyer_admin_id","item_id"]])
    ans,user_neighbor = usercf(data=data, test_user=test_user, n=100,p=p_map["zz","zz"])
    print(score(ans,test_label)[:-2])
    save(ans, f"from_zz_to_zz.pkl")
    #save(user_neighbor, f"user_neighbor_from_zz_to_zz.pkl")

    print(f"from_yy_to_yy")
    data = read_df([ f"yy_clean.csv"]) # 有什么数据
    #user = mydrop(df=data, fn="yy", lim=[0, 0.99], get_user=True)
    test_user = pd.read_csv(f"yy_clean.csv")["buyer_admin_id"].unique()#预测谁
    test_label = pd.read_csv(f"yy_clean_label.csv") #答案是什么
    test_label = read_map_from_df(test_label[["buyer_admin_id", "item_id"]])
    ans,user_neighbor = usercf(data=data, test_user=test_user, n=100,p=p_map["yy","yy"])
    print(score(ans, test_label)[:-2])
    save(ans, f"from_yy_to_yy.pkl")
    #save(user_neighbor, f"user_neighbor_from_yy_to_yy.pkl")