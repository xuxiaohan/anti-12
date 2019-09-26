from cand_func import *


def others(fn="zz", n=50, offline=True,diff_price=100):
    if (offline):
        file = f"{fn}_clean.csv"
        label = pd.read_csv(f"{fn}_clean_label.csv")
        user = label["buyer_admin_id"]
        label = read_map_from_df(label[["buyer_admin_id", "item_id"]])
        df = pd.read_csv(file)
    else:
        file = f"{fn}.csv"
        df = read_df([f"{fn}_clean.csv", f"{fn}_clean_label.csv", f"test_{fn}.csv"])
        user = pd.read_csv(f"test_{fn}.csv")["buyer_admin_id"].unique()

    ans0 = {}
    ans1 = {}
    ans2 = {}
    ans3 = {}
    ans4 = {}
    ans5 = {}  # 和最后一条行为记录同种类且价格相近的topn
    ans6 = {}  # 和最后一条行为记录同商店且价格相近的topn
    ans7 = {}  # 和最后一条购买记录同种类且价格相近的topn
    ans8 = {}  # 和最后一条购买记录同商店且价格相近的topn
    ans9 = {}
    ans10 = {}

    df = df.sort_values("log_time", ascending=False)
    df["day"] = df["log_time"].str[8:10].astype("int")
    user_last = df.drop_duplicates("buyer_admin_id").set_index("buyer_admin_id")  # 用户的最后一条行为记录
    user_last_buy = df[df["buy_flag"] == 1].drop_duplicates("buyer_admin_id").set_index("buyer_admin_id")  # 用户的最后一条购买记录

    # 该用户最后两天里行为次数最多的种类/商店
    user_day=df.drop_duplicates("buyer_admin_id").set_index("buyer_admin_id")["day"]
    df["user_day"]=df["buyer_admin_id"].map(user_day)
    dd = df[(df["user_day"]==df["day"]) | (df["user_day"]-1==df["day"])]#选择该用户的最后一天的记录或者最后两天的记录，找出它最常看的那个店或者那个种类
    user_most_cate = \
        dd.groupby(["buyer_admin_id", "cate_id"]).size().sort_values(ascending=False).reset_index().drop_duplicates(
            "buyer_admin_id").set_index("buyer_admin_id")["cate_id"]
    user_most_store = \
        dd.groupby(["buyer_admin_id", "store_id"]).size().sort_values(ascending=False).reset_index().drop_duplicates(
            "buyer_admin_id").set_index("buyer_admin_id")["store_id"]

    df = df[df["buy_flag"] == 1].drop_duplicates(["buyer_admin_id", "item_id", "day"])
    item_day_ppl = df.groupby(["item_id", "day"]).size() / df.groupby("day").size()
    item_store_day_ppl = df.groupby(["item_id", "store_id", "day"]).size() / df.groupby(["store_id", "day"]).size()
    item_cate_day_ppl = df.groupby(["item_id", "cate_id", "day"]).size() / df.groupby(["cate_id", "day"]).size()
    item_store_ppl = df.groupby(["item_id", "store_id"]).size() / df.groupby(["store_id"]).size()
    item_cate_ppl = df.groupby(["item_id", "cate_id"]).size() / df.groupby(["cate_id"]).size()

    item_day_ppl = item_day_ppl.sort_values(ascending=False).reset_index().groupby("day").apply(
        lambda x: x[["item_id", 0]].values[:n].tolist())
    item_store_day_ppl = item_store_day_ppl.sort_values(ascending=False).reset_index().groupby(
        ["store_id", "day"]).apply(lambda x: x[["item_id", 0]].values[:n].tolist())
    item_cate_day_ppl = item_cate_day_ppl.sort_values(ascending=False).reset_index().groupby(["cate_id", "day"]).apply(
        lambda x: x[["item_id", 0]].values[:n].tolist())
    item_store_ppl = item_store_ppl.sort_values(ascending=False).reset_index().groupby("store_id").apply(
        lambda x: x[["item_id", 0]].values[:n].tolist())
    item_cate_ppl = item_cate_ppl.sort_values(ascending=False).reset_index().groupby("cate_id").apply(
        lambda x: x[["item_id", 0]].values[:n].tolist())

    item_attr = pd.read_csv("Antai_AE_round2_item_attr_20190813.zip")
    cate2item = item_attr[["cate_id", "item_id", "item_price"]].drop_duplicates(["cate_id", "item_id"]).set_index(
        "cate_id")
    store2item = item_attr[["store_id", "item_id", "item_price"]].drop_duplicates(["store_id", "item_id"]).set_index(
        "store_id")
    n_not_nuy = 0
    for ii, uid in enumerate(user):
        if (ii % 5000 == 0):
            print(f"{ii}  ({ii / user.shape[0]})")
        last = user_last.loc[uid]#和最后一次行为相关的（不区分购买还是查看）
        try:
            ans0[uid] = [b + [a] for a, b in enumerate(item_day_ppl[last["day"]][:n])]
        except:
            ans0[uid] = []
        try:#同一个店的热门
            ans1[uid] = [b + [a] for a, b in enumerate(item_store_day_ppl[last["store_id"], last["day"]][:n])]
        except:
            ans1[uid] = []
        try:
            ans2[uid] = [b + [a] for a, b in enumerate(item_cate_day_ppl[last["cate_id"], last["day"]][:n])]
        except:
            ans2[uid] = []
        try:
            ans3[uid] = [b + [a] for a, b in enumerate(item_store_ppl[last["store_id"]][:n])]
        except:
            ans3[uid] = []
        try:
            ans4[uid] = [b + [a] for a, b in enumerate(item_cate_ppl[last["cate_id"]][:n])]
        except:
            ans4[uid] = []

        aim = last["cate_id"]
        if (pd.notna(aim)):
            cand = cate2item.loc[aim].copy()
            if (cand.ndim == 2):
                cand["score"] = np.abs(cand["item_price"] - last["item_price"])
                cand = cand.sort_values("score")
                ans5[uid] = [[b, a] for a, b in enumerate(cand["item_id"].iloc[:n])]
            elif (cand.ndim == 1):
                ans5[uid] = [[cand["item_id"], 0]]
            else:
                raise ValueError(uid)

        aim = last["store_id"]
        if (pd.notna(aim)):
            cand = store2item.loc[aim].copy()
            if (cand.ndim == 2):
                cand["score"] = np.abs(cand["item_price"] - last["item_price"])
                cand = cand.sort_values("score")
                ans6[uid] = [[b, a] for a, b in enumerate(cand["item_id"].iloc[:n])]
            elif (cand.ndim == 1):
                ans6[uid] = [[cand["item_id"], 0]]
            else:
                raise ValueError(uid)

        try:
            aim = user_most_cate[uid]
        except KeyError:
            aim=np.NaN
        if (pd.notna(aim)):
            cand = cate2item.loc[aim].copy()
            if (cand.ndim == 2):
                cand["score"] = np.abs(cand["item_price"] - last["item_price"])
                cand = cand.sort_values("score")
                ans9[uid] = [[b, a] for a, b in enumerate(cand["item_id"].iloc[:n])]
            elif (cand.ndim == 1):
                ans9[uid] = [[cand["item_id"], 0]]
            else:
                raise ValueError(uid)

        #上面全是和最后一次查看相关的
        try:#下面全是和最后一次购买记录相关的
            last = user_last_buy.loc[uid]
        except KeyError:
            n_not_nuy += 1
            continue
        aim = last["cate_id"]
        if (pd.notna(aim)):
            cand = cate2item.loc[aim].copy()
            if (cand.ndim == 2):
                cand["score"] = np.abs(cand["item_price"] - last["item_price"])
                cand = cand.sort_values("score")
                ans7[uid] = [[b, a] for a, b in enumerate(cand["item_id"].iloc[:n])]
            elif (cand.ndim == 1):
                ans7[uid] = [[cand["item_id"], 0]]
            else:
                raise ValueError(uid)
        aim = last["store_id"]
        if (pd.notna(aim)):
            cand = store2item.loc[aim].copy()
            if (cand.ndim == 2):
                cand["score"] = np.abs(cand["item_price"] - last["item_price"])
                cand = cand.sort_values("score")
                ans8[uid] = [[b, a] for a, b in enumerate(cand["item_id"].iloc[:n])]
            elif (cand.ndim == 1):
                ans8[uid] = [[cand["item_id"], 0]]
            else:
                raise ValueError(uid)
        try:
            aim = user_most_store[uid]
            if (pd.notna(aim)):
                cand = store2item.loc[aim].copy()
                if (cand.ndim == 2):
                    cand["score"] = np.abs(cand["item_price"] - last["item_price"])
                    cand = cand.sort_values("score")
                    ans10[uid] = [[b, a] for a, b in enumerate(cand["item_id"].iloc[:n])]
                elif (cand.ndim == 1):
                    ans10[uid] = [[cand["item_id"], 0]]
                else:
                    raise ValueError(uid)
        except KeyError:
            pass

    print(n_not_nuy, "not buy!")
    if (offline):
        for anss in [ans0, ans1, ans2, ans3, ans4, ans5, ans6, ans7, ans8,ans9,ans10]:
            print(fn, "  ", score(anss, label)[:-2],sum(get_len_each_ans(anss)))
        save(ans0, f"ans0_{fn}.pkl")
        save(ans1, f"ans1_{fn}.pkl")
        save(ans2, f"ans2_{fn}.pkl")
        save(ans3, f"ans3_{fn}.pkl")
        save(ans4, f"ans4_{fn}.pkl")
        save(ans5, f"ans5_{fn}.pkl")
        save(ans6, f"ans6_{fn}.pkl")
        save(ans7, f"ans7_{fn}.pkl")
        save(ans8, f"ans8_{fn}.pkl")
        save(ans9, f"ans9_{fn}.pkl")
        save(ans10, f"ans10_{fn}.pkl")
    else:
        save(ans0, f"ans0_{fn}_test.pkl")
        save(ans1, f"ans1_{fn}_test.pkl")
        save(ans2, f"ans2_{fn}_test.pkl")
        save(ans3, f"ans3_{fn}_test.pkl")
        save(ans4, f"ans4_{fn}_test.pkl")
        save(ans5, f"ans5_{fn}_test.pkl")
        save(ans6, f"ans6_{fn}_test.pkl")
        save(ans7, f"ans7_{fn}_test.pkl")
        save(ans8, f"ans8_{fn}_test.pkl")
        save(ans9, f"ans9_{fn}_test.pkl")
        save(ans10, f"ans10_{fn}_test.pkl")


# 对每个用户推荐完后应该删除他buy_flag为0的商品，然后再补全

def pipe_others():
    others(fn="yy", n=50, offline=True)
    others(fn="yy", n=50, offline=False)

    others(fn="zz", n=50, offline=True)
    others(fn="zz", n=50, offline=False)

if __name__ == '__main__':
    print("fine")