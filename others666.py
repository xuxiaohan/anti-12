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


    ans11 = {}#在最后一条记录对应的商店中该种类的价格升序推荐
    ans12 = {}#在最后一条记录对应的商店中该种类的销量降序推荐
    ans13 = {}  # 在最后一条购买记录对应的商店中该种类的价格升序推荐
    ans14 = {}  # 在最后一条购买记录对应的商店中该种类的销量降序推荐


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
    item_attr=item_attr.sort_values("item_price")
    #任一确定商店和种类后的价格升序排列
    cate_store_cheap= item_attr.groupby(["cate_id","store_id"]).apply(lambda x: x["item_id"].values[:n].tolist())
    #任一确定商店和种类后的销量降序排列
    cate_store_ppl = df.groupby(["item_id","cate_id", "store_id"]).size()
    cate_store_ppl=cate_store_ppl/df.groupby(["cate_id", "store_id"]).size()
    cate_store_ppl=cate_store_ppl.sort_values(ascending=False).reset_index().groupby(["cate_id", "store_id"]).apply(lambda x: x[["item_id", 0]].values[:n].tolist())
    # # 任一确定商店和种类和时间某天，的销量降序排列
    # cate_store_day_ppl = df.groupby(["cate_id", "store_id","day"]).size().sort_values(ascending=False).apply(lambda x: x[["item_id", 0]].values[:n].tolist())

    cate2item = item_attr[["cate_id", "item_id", "item_price"]].drop_duplicates(["cate_id", "item_id"]).set_index(
        "cate_id")
    store2item = item_attr[["store_id", "item_id", "item_price"]].drop_duplicates(["store_id", "item_id"]).set_index(
        "store_id")
    n_not_nuy = 0


    for ii, uid in enumerate(user):
        if (ii % 5000 == 0):
            print(f"{ii}  ({ii / user.shape[0]})")
        last = user_last.loc[uid]#和最后一次行为相关的（不区分购买还是查看）


        try:#商店种类的销售量降序排列
            ans11[uid] = [b + [a] for a, b in enumerate(cate_store_ppl[last["cate_id"],last["store_id"]])]
        except:
            ans11[uid] = []

        try:#商店种类的价格升序排列
            ans12[uid] = [[b,0,a] for a, b in enumerate(cate_store_cheap[last["cate_id"],last["store_id"]])]
        except:
            ans12[uid] = []


        #上面全是和最后一次查看相关的
        try:#下面全是和最后一次购买记录相关的
            last = user_last_buy.loc[uid]
        except KeyError:
            n_not_nuy += 1
            continue
        try:#商店种类的销售量降序排列
            ans13[uid] = [b + [a] for a, b in enumerate(cate_store_ppl[last["cate_id"],last["store_id"]])]
        except:
            ans13[uid] = []

        try:#商店种类的价格升序排列
            ans14[uid] = [[b,0,a] for a, b in enumerate(cate_store_cheap[last["cate_id"],last["store_id"]])]
        except:
            ans14[uid] = []



    print(n_not_nuy, "not buy!")
    if (offline):
        for anss in [ans11,ans12,ans13,ans14]:
            print(fn, "  ", score(anss, label)[:-2],sum(get_len_each_ans(anss)))
        save(ans11, f"ans11_{fn}.pkl")
        save(ans12, f"ans12_{fn}.pkl")
        save(ans13, f"ans13_{fn}.pkl")
        save(ans14, f"ans14_{fn}.pkl")
    else:
        save(ans11, f"ans11_{fn}_test.pkl")
        save(ans12, f"ans12_{fn}_test.pkl")
        save(ans13, f"ans13_{fn}_test.pkl")
        save(ans14, f"ans14_{fn}_test.pkl")


# 对每个用户推荐完后应该删除他buy_flag为0的商品，然后再补全

def pipe_others666():
    others(fn="yy", n=50, offline=True)
    others(fn="yy", n=50, offline=False)

    others(fn="zz", n=50, offline=True)
    others(fn="zz", n=50, offline=False)

if __name__ == '__main__':
    print("fine")

