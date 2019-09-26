from cand_func import *
import pandas as pd

def pipe_baseline():
    for fn in ["yy","zz"]:
        print(fn, ": ")
        ans=baseline(f"{fn}_clean.csv",True)
        label=pd.read_csv(f"{fn}_clean_label.csv")
        label=read_map_from_df(label[["buyer_admin_id","item_id"]])
        save(ans,f"{fn}_baseline.pkl")
        print(score(ans,label)[:-2])
        ans = baseline(f"test_{fn}.csv",True)
        save(ans, f"{fn}_baseline_test.pkl")


# res={}
# for fn in ["yy","zz"]:
#     sc=[]
#     print(fn, ": ")
#     for rand in range(5):
#         ans=baseline(f"{rand}_{fn}_val.csv")
#         label=pd.read_csv(f"offline/{rand}_{fn}_val_label.csv")
#         label=read_map_from_df(label[["buyer_admin_id","item_id"]])
#         sc.append(score(ans,label)[:-2])
#         print(rand,sc[-1])
#     res[fn]=sc
