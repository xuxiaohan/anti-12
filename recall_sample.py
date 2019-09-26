from sklearn.model_selection import ParameterSampler,ParameterGrid
from cand_func import *
import xgboost as xgb


def recall_sample(fn="zz",offline=True,n=100):
    label=None
    if(offline):
        ans0=load(f"{fn}_baseline.pkl")
        ans1=load(f"from_xx_to_{fn}.pkl")
        ans2=load(f"from_yy_to_{fn}.pkl")
        ans3 = load(f"from_zz_to_{fn}.pkl")
        ans4=load(f"ans0_{fn}.pkl")
        ans5=load(f"ans1_{fn}.pkl")
        ans6=load(f"ans2_{fn}.pkl")
        ans7=load(f"ans3_{fn}.pkl")
        ans8=load(f"ans4_{fn}.pkl")
        ans9 = load(f"ans5_{fn}.pkl")
        ans10 = load(f"ans6_{fn}.pkl")
        ans11 = load(f"ans7_{fn}.pkl")
        ans12 = load(f"ans8_{fn}.pkl")
        ans13 = load(f"ans9_{fn}.pkl")
        ans14 = load(f"ans10_{fn}.pkl")
        ans15 = load(f"ans11_{fn}.pkl")
        ans16 = load(f"ans12_{fn}.pkl")
        ans17 = load(f"ans13_{fn}.pkl")
        ans18 = load(f"ans14_{fn}.pkl")
        all_ans=[ans0,ans1,ans2,ans3,ans4,ans5,ans6,ans7,ans8,ans9,ans10,ans11,ans12,ans13,ans14,ans15, ans16,ans17,ans18]

        sample=get_sample(all_ans,has_rate=[True for _ in range(len(all_ans))],n=n)
        sample=sample.drop(columns=["score_from_0_0"])

        print("has label file")
        label=pd.read_csv(f"{fn}_clean_label.csv")
        label=read_map_from_df(label[["buyer_admin_id","item_id"]])
        lb=pd.Series(label)
        lb=lb.reset_index()
        lb.columns=["user_id","item_id"]
        lb["label"]=1
        sample=pd.merge(sample,lb,how="left",on=["user_id","item_id"])
        sample["label"]=sample["label"].fillna(0).astype("int")
        for ans,name in zip(all_ans,["ans"+str(i) for i in range(len(all_ans))]):
            print(name,score(ans,label)[:-2],sum(get_len_each_ans(ans)))
    else:
        ans0 = load(f"{fn}_baseline_test.pkl")#记得后缀test
        ans1=load(f"from_xx_to_{fn}_test.pkl")
        ans2 = load(f"from_yy_to_{fn}_test.pkl")
        ans3 = load(f"from_zz_to_{fn}_test.pkl")
        ans4 = load(f"ans0_{fn}_test.pkl")
        ans5 = load(f"ans1_{fn}_test.pkl")
        ans6 = load(f"ans2_{fn}_test.pkl")
        ans7 = load(f"ans3_{fn}_test.pkl")
        ans8 = load(f"ans4_{fn}_test.pkl")
        ans9 = load(f"ans5_{fn}_test.pkl")
        ans10 = load(f"ans6_{fn}_test.pkl")
        ans11 = load(f"ans7_{fn}_test.pkl")
        ans12 = load(f"ans8_{fn}_test.pkl")
        ans13 = load(f"ans9_{fn}_test.pkl")
        ans14 = load(f"ans10_{fn}_test.pkl")
        ans15 = load(f"ans11_{fn}_test.pkl")
        ans16 = load(f"ans12_{fn}_test.pkl")
        ans17 = load(f"ans13_{fn}_test.pkl")
        ans18 = load(f"ans14_{fn}_test.pkl")
        all_ans = [ans0, ans1, ans2, ans3, ans4, ans5, ans6, ans7, ans8, ans9, ans10, ans11, ans12,ans13,ans14,ans15, ans16,ans17,ans18]

        sample = get_sample(all_ans, has_rate=[True for _ in range(len(all_ans))],n=n)
        sample = sample.drop(columns=["score_from_0_0"])
    return sample,label,len(all_ans)

def main(fn="zz",offline=True,n=100):
    """

    :param fn: 要哪个国家
    :param offline: True表示只要线下，False表示线上和线下都要
    :return: 返回sample和map类型的label   （sample里也有label，但是只包含了被召回的样本部分的）
    """

    sample,label,n_ans=recall_sample(fn,True,n=n)
    df_list = [sample]
    if(not offline):
        test,_,n_ans=recall_sample(fn,False,n=n)
        df_list.append(test)

    print("n_ans",n_ans)
    # for df in df_list:
    #     for ii in range(n_ans):
    #         df[f"score_from_{ii}"] = 1 / (df[f"score_from_{ii}_1"] + 1)

    print(f"{fn} train user num is :",sample["user_id"].nunique())
    print("befor reduce:",sample.memory_usage().sum()/1000**2,"MB")
    sample=reducemem(sample)
    print("after reduce:", sample.memory_usage().sum() / 1000 ** 2, "MB")
    # save(sample,f"{fn}_sample.pkl")
    sample.to_csv(f"{fn}_sample.csv",index=False)
    if (not offline):
        print(f"{fn} test user num is :", test["user_id"].nunique())
        print("befor reduce:", test.memory_usage().sum() / 1000 ** 2, "MB")
        test=reducemem(test)
        print("after reduce:", test.memory_usage().sum() / 1000 ** 2, "MB")
        # save(test, f"{fn}_sample_test.pkl")
        test.to_csv(f"{fn}_sample_test.csv",index=False)
    return sample,label

#if __name__ == '__main__':
def pipe_recall_sample(n=100):
    """
    分为测试样本的召回和训练样本的召回，两者都要修改recall_sample里的ans和对应的get_sample的参数
    """
    #offline为false表示线下和线上都要，true表示只要线下
    print("compute for yy")
    res_yy=main("yy",offline=False,n=n)
    print("compute for zz")
    res_zz=main("zz",offline=False,n=n)

if __name__ == '__main__':
    pipe_recall_sample(n=100)
