from function import *

class fun:
    def __init__(self):
        self.offset=0
        self.rating_scale=(0,1)

def get_cluster_label(svd,pre,user=True):
    ans={}
    if(user):
        for i,e in enumerate(pre):
            ans[svd.trainset.to_raw_uid(i)]=e
    else:
        for i,e in enumerate(pre):
            ans[svd.trainset.to_raw_iid(i)]=e
    return ans


def baseline(file_name,need_rate=False,result_name="baseline.pkl",n=30):
    # test = pd.read_csv("offline/"+name+"_train.csv")  # 线下xx
    test = pd.read_csv(file_name)  # 线下xx
    test["log_time_day"] = split_time(test["log_time"])["day"]
    all_test_user = test["buyer_admin_id"].unique()
    user_country = test[["buyer_admin_id", "country_id"]].drop_duplicates(["buyer_admin_id", "country_id"]).set_index(
        "buyer_admin_id")["country_id"]

    test = test[test["buy_flag"] == 1]
    test = test.sort_values(["buyer_admin_id", "irank"])
    test = test.drop_duplicates(["buyer_admin_id", "item_id"])
    group = dict(iter(test.groupby("buyer_admin_id")))
    ans = {}
    for user_id in all_test_user:
        if (group.get(user_id) is not None):
            if (need_rate):
                t = []
                for ii,(a, b) in enumerate(group[user_id][["item_id", "log_time_day"]].values.tolist()[:n]):
                    t.append([a, b, ii])
                ans[user_id] = t
            else:
                ans[user_id] = group[user_id]["item_id"].values.tolist()[:n]
        else:
            ans[user_id] = []
    return ans

def transform(A):
    ans=[]
    for i,a in enumerate(A):
        for j,b in enumerate(a):
            if(b!=0):
                ans.append((i,j,b))
    return ans

class Data:
    def __init__(self,A):
        self.A=A
        self.n,self.m=A.shape
        self.posi,self.posj=0,0

    def __iter__(self):
        return self

    def __next__(self):
        posi,posj=self.posi,self.posj
        if (posi >= self.n or posj >= self.m):
            raise StopIteration
        ans=posi,posj,self.A[posi,posj]
        if(posj+1==self.m):
            posj=0
            posi=posi+1
        else:
            posj=posj+1
        self.posi, self.posj = posi,posj
        return ans

    def itertuples(self,*k,**k1):
        return self

def late_processing(ans,notans,n=30,drop=True,not_set=True):
    if(not_set):
        if(n!=-1):
            for e in notans:
                notans[e]=set(notans[e][:n])
        else:
            for e in notans:
                notans[e]=set(notans[e])

    if(drop):
        for uid in ans:
            if(notans.get(uid) is not None):
                aa=[]
                for e in  ans[uid]:
                    if(e not in notans[uid]):
                        aa.append(e)
                ans[uid]=aa
    else:
        for uid in ans:
            if(notans.get(uid) is not None):
                aa=[]
                bb=[]
                for e in  ans[uid]:
                    if(e not in notans[uid]):
                        aa.append(e)
                    else:
                        bb.append(e)
                ans[uid]=aa+bb
    return ans

def late_processing_1(ans,ans1,n=3):
    for uid in ans:
        if(ans1.get(uid) is not None):
            aa=[]
            bb=[]
            for e in ans[uid]:
                if(e not in ans1[uid][:1]):
                    aa.append(e)
                else:
                    bb.append(e)
            ans[uid]=ans1[uid][:1]+aa
    return ans