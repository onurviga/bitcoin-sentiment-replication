import pandas as pd, numpy as np, json, warnings; warnings.filterwarnings("ignore")
from statsmodels.tsa.api import VAR
d=pd.read_csv("dataset_final.csv",parse_dates=["date"]).set_index("date")
out={}
for x in ["dlgt","dlvol","dfg"]:
    data=d[["ret",x]].dropna()
    dow=pd.get_dummies(data.index.dayofweek,prefix="d",drop_first=True).astype(float); dow.index=data.index
    v=VAR(data,exog=dow); sel=v.select_order(14)
    row={"sel":{"AIC":int(sel.aic),"BIC":int(sel.bic)}}
    for lab,lag in [("AIC",max(1,int(sel.aic))),("BIC",max(1,int(sel.bic))),("fix14",14)]:
        r=v.fit(lag); t1=r.test_causality("ret",[x],kind="f"); t2=r.test_causality(x,["ret"],kind="f")
        row[lab]={"lag":lag,"x_to_ret":[round(t1.test_statistic,3),round(t1.pvalue,4)],"ret_to_x":[round(t2.test_statistic,3),round(t2.pvalue,4)],
                  "portmanteau_p":round(r.test_whiteness(nlags=lag+10,adjusted=True).pvalue,4),"stable":bool(r.is_stable())}
    out[x]=row
json.dump(out,open("dow_results.json","w"),indent=1); print(json.dumps(out,indent=1))
