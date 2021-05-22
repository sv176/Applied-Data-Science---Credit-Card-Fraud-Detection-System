import forex_python.converter as fx
from datetime import datetime
import numpy as np
import pandas as pd

def save_data(df:pd.DataFrame,file_path):
    df.to_csv(file_path)

def anonymise_to_cats(series:pd.Series) -> pd.Series:
    return series.astype("category").cat.codes

def dob_to_age(df) -> pd.Series:
    return (df["trans_date_trans_time"]-df["dob"])//np.timedelta64(1,"Y")

def k_anon_jobs(df,k=2) -> pd.Series:
    """
    k anonymise jobs. Note -1 means a None value (Used for jobs which less than k people do)
    """
    df["job_category"]=anonymise_to_cats(df["job"].str.split(",").str[0].str.strip())
    group_by=df[["person_id","job_category"]].groupby(["job_category"])
    rare_job_cats=np.where(group_by["person_id"].nunique()<k)[0].tolist()

    return df["job_category"].transform(lambda x:-1 if (x in rare_job_cats) else x)

# k-anonymous clustering
def k_anon_clustering(series,k=2) -> pd.Series:
    series=series.copy()
    unique_vals=series.unique().tolist()
    unique_vals=pd.Series(unique_vals,unique_vals)
    unique_vals.sort_values(ascending=False,inplace=True)

    bin_id=0
    i=0
    for i in range(0,unique_vals.size-k+1,k):
        unique_vals[i:i+k]=bin_id
        if (i+2*k>unique_vals.size): unique_vals[i+k:]=bin_id
        bin_id+=1

    return series.transform(lambda x:unique_vals[x])

    return series

def anonymise_data(df, printing=True) -> pd.DataFrame:
    # prepare data
    if (printing): print("Data ",end="",flush=True)
    df["trans_date_trans_time"]=pd.to_datetime(df["trans_date_trans_time"],format="%Y-%m-%d %H:%M:%S")
    df["dob"]=pd.to_datetime(df["dob"],format="%Y-%m-%d")
    if (printing): print("PREPARED")

    # clean data
    clean_df=pd.DataFrame()
    clean_df["is_fraud"]=df["is_fraud"]

    if (printing): print("Time ",end="",flush=True)
    clean_df["unix_time"]=df["unix_time"].copy()
    if (printing): print("DONE")

    if (printing): print("Amount ",end="",flush=True)
    clean_df["amt"]=df["amt"].copy()
    if (printing): print("DONE")

    if (printing): print("Credit Card ",end="",flush=True)
    clean_df["cc_id"]=anonymise_to_cats(df["cc_num"])
    if (printing): print("DONE")

    if (printing): print("Person  ",end="",flush=True)
    clean_df["person_id"]=anonymise_to_cats(df["first"]+"_"+df["last"]+"_"+df["job"]+"_"+df["dob"].apply(lambda x: x.strftime('%Y-%m-%d')))
    if (printing): print("DONE")

    if (printing): print("Gender ",end="",flush=True)
    clean_df["gender_id"]=anonymise_to_cats(df["gender"])
    if (printing): print("DONE")

    if (printing): print("Job ",end="",flush=True)
    clean_df["job_category"]=k_anon_jobs(pd.concat([df["job"],clean_df["person_id"]],axis=1),k=5)
    if (printing): print("DONE")

    if (printing): print("Age ",end="",flush=True)
    clean_df["age"]=dob_to_age(df[["dob","trans_date_trans_time"]])
    if (printing): print("DONE")

    if (printing): print("City Pop ",end="",flush=True)
    clean_df["city_pop_cluster_id"]=k_anon_clustering(df["city_pop"],k=10)
    if (printing): print("DONE")

    if (printing): print("Merchant ",end="",flush=True)
    clean_df["merchant_id"]=anonymise_to_cats(df["merchant"])
    clean_df["merchant_category"]=anonymise_to_cats(df["category"])
    if (printing): print("DONE")

    return clean_df

print("Data ",end="",flush=True)
training_data=pd.read_csv("data/synthetic_train.csv", index_col=0)
test_data=pd.read_csv("data/synthetic_test.csv", index_col=0)
full_data=pd.concat([training_data, test_data])
print("LOADED")

clean_df=anonymise_data(full_data)
save_data(clean_df,"data/cleaned_synthetic_data.csv")
