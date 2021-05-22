import forex_python.converter as fx
import numpy as np
import pandas as pd

from datetime import datetime
from math import ceil

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

def standardise_time(series) -> pd.Series:
    min_time = datetime.utcfromtimestamp(series.min())
    min_day = min_time.replace(second=0, minute=0, hour=0)
    return ((series-min_day.timestamp())).astype(int)

def convert_currency(amount : float, date : datetime, cur_currency : str, tar_currency) -> float:
    """
    Determine the value of an amount of one currency in another currency at a specified point in time

    PARAMETERS
    amount (float) - amount of current currency
    date (datetime) - date of exchange rate to use
    cur_currency (str) - three character code for current currency
    tar_currency (str) - three character code for target currency

    RETURNS
    float - amount of target currency
    """
    exchange_rate = fx.get_rate(cur_currency, tar_currency, date)
    return round(amount * exchange_rate, 2)

def prepare_amount(df, cur_label, cur_currency="USD", tar_currency="GBP") -> pd.Series:
    """
    Convert amounts in a dataframe between currencies, using the exchange rate at the start of the date on which transaction occurred
    NOTE - conversion rate taken at start of day for speed.

    PARAMETERS
    df (pd.Dataframe) - dataframe of transactions with at least ["data",cur_label] columns
    cur_label (str) - name of column which contains amounts to convert
    """
    df_local = df.copy(deep=True)
    df_local["date"] = pd.to_datetime(df["unix_time"], unit="s").dt.date

    # determine the exchange rate for each day
    exchange_rates = pd.DataFrame()
    exchange_rates["date"] = pd.to_datetime(df_local["date"].unique(), format="%Y-%m-%d")
    exchange_rates["rate"] = exchange_rates.apply(lambda x:convert_currency(1, x["date"], cur_currency, tar_currency), axis=1)

    # merge dataframes
    exchange_rates["date"] = exchange_rates["date"].dt.date
    df_merged = df_local[["date", "amt"]].reset_index().merge(exchange_rates[["date", "rate"]], on="date", how="left").set_index('index')

    # calculated exchanged amounts
    tar_label = "amount_{}".format(tar_currency)
    df_merged[tar_label] = df_merged.apply(lambda x:round(x["amt"] * x["rate"], 2), axis=1)

    return df_merged[tar_label]

# Total number of transactions performed by each entity (person or merchant) in the dataset
def transactions_per_entity(ids) -> pd.Series:
    """ids - series of either `person_id` or `merchant_id`"""
    pp_trans = ids.value_counts()
    return ids.apply(lambda x: pp_trans[x])

# time since last transaction (merchant and customer)
def time_since_last_transaction(id_col, df, clean_df) -> pd.Series:
    """
    id_col (str) - name of column which contain ids to group by
    df (pd.DataFrame) - dataframe containing `unix_time` and `id_col`
       NOTE -1 = first transaction on record
    """
    times = df[id_col].copy(deep=True)

    clean_df["time_since_last_transaction_person"] = -1
    for id_code in df[id_col].unique():
        trans_times = df[df[id_col] == id_code]["unix_time"]
        times.loc[df[id_col] == id_code] = trans_times.diff()

    return times.replace(np.nan, -1).astype(int)

# mean/min/max amt per merchant/customer
# NOTE this is USD val so maybe change
def entity_amount_statistic(id_col, df, agg_calc) -> pd.Series:

    group_by = df[[id_col, "amt"]].groupby([id_col])
    vals = group_by["amt"].agg(agg_calc)
    return df[id_col].transform(lambda x:vals[x])

# mean/min/max amt per merchant/customer
# NOTE this is USD val so maybe change
def entity_amount_statistic_by_day(id_col, df, agg_calc) -> pd.Series:

    df_copy=df[[id_col, "amt", "unix_time"]].copy()
    df_copy["date"] = pd.to_datetime(df_copy["unix_time"], unit="s").dt.date

    group_by = df_copy.groupby([id_col, "date"], as_index=False)
    vals = group_by["amt"].agg(agg_calc)

    df_copy = df_copy.reset_index().merge(vals,on=[id_col, "date"], how='inner', suffixes=("_orig", "_group_by")).set_index("index")

    return df_copy["amt_group_by"].values

# Number of Transactions done on same say by Entity
def transaction_on_date(id_col, df) -> pd.Series:
    df_copy = df[[id_col, "unix_time"]].copy()
    df_copy["date"] = pd.to_datetime(df_copy["unix_time"], unit="s").dt.date

    group_by = df_copy[[id_col, "date"]].groupby(["person_id", "date"], as_index=False)
    counts = group_by.size()

    df_copy = df_copy.reset_index().merge(counts, on=[id_col, "date"], how='inner', suffixes=("_orig", "_group_by")).set_index("index")

    return df_copy["size"].values

def extend_meta(clean_df : pd.DataFrame) -> pd.DataFrame:
    meta_data = clean_df.copy()

    print("PREPARING TIME META DATA")
    meta_data["seconds_from_start"] = standardise_time(clean_df["unix_time"])
    meta_data["hour_of_day"] = pd.to_datetime(clean_df["unix_time"], unit="s").dt.hour

    meta_data["time_since_last_transaction_person"] = time_since_last_transaction("person_id", clean_df[["person_id", "unix_time"]], clean_df)
    meta_data["time_since_last_transaction_merchant"] = time_since_last_transaction("merchant_id", clean_df[["merchant_id", "unix_time"]], clean_df)

    meta_data["transactions_on_day_person"] = transaction_on_date("person_id", clean_df[["person_id", "unix_time"]])

    print("PREPARING AMOUNT META DATA")
    meta_data["amount_USD"] = clean_df["amt"].copy()
    meta_data["amount_GBP"] = prepare_amount(clean_df[["unix_time", "amt"]], "amt", "USD", "GBP")

    meta_data["transaction_by_person"] = transactions_per_entity(clean_df["person_id"])
    meta_data["transaction_by_merchant"] = transactions_per_entity(clean_df["merchant_id"])

    meta_data["mean_amt_person"] = entity_amount_statistic("person_id", clean_df[["person_id", "amt"]], "mean")
    meta_data["max_amt_merchant"] = entity_amount_statistic("merchant_id", clean_df[["merchant_id", "amt"]], "max")

    meta_data["mean_amt_merchant_on_day"] = entity_amount_statistic_by_day("merchant_id", clean_df[["merchant_id", "amt", "unix_time"]], "mean")
    meta_data["max_amt_person_on_day"] = entity_amount_statistic_by_day("person_id", clean_df[["person_id", "amt", "unix_time"]], "mean")

    return meta_data

sets = ["train", "test"]

for set in sets:
    print(f"EXTENDING SET: {set.upper()}")
    # Load the data
    print("Data ", end="", flush=True)
    loaded_data = pd.read_csv(f"data/synthetic_{set}.csv", index_col=0)
    print("LOADED")


    print("DATA SIZE:", len(loaded_data))
    clean_df = anonymise_data(loaded_data)
    meta_data = extend_meta(clean_df)
    save_data(meta_data, f"data/meta_features_{set}.csv")
