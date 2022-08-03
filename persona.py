import pandas as pd

# Görev 1:
df = pd.read_csv("datasets/persona.csv")
print("########################### Shape #############################")
print(df.shape)
print("########################### Types #############################")
print(df.dtypes)
print("########################### Col Names #############################")
print(df.columns)
print("########################### NA #############################")
print(df.isnull().sum())
print("########################### Head #############################")
print(df.head())
print("########################### Tail #############################")
print(df.tail())

df["SOURCE"].nunique()
df["SOURCE"].value_counts()
# Soru 3:
df["PRICE"].nunique()
# Soru 4:
df["PRICE"].value_counts()
# Soru 5:
df["COUNTRY"].value_counts()
# Soru 6:
df.groupby("COUNTRY")["PRICE"].sum()
# Soru 7:
df["SOURCE"].value_counts()
# Soru 8:
df.groupby("COUNTRY")["PRICE"].mean()
# sORU 9:
df.groupby("SOURCE")["PRICE"].mean()
# Soru 10:
df.groupby(["SOURCE", "COUNTRY"]).mean()
################################################################
# Görev 2:
df.groupby(["COUNTRY", "SEX", "AGE", "SOURCE"])["PRICE"].mean().head(7)
# Görev 3:
groups = df.groupby(["COUNTRY", "SOURCE", "SEX", "AGE"]).agg({"PRICE": "mean"})
agg_df = groups.sort_values("PRICE", ascending=False)
# Görev 4:
agg_df.reset_index(inplace=True)
# Görev 5:
agg_df['AGE_CAT'] = pd.cut(x=agg_df['AGE'],
                           bins=[0, 18, 23, 30, 40, 70],
                           labels=['0_18', '19_23', '24_30', '31_40', '41_70'])
agg_df.head()

# Görev 6 :
agg_df["CUSTOMERS_LEVEL_BASED"] = [row[0].upper()+"_" + row[1].upper() + "_" +row[2].upper()+"_" + row[5].upper() for row in agg_df.values]
agg_df.head(50)
agg_df = agg_df.groupby(["CUSTOMERS_LEVEL_BASED"])[["PRICE"]].agg("mean")
agg_df

# Görev 7:
agg_df["SEGMENTS"] = pd.qcut(agg_df["PRICE"], 4, labels=["D", "C", "B", "A"])
agg_df.reset_index(inplace=True)
agg_df.groupby("SEGMENTS").agg({"PRICE": ["mean", "max", "sum"]})


# Görev 8 :
new_user = "TUR_ANDROID_FEMALE_31_40"
agg_df[agg_df["CUSTOMERS_LEVEL_BASED"] == new_user]
new_user1 = "FRA_IOS_FEMALE_31_40"
agg_df[agg_df["CUSTOMERS_LEVEL_BASED"] == new_user1]