import numpy as np
import seaborn as sns
import pandas as pd
import matplotlib.pyplot as plt
import missingno as msno
from datetime import date
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split
from sklearn.neighbors import LocalOutlierFactor
from sklearn.preprocessing import MinMaxScaler, LabelEncoder, StandardScaler, RobustScaler

pd.set_option("display.width", 500)
pd.set_option("display.max_columns", None)


def load_titanic():
    data = pd.read_csv("ozellik_muhendisligi/datasets/titanic.csv")
    return data


df = load_titanic()
df.shape
df.head()

df.columns = [col.upper() for col in df.columns]

df["AGE_CUT"] = pd.cut(x=df["AGE"], bins=[0,18,56,max(df["AGE"])], labels=["Young","Mature","Senior"])
##############################################################
# 1 - Değişken Mühendisliği Feature Engineering
##############################################################
# CABIN BOOL
df["NEW_CABIN_BOOL"] = df["CABIN"].notnull().astype("int")
# NAME COUNT
df["NEW_NAME_COUNT"] = df["NAME"].str.len()
# NAME WORD COUNT
df["NEW_WORD_COUNT"] = df["NAME"].apply(lambda x: len(str(x).split(" ")))
# NAME DR
df["NEW_NAME_DR"] = df["NAME"].apply(lambda x: len([x for x in x.split() if x.startswith("Dr")]))
# NAME TITLE
df["NEW_TITLE"] = df["NAME"].str.extract(" ([A-Za-z]+)\.", expand=False)
#
df["NEW_AGE_PCLASS"] = df["AGE"] * df["PCLASS"]

df["NEW_FAMILY_SIZE"] = df["SIBSP"] + df["PARCH"] + 1

df.loc[((df["SIBSP"] + df["PARCH"]) > 0), "NEW_IS_ALONE"] = "No"
df.loc[((df["SIBSP"] + df["PARCH"]) == 0), "NEW_IS_ALONE"] = "Yes"

df.loc[(df["SEX"] == "male") & (df["AGE"] <= 21), "New_Sex_Cat"] = "young_male"
df.loc[(df["SEX"] == "male") & ((df["AGE"] > 21) & df["AGE"] <= 50), "New_Sex_Cat"] = "mature_male"
df.loc[(df["SEX"] == "male") & (df["AGE"] > 50), "New_Sex_Cat"] = "senior_male"
df.loc[(df["SEX"] == "female") & (df["AGE"] <= 21), "New_Sex_Cat"] = "young_female"
df.loc[(df["SEX"] == "female") & ((df["AGE"] > 21) & df["AGE"] <= 50), "New_Sex_Cat"] = "mature_female"
df.loc[(df["SEX"] == "female") & (df["AGE"] > 50), "New_Sex_Cat"] = "senior_female"

df.loc[(df["AGE"] < 18), "NEW_AGE_CAT"] = "young"
df.loc[(df["AGE"] >= 18) & (df["AGE"] < 56), "NEW_AGE_CAT"] = "mature"
df.loc[(df["AGE"] >= 56), "NEW_AGE_CAT"] = "senior"

df.head()
df.info()


def grab_col_names(dataFrame, cat_th=10, car_th=30):
    """
    Veri setindeki kategorik, numerik ve kategorik fakat kardinal değişkenlerin isimlerini verir
    Parameters
    ----------
    dataFrame: dataframe
        değişken isimleri alınmak istenen dataframe
    cat_th: int, float
        numerik fakat kategorik olan değişkenler için sınıf eşik değeri
    car_th: int, float
        kategorik fakat kardinal olan değişkenler için sınıf eşik değeri

    Returns
    -------
    cat_cols: list
        Kategorik değişken listesi
    num_cols: list
         Numerik değişken listesi
    cat_but_car: list
        Kategorik gibi gözüken kardinal değişken listesi

    Notes
    -------
        cat_cols + num_cols + cat_but_car = toplam değişken sayısı
        num_but_cat cat_cols'un içinde

    """
    cat_cols = [col for col in dataFrame.columns if
                str(dataFrame[col].dtypes) in ["object", "category", "bool"]]

    num_but_cat = [col for col in dataFrame.columns if
                   dataFrame[col].nunique() < cat_th and dataFrame[col].dtypes in ["int64", "float64", "int32",
                                                                                   "uint8"]]

    cat_but_car = [col for col in dataFrame.columns if
                   dataFrame[col].nunique() > car_th and str(dataFrame[col].dtypes) in ["object", "category"]]

    cat_cols = cat_cols + num_but_cat
    cat_cols = [col for col in cat_cols if col not in cat_but_car]

    num_cols = [col for col in dataFrame.columns if dataFrame[col].dtypes in ["int64", "float64"]]

    num_cols = [col for col in num_cols if col not in cat_cols]

    print(f"Observations: {dataFrame.shape[0]}")
    print(f"Variables: {dataFrame.shape[1]}")
    print(f"cat_cols: {len(cat_cols)}")
    print(f"num_cols: {len(num_cols)}")
    print(f"cat_but_car: {len(cat_but_car)}")
    print(f"num_but_car: {len(num_but_cat)}")

    return cat_cols, num_cols, cat_but_car


cat_cols, num_cols, cat_but_car = grab_col_names(df)
num_cols = [col for col in num_cols if col not in "PASSENGERID"]


#######################################################################################
# 2 - Aykırı Değerler
#######################################################################################


def outlier_threshold(dataframe, col_name, q1=0.25, q3=0.75):
    quartile1 = dataframe[col_name].quantile(q1)
    quartile3 = dataframe[col_name].quantile(q3)
    interquantile_range = quartile3 - quartile1
    up_limit = quartile3 + 1.5 * interquantile_range
    low_limit = quartile1 - 1.5 * interquantile_range
    return low_limit, up_limit


def check_outlier(dataframe, col_name):
    low_limit, up_limit = outlier_threshold(dataframe, col_name)
    if dataframe[(dataframe[col_name] < low_limit) | (dataframe[col_name] > up_limit)].any(axis=None):
        return True
    else:
        return False


for col in num_cols:
    print(col, check_outlier(df, col))


def replace_with_thresholds(dataframe, variable):
    low_limit, up_limit = outlier_threshold(dataframe, variable)
    dataframe.loc[(dataframe[variable] > up_limit), variable] = up_limit
    dataframe.loc[(dataframe[variable] < low_limit), variable] = low_limit


for col in num_cols:
    replace_with_thresholds(df, col)


#######################################################################################
# 3 - Eksik Değer
#######################################################################################


def missing_values_table(dataframe, na_name=False):
    na_columns = [col for col in dataframe.columns if dataframe[col].isnull().sum() > 0]
    n_miss = dataframe[na_columns].isnull().sum().sort_values(ascending=False)
    ratio = (dataframe[na_columns].isnull().sum() / dataframe.shape[0] * 100).sort_values(ascending=False)
    missing_df = pd.concat([n_miss, np.round(ratio, 2)], axis=1, keys=["n_miss", "ratio"])
    print(missing_df, end="\n")

    if na_name:
        return na_columns


missing_values_table(df)
df.drop("CABIN", inplace=True, axis=1)

remove_cols = ["TICKET", "NAME"]
df.drop(remove_cols, inplace=True, axis=1)

df["AGE"] = df["AGE"].fillna(df.groupby("NEW_TITLE")["AGE"].transform("median"))

df["NEW_AGE_PCLASS"] = df["AGE"] * df["PCLASS"]

df.loc[(df["AGE"] < 18), "NEW_AGE_CAT"] = "young"
df.loc[(df["AGE"] >= 18) & (df["AGE"] < 56), "NEW_AGE_CAT"] = "mature"
df.loc[(df["AGE"] >= 56), "NEW_AGE_CAT"] = "senior"

df.loc[(df["SEX"] == "male") & (df["AGE"] <= 21), "New_Sex_Cat"] = "young_male"
df.loc[(df["SEX"] == "male") & ((df["AGE"] > 21) & df["AGE"] <= 50), "New_Sex_Cat"] = "mature_male"
df.loc[(df["SEX"] == "male") & (df["AGE"] > 50), "New_Sex_Cat"] = "senior_male"
df.loc[(df["SEX"] == "female") & (df["AGE"] <= 21), "New_Sex_Cat"] = "young_female"
df.loc[(df["SEX"] == "female") & ((df["AGE"] > 21) & df["AGE"] <= 50), "New_Sex_Cat"] = "mature_female"
df.loc[(df["SEX"] == "female") & (df["AGE"] > 50), "New_Sex_Cat"] = "senior_female"

df = df.apply(lambda x: x.fillna(x.mode()[0]) if (x.dtype == "O" and len(x.unique()) <= 10) else x, axis=0)

#######################################################################################
# 4 - Label Encoding
#######################################################################################
binary_cols = [col for col in df.columns if df[col].dtype == "O" and df[col].nunique() == 2]


def label_encoder(dataframe, binary_col):
    labelencoder = LabelEncoder()
    dataframe[binary_col] = labelencoder.fit_transform(dataframe[binary_col])
    return dataframe


for col in binary_cols:
    label_encoder(df, col)


#######################################################################################
# 5 - Rare Encoding
#######################################################################################


def rare_analyser(dataframe, target, cat_cols):
    for col in cat_cols:
        print(col, "", len(dataframe[col].value_counts()))
        print(pd.DataFrame({"Count": dataframe[col].value_counts(),
                            "Ratio": dataframe[col].value_counts() / len(dataframe),
                            "Target_Mean": dataframe.groupby(col)[target].mean()}),
              end="\n\n\n")


rare_analyser(df, "SURVIVED", cat_cols)


def rare_encoder(dataframe, rare_perc):
    temp_df = dataframe.copy()

    rare_columns = [col for col in temp_df.columns if temp_df[col].dtypes == "O"
                    and (temp_df[col].value_counts() / len(temp_df) < rare_perc).any(axis=None)]

    for var in rare_columns:
        tmp = temp_df[var].value_counts() / len(temp_df)
        rare_labels = tmp[tmp < rare_perc].index
        temp_df[var] = np.where(temp_df[var].isin(rare_labels), "Rare", temp_df[var])

    return temp_df


df = rare_encoder(df, rare_perc=0.01)
df["NEW_TITLE"].value_counts()

#######################################################################################
# 6 - One Hot Encoding
#######################################################################################
ohe_cols = [col for col in df.columns if 10 >= df[col].nunique() > 2]


def one_hot_encoder(dataframe, categorical_cols, drop_first=False):
    dataframe = pd.get_dummies(dataframe, columns=categorical_cols, drop_first=drop_first)
    return dataframe


df = one_hot_encoder(df, ohe_cols, drop_first=True)
df.head()
df.info()

cat_cols, num_cols, cat_but_car = grab_col_names(df)
num_cols = [col for col in num_cols if col not in "PASSENGERID"]
rare_analyser(df, "SURVIVED", cat_cols)

useless_cols = [col for col in df.columns if df[col].nunique() == 2 and
                (df[col].value_counts() / len(df) < 0.01).any(axis=None)]

df.drop(useless_cols, axis=1, inplace=True)

#######################################################################################
# 7 - Scaler
#######################################################################################
scaler = StandardScaler()
df[num_cols] = scaler.fit_transform(df[num_cols])
df[num_cols].head()
df.head()
######################################################################################################

#######################################################################################
# 8 - Model
#######################################################################################
y = df["SURVIVED"]
X = df.drop(["PASSENGERID", "SURVIVED"], axis=1)

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.30, random_state=17)

from sklearn.ensemble import RandomForestClassifier

rf_model = RandomForestClassifier(random_state=46).fit(X_train, y_train)
y_pred = rf_model.predict(X_test)
accuracy_score(y_pred, y_test)

###################################################################
# Yeni Üretilen değişkenlerin durumu
#################################################################


def plot_importance(model, features, num=len(X), save=False):
    feature_imp = pd.DataFrame({"Value": model.feature_importances_,
                                "Feature": features.columns})
    plt.figure(figsize=(10, 10))
    sns.set(font_scale=1)
    sns.barplot(x="Value", y="Feature", data=feature_imp.sort_values(by="Value",
                                                                     ascending=False)[0:num])
    plt.title("Features")
    plt.tight_layout()
    plt.show()
    if save:
        plt.savefig("importances.png")


plot_importance(rf_model, X_train)
df.head()