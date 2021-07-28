
import pandas as pd
import datetime as dt
import os
from sklearn.preprocessing import MinMaxScaler
import scipy.stats as st
import math

pd.set_option('display.max_columns', None)
pd.set_option('display.max_rows', 20)
pd.set_option('display.expand_frame_repr', False)
pd.set_option('display.float_format', lambda x: '%.5f' % x)

df_ = pd.read_csv("amazon_review.csv")
df = df_.copy()
df.head()

df["overall"].mean()
df["reviewTime"].max()
today_date = dt.datetime(2014,12,10)

df.info()
df["reviewTime"] = pd.to_datetime(df["reviewTime"])

df["day"] = (today_date - df["reviewTime"]).dt.days


df["day"].describe([0.10, 0.25, 0.50, 0.70, 0.80, 0.90, 0.95, 0.99]).T
# 20 q = 250
# 50 q = 433
# 75 q = 603
df["day"].quantile(0.75)

# we have to take into account the last comments

(df.loc[df["day"] <= 250 , "overall"].mean()) * (29/100) + \
(df.loc[(df["day"] <= 433) & (df["day"] >250) , "overall"].mean()) * (27/100) +\
(df.loc[(df["day"] <= 603) &  (df["day"] > 433)  , "overall"].mean()) * (23/100)  + \
(df.loc[df["day"] > 603 , "overall"].mean()) * (21/100)


def time_based_weighted_average(dataframe, w1=29, w2=27, w3=23, w4=21):
    return (dataframe.loc[df["day"] <= 250 , "overall"].mean()) * (29/100) + \
    (dataframe.loc[(dataframe["day"] <= 433) & (dataframe["day"] >250) , "overall"].mean()) * (27/100) +\
    (dataframe.loc[(dataframe["day"] <= 603) &  (dataframe["day"] > 433)  , "overall"].mean()) * (23/100)  + \
    (dataframe.loc[dataframe["day"] > 603 , "overall"].mean()) * (21/100)

time_based_weighted_average(df)

df["overall"].mean()
# according to previous ones = 4.599767755823966
# direct mean = 4.587589013224822


#######################################################################################


df.loc[df["helpful_yes"] == 2, ["helpful_yes","total_vote" ]]

df["helpful_no"] = df["total_vote"] - df["helpful_yes"]
df.head()

# usual sorting methods but they are not right
# we'll be observing this situation

def score_average_rating(up, down):
    if up + down == 0:
        return 0
    return up / (up + down)

df["score_average_rating"] = df.apply(lambda x: score_average_rating(x["helpful_yes"], x["helpful_no"]), axis=1)

def score_up_down_diff(up, down):
    return up - down

df["score_pos_neg_diff"] = df.apply(lambda x: score_up_down_diff(x["helpful_yes"],
                                                                             x["helpful_no"]),axis=1)

def wilson_lower_bound(up, down, confidence=0.95):
    """
    Wilson Lower Bound Score hesapla

    - Bernoulli parametresi p için hesaplanacak güven aralığının alt sınırı WLB skoru olarak kabul edilir.
    - Hesaplanacak skor ürün sıralaması için kullanılır.
    - Not:
    Eğer skorlar 1-5 arasıdaysa 1-3 negatif, 4-5 pozitif olarak işaretlenir ve bernoulli'ye uygun hale getirilebilir.
    Bu beraberinde bazı problemleri de getirir. Bu sebeple bayesian average rating yapmak gerekir.

    Parameters
    ----------
    up: int
        up count
    down: int
        down count
    confidence: float
        confidence

    Returns
    -------
    wilson score: float

    """
    n = up + down
    if n == 0:
        return 0
    z = st.norm.ppf(1 - (1 - confidence) / 2)
    phat = 1.0 * up / n
    return (phat + z * z / (2 * n) - z * math.sqrt((phat * (1 - phat) + z * z / (4 * n)) / n)) / (1 + z * z / n)




df["comment_score"] = df.apply(lambda x : wilson_lower_bound(x["helpful_yes"], x["helpful_no"]),axis=1)
df.head()

df.sort_values(by = "comment_score",ascending=False).head(20)

df[["reviewText","comment_score"]].sort_values(by = "comment_score",ascending=False).head(1)
df.iloc[:, 10:].sort_values(by = "comment_score",ascending=False).head(20)

# results

# helpful_yes  total_vote   day  helpful_no  score_average_rating  score_pos_neg_diff  comment_score
# 2031         1952        2020   704          68               0.96634                1884        0.95754
# 3449         1428        1505   805          77               0.94884                1351        0.93652
# 4212         1568        1694   581         126               0.92562                1442        0.91214
# 317           422         495  1035          73               0.85253                 349        0.81858
# 4672           45          49   160           4               0.91837                  41        0.80811
# 1835           60          68   285           8               0.88235                  52        0.78465
# 3981          112         139   779          27               0.80576                  85        0.73214
# 3807           22          25   651           3               0.88000                  19        0.70044
# 4306           51          65   825          14               0.78462                  37        0.67033
# 4596           82         109   809          27               0.75229                  55        0.66359
# 315            38          48   849          10               0.79167                  28        0.65741
# 1465            7           7   240           0               1.00000                   7        0.64567
# 1609            7           7   259           0               1.00000                   7        0.64567
# 4302           14          16   264           2               0.87500                  12        0.63977
# 4072            6           6   761           0               1.00000                   6        0.60967
# 1072            5           5   944           0               1.00000                   5        0.56552
# 2583            5           5   491           0               1.00000                   5        0.56552
# 121             5           5   945           0               1.00000                   5        0.56552
# 1142            5           5   309           0               1.00000                   5        0.56552
# 1753            5           5   779           0               1.00000                   5        0.56552
