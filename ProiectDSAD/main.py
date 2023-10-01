import columns as columns
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from matplotlib import cm
import seaborn as sns
from PIL.XVThumbImagePlugin import PALETTE
from pandas.core.interchange import dataframe

plt.style.use('ggplot')

#citirea datelor din csv
dataset=pd.read_csv("Most Popular Programming Languages from 2004 to 2022.csv")

dataset["Date"]=pd.to_datetime(dataset["Date"])
dataset.set_index("Date",inplace=True)
#print(dataset)
print(dataset.describe())

data=list(dataset.index)
#print(data)

nume_limbaje=list(dataset.columns)
#print(nume_limbaje)

x=dataset.values
#print(x)

#returneaza o matrice de valori
m,n=x.shape
#print(x[0,:])

#eliminare valori lipsa
is_nan=pd.isna(x)
#print(is_nan)

#aflu indexul fiecarei valori nan
k=np.where(is_nan)
#print(k)

#media pe randuri
x[k]=np.nanmean(x[:,k[1]],axis=0)
#print(x[3,:])


popular_2004 = pd.DataFrame({"Languages": dataset.iloc[0].T.index,
                            "Popularity": dataset.iloc[0].values.T})


popular_2022 = pd.DataFrame({"Languages": dataset.iloc[210].T.index,
                            "Popularity": dataset.iloc[210].values.T})

popular_2004=popular_2004.to_csv("Most popular languages in 2004")
popular_2022=popular_2022.to_csv("Most popular languages in 2022")


columns = dataset.columns.tolist()


var=list(dataset.columns)[1:]

pop_lang=pd.read_csv("Most popular languages in 2022")
t1=dataset.merge(right=pop_lang,left_index=True,right_index=True)
popularity=t1[var + ["Popularity"]].groupby(by="Popularity").agg(sum)



# primul grafic
mask = dataset.mean() > 2.5

data = dataset.loc[:, mask]

clms = data.columns.tolist()
clms
plt.figure(figsize=(15, 8))
sns.set(style='dark')

for language in clms:
    sns.lineplot(x=data.index, y=data[language], label=language)

plt.ylabel('Popularity', fontsize=12)
plt.xlabel('Year', fontsize=12)
plt.title('Popular Prgramming Languages', fontsize=20)
plt.legend(loc=2)
plt.yticks(fontsize=10)
plt.xticks(rotation=45, fontsize=10)
plt.tight_layout()
plt.show()


#al doilea grafic
df_popular = pd.DataFrame({'Languages':dataset.tail(1).T.index, 'Popularity':dataset.tail(1).T.values.flatten()})
print(df_popular)

df_popular_sorted = df_popular.sort_values(by=['Popularity'])
fig, ax = plt.subplots(figsize=(12,10))
df_popular_sorted.plot(x='Languages', y='Popularity',
                kind='barh', ax=ax, color='green')
plt.xlabel('Popularity Score')
plt.ylabel('Programming Language')
plt.title('Programming Language Popularity (January 2022)')
plt.legend(bbox_to_anchor=(1.05,1))
plt.show()

#al treilea grafic

PALETTE = "magma_r"
sns.set(style="darkgrid")


def generate_color_series(n):
    segments = cm.get_cmap(PALETTE, n)
    return segments(range(n))
data_mean_list = []

for col in data:
    if col == "Date":
        pass
    else:
        data_mean_list.append([col, data[col].mean()])

data_mean = pd.DataFrame(data_mean_list, columns = ["language", "mean"])
data_mean = data_mean.sort_values(by=["mean"], ascending=False)

plt.figure(figsize=(16, 6))
plt.subplot(1, 2, 1)
plt.title("Mean popularity of languages")
sns.barplot(data=data_mean, x="mean", y="language", palette=generate_color_series(28))
plt.xlabel("Mean popularity in %")
plt.ylabel("Programming language")


plt.subplot(1, 2, 2)
data_mean_top10 = data_mean.nlargest(10, "mean")
plt.title("Top 10 mean popular languages")
donut_top10 = plt.Circle( (0,0), 0.7, color='white')
plt.pie(data_mean_top10["mean"],labels=data_mean_top10["language"], wedgeprops = {"linewidth": 5, "edgecolor": "white"}, colors=generate_color_series(10))
p = plt.gcf()
p.gca().add_artist(donut_top10)

plt.show()