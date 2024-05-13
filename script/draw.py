from time import time
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from tqdm import tqdm
 
from sklearn import datasets
from sklearn.manifold import TSNE

color_table = ['b','g','r','c','m','y','k','w','tab:blue','tab:orange','tab:green','tab:red','tab:purple','tab:pink','tab:grey','tab:olive','tab:cyan', 'lightcoral','indianred','maroon','brown','firebrick','darkred','salmon','tomato','orangered']

def get_data():
    digits = datasets.load_digits(n_class=6)
    data = digits.data
    print(data.shape)
    assert 1==0
    label = digits.target
    n_samples, n_features = data.shape
    return data, label, n_samples, n_features


def get_app_data(file):
    df = pd.read_csv(file)
    df = df[:3000]
    label = np.array(df["pack_name"])
    df.drop(columns=["pack_name"], inplace=True)
    data = df.values
    print(data.shape)
    return data, label


def get_data_label(file):
    df = pd.read_csv(file, sep='\t',usecols=["label"]+[str(i) for i in range(64)], nrows=20000)
    df.dropna(inplace=True)
    df = df[:10000]
    # print(len(df))
    label = np.array(df["label"])
    df.drop(columns=["label"], inplace=True)
    data = df.values
    print(data.shape)
    return data, label


def get_user_data(file):
    df = pd.read_csv(file)
    df = df[:3000]
    label = np.array(df["imei"])
    df.drop(columns=["imei"], inplace=True)
    data = df.values
    print(data.shape)
    return data, label


def get_bc_app_data(file):
    df = pd.read_csv(file)
    label = np.array(df["pack_name"])
    emb_list = [eval(e) for e in df["emb"]]
    df["emb"] = emb_list
    for i in tqdm(range(64)):
        df['a_{}'.format(i)] = df['emb'].map(lambda x:x[i])
    df.drop(columns=["pack_name", "emb"], inplace=True)
    data = df.values
    print(data.shape)
    return data, label

 
def plot_embedding(data, label, title):
    x_min, x_max = np.min(data, 0), np.max(data, 0)
    data = (data - x_min) / (x_max - x_min)
 
    fig = plt.figure(figsize=(10,10))
    # ax = plt.subplot(111)
    for i in range(data.shape[0]):
        # plt.scatter(data[i, 0], data[i, 1], s=0.05, color='b' if label[i] == 0 else 'r'),
        # plt.scatter(data[i, 0], data[i, 1], s=0.1, color='b'),
        plt.scatter(data[i, 0], data[i, 1], color=color_table[label[i] % len(color_table)]),
    plt.xticks([])
    plt.yticks([])
    plt.title(title)
    return fig

def plot_seborn_embedding(data):
    fig = sns.scatterplot(data = data, x="x", y="y", hue="label", palette=sns.color_palette("magma", 32), legend=False,s=10)
    # "tab20c"
    # "Paired", 32
    fig.get_figure().savefig("/home/notebook/data/group/app-series/ad_platform/in_un_embs/app_emb_abcn.png", dpi=400)
    #sns.relplot(x = "x", y = )


def kmeans(data):
    from sklearn.cluster import KMeans
    n_cluster = 32
    cluster = KMeans(n_clusters=n_cluster, random_state=42).fit(data)
    label = cluster.labels_
    return label
 

def app_plot_embedding(data, label, title):
    x_min, x_max = np.min(data, 0), np.max(data, 0)
    data = (data - x_min) / (x_max - x_min)
 
    fig = plt.figure(figsize=(20,20))
    ax = plt.subplot(111)
    for i in range(data.shape[0]):
        plt.text(data[i, 0], data[i, 1], str(label[i]),
                 #color=plt.cm.Set1(label[i] / 10.),
                 color='b',
                 fontdict={'weight': 'bold', 'size': 1})
    plt.xticks([])
    plt.yticks([])
    plt.title(title)
    return fig

 
def main():
    sns.set_theme(style='white')
    # plt.rcParams['axex.unicode_minus'] = False

    # data, label, n_samples, n_features = get_data()
    data, label = get_app_data('/home/notebook/data/personal/adT1_app_embs.csv')
    # data, label = get_app_data('/home/notebook/data/personal/adT1_app_embs_abcn.csv')
    # data, label = get_data_label("/home/notebook/data/group/app-series/pretrain_game/gy_open/down_train_all.csv")
    # data, label = get_app_data('/home/notebook/data/personal/adT1_app_embs_wo_cate.csv')
    # data, label = get_bc_app_data('/home/notebook/data/group/app-series/ad_platform/app_emb_ad_202207_v2.csv')
    # df = pd.read_csv("/home/notebook/data/group/app-series/ad_platform/adT1/abcn_sample_hist.csv", nrows=10001, sep='\t', usecols=["imei","pack_keep_name", "install_seq","uninstall_seq","gender","age","province"])
    # df.to_csv("/home/notebook/data/group/app-series/ad_platform/in_un_embs/user_emb_abcn_records.csv", sep='^', index=False)
    # assert 1==0
    

    # data, label = get_user_data('/home/notebook/data/group/app-series/ad_platform/in_un_embs/abcn_cl_imei_embs_sum_0.csv')
    print('Computing t-SNE embedding')
    tsne = TSNE(n_components=2, init='pca', random_state=42)
    t0 = time()
    result = tsne.fit_transform(data)
    df_re = pd.DataFrame(result, columns=["x","y"])
    label = kmeans(result)
    df_re["label"] = label 
    plot_seborn_embedding(df_re)

    # fig = plot_embedding(result, label,
    #                      't-SNE embedding of the digits (time %.2fs)'
    #                      % (time() - t0))
    # fig = app_plot_embedding(result, label,
    #                      't-SNE embedding of the digits (time %.2fs)'
    #                      % (time() - t0))
    # print(fig)
    # fig.show()
    # fig.savefig("/home/notebook/data/group/app-series/ad_platform/in_un_embs/user_emb_abcn.png", dpi=400)
 
 
if __name__ == '__main__':
    main()