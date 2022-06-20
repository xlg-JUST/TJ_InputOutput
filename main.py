import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.metrics.pairwise import cosine_similarity

plt.rcParams['font.sans-serif'] = ['Simhei']  # 显示中文
plt.rcParams['axes.unicode_minus'] = False    # 显示负号

# table2018 = pd.read_excel(r'dataset/2018.xlsx', sheet_name='生产价投入产出表')
# table2020 = pd.read_excel(r'dataset/2020.xlsx', sheet_name='2020年全国投入产出表')
# columns = table2020.iloc[3, 3:156]
#
# first_quadrant_2018 = table2018.iloc[5:158, 3:156]
# first_quadrant_2018 = first_quadrant_2018.set_index(columns)
# print(first_quadrant_2018)

# first_quadrant_2020 = table2020.iloc[5:158, 3:156]

# sim = np.zeros(shape=(first_quadrant_2018.shape[-1]))
# for i in range(first_quadrant_2020.shape[-1]):
#     cur2020, cur2018 = first_quadrant_2020.iloc[:, i].to_numpy(), first_quadrant_2018.iloc[:, i].to_numpy()
#     # cur2020 /= np.sum(cur2020)
#     # cur2018 /= np.sum(cur2018)
#     cur_sim = cosine_similarity([cur2020, cur2018])[0, 1]
#     sim[i] += cur_sim
#
# arg = np.argsort(sim)
# first = arg[:10]
# last = arg[-10:]
#
# first_columns = columns[first]
# last_columns = columns[last]
#
#
# first_data_2020 = first_quadrant_2020.iloc[:, first].copy()
# first_data_2020 /= first_data_2020.sum()
# first_data_2018 = first_quadrant_2018.iloc[:, first].copy()
# first_data_2018 /= first_data_2018.sum()
# reduce = first_data_2020 - first_data_2018
#
# for i in range(1):
#     cur = reduce.iloc[:, i].sort_values(ascending=False)
#     cur_first = cur.iloc[:10]
#     cur_last = cur.iloc[-10:]
#     cur = pd.concat([cur_first, cur_last])
#     cur_index = (columns[cur.index.to_numpy()-5]).to_list()  # 这里index在完整的表格中，是从第五个开始出数据的  所以要把减5以对齐columns
#     cur = np.float64(cur.to_numpy().reshape((-1, 1)))
#     plt.imshow(cur, aspect='auto', cmap=plt.cm.Blues)
#     plt.colorbar()
#     plt.yticks(np.arange(len(cur_index)), cur_index)
#     for j in range(cur.shape[0]):
#         plt.text(i, j, '{}->{}'.format(first_data_2018.iloc[i, j], first_data_2020.iloc[i, j]), va='center', ha='center')
#     # plt.ylabel(cur_index)
#     plt.show()


table2018, table2020 = pd.read_excel('dataset/第一象限.xlsx', sheet_name='2018'),\
                       pd.read_excel('dataset/第一象限.xlsx', sheet_name='2020')

table2018.set_index('Unnamed: 0', inplace=True)
table2020.set_index('Unnamed: 0', inplace=True)

# 计算相似度
sim = np.zeros(shape=(table2020.shape[-1]))
for i in range(table2018.shape[-1]):
    cur2020, cur2018 = table2020.iloc[:, i].to_numpy(), table2018.iloc[:, i].to_numpy()
    # cur2020 /= np.sum(cur2020)
    # cur2018 /= np.sum(cur2018)
    cur_sim = cosine_similarity([cur2020, cur2018])[0, 1]
    sim[i] += cur_sim

arg = np.argsort(sim)
first = arg[:10]
last = arg[-10:]

# 先算前10
first_data_2020 = table2020.iloc[:, first].copy()
first_data_2020 /= first_data_2020.sum(axis=0)
first_data_2018 = table2018.iloc[:, first].copy()
first_data_2018 /= first_data_2018.sum(axis=0)
reduce = first_data_2020 - first_data_2018

# 作图
for i in range(1):
    column = first_data_2020.columns[i]
    cur = reduce.iloc[:, i].sort_values(ascending=False)
    cur_first = cur.iloc[:10]
    cur_last = cur.iloc[-10:]
    cur = pd.concat([cur_first, cur_last])
    plt.imshow(np.float64(cur.to_numpy().reshape((-1, 1))), aspect='auto', cmap=plt.cm.Blues)
    plt.colorbar()
    plt.xticks([0], [column])
    plt.yticks(np.arange(cur.shape[0]), cur.index)
    for j in range(cur.shape[0]):
        if j <= int(cur.shape[0]/2):
            plt.text(i, j, '{}%->{}%'.format(np.round(first_data_2018.loc[cur.index[j], column]*100, 4), np.round(first_data_2020.loc[cur.index[j], column]*100, 4)), va='center', ha='center', color='white')
        else:
            plt.text(i, j, '{}%->{}%'.format(np.round(first_data_2018.loc[cur.index[j], column]*100, 4),
                     np.round(first_data_2020.loc[cur.index[j], column]*100, 4)), va='center',
                     ha='center')
    plt.show()








