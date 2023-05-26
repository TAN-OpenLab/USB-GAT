import os
import pickle
import numpy as np
import networkx as nx
import scipy.sparse as sp

TIME_WINDOWS = 6
NUM_NODES = 40
MAX_TXT_LEN = 768
url = r'D:\zys\pheme-rnr-dataset\pheme-rnr-dataset'
# datasets = ['germanwings-crash']
datasets = ['germanwings-crash', 'charliehebdo', 'ferguson', 'ottawashooting', 'sydneysiege']
# datasets = ['charliehebdo', 'ferguson', 'ottawashooting', 'sydneysiege']
types = ['rumours', 'non-rumours']
x_len = {}
x_len['germanwings-crash'] = 78
x_len['charliehebdo'] = 274
x_len['ferguson'] = 204
x_len['sydneysiege'] = 365
x_len['ottawashooting'] = 232

type2id = {}
type2id['rumours'] = '1'
type2id['non-rumours'] = '0'

for dataset in datasets:

    for type in types:
        len_data = 0
        batch_x_len = {}
        # 按照l参与者数量排序
        f = open(os.path.join(url, 'pro_cascades', dataset + '_' + type + '_cascade.txt'), 'r', encoding='utf-8')
        print(os.path.join(url, 'pro_cascades', dataset + '_' + type + '_cascade.txt'))
        for line in f.readlines():
            g = nx.DiGraph()
            node_time = {}
            cascade = {}
            line = line.strip('\n').split('\t')
            # 文本id
            wid = line[1]
            # 传播路径
            paths = line[5].split(' ')
            # 标签
            label = int(line[6])
            # 路径长度
            batch_x_len[wid] = len(paths)
            # key:wid value:length
            sort_x = sorted(batch_x_len.keys(), key=lambda x: batch_x_len[x], reverse=True)
            sort_x = sort_x[:274]
            len_data += 1
        print(len_data)
        f.close()
        # 保留长度靠前的数据
        len_temp = len_data
        len_data = 0
        f = open(os.path.join(url, 'pro_cascades', dataset + '_' + type + '_cascade.txt'), 'r', encoding='utf-8')
        for line in f.readlines():
            if len_data == len_temp / 2:
                break
            g = nx.DiGraph()
            node_time = {}
            cascade = {}
            line = line.strip('\n').split('\t')
            wid = line[1]
            paths = line[5].split(' ')
            label = int(line[6])
            if wid not in sort_x:
                continue
            # 记录该信息的最后转发时间，并将所有时间划分为6个时间间隔
            max_time = 0
            # 节点的对应新id
            node_new_id = {}
            n = 0
            cascade[wid] = {}
            for path in paths:
                # nodes 数组 <父节点，子节点>
                nodes = path.split(':')[0].split('/')
                # 时间
                t = int(path.split(":")[1])
                # 控制节点数量
                if n >= NUM_NODES:
                    # i=0
                    for i in range(len(nodes) - 1):
                        if nodes[i + 1] not in node_new_id.keys():
                            continue
                        else:
                            if nodes[i] in node_new_id.keys():
                                g.add_edge(nodes[i], nodes[i + 1])
                            node_time[nodes[i + 1]].append(t)
                            if t > max_time:
                                max_time = t
                else:
                    # 根节点
                    if len(nodes) == 1:
                        if nodes[0] not in node_time.keys():
                            node_time[nodes[0]] = [0]
                            # 根节点 node_new_id[root_userid]=0
                            node_new_id[nodes[0]] = n
                            n += 1
                    else:
                        # i=0
                        for i in range(len(nodes) - 1):
                            # 在g中添加边
                            g.add_edge(nodes[i], nodes[i + 1])
                            if nodes[i] not in node_time.keys():
                                node_time[nodes[i]] = [1]
                                node_new_id[nodes[i]] = n
                                n += 1
                            if nodes[i + 1] not in node_time.keys():
                                node_time[nodes[i + 1]] = []
                                node_new_id[nodes[i + 1]] = n
                                n += 1
                            node_time[nodes[i + 1]].append(t)
                            if t > max_time:
                                max_time = t
            # 打开bert文件
            bert_f = open(os.path.join(url, 'pro_content_emb', dataset, type, wid + '.txt'), 'r', encoding='utf-8')
            user_f = open(os.path.join(url, 'pro_user_emb', dataset, type, wid + '.txt'), 'r', encoding='utf-8')
            user_content = np.zeros((len(node_new_id), MAX_TXT_LEN), dtype=float)
            user_emb = np.zeros((len(node_new_id), 8), dtype=float)

            node_app = {}
            # 读取词向量文件
            for line in bert_f.readlines():
                line = line.strip('\n').split('\t')
                # node: userid
                node = line[1]
                txt = line[2].split(' ')
                if node not in node_new_id.keys():
                    continue
                if node not in node_app.keys():
                    node_app[node] = 0
                node_app[node] += 1
                if txt == ['']:
                    user_content[node_new_id[node], :768] = 0
                else:
                    txt = list(map(float, txt))
                    txt_len = len(txt)
                    user_content[node_new_id[node], :txt_len] = np.array(txt)

            node_app_2 = {}
            for line in user_f.readlines():
                line = line.strip('\n').split('\t')
                node2 = line[1]
                # 源id 转发id 转发名字 verified geo followers friends favourites time_cha user_time_cha lichang
                user_verified = int(line[3])
                user_geo = int(line[4])
                user_followers = int(line[5])
                user_friends = int(line[6])
                user_favourites = int(line[7])
                user_time_cha = int(line[8])
                user_zhucetime_cha = int(line[9])
                user_lichang = int(line[10])
                # user_followers
                if user_followers >=0 and user_followers < 100:
                    user_followers = 0
                elif user_followers >=100 and user_followers < 500:
                    user_followers = 2
                else:
                    user_followers = 5
                # user_friends
                if user_friends >=0 and user_friends < 100:
                    user_friends = 0
                elif user_friends >=100 and user_friends < 200:
                    user_friends = 1
                else:
                    user_friends = 3
                # user_favourites
                if user_favourites >=0 and user_favourites < 100:
                    user_favourites = 0
                elif user_favourites >=100 and user_favourites < 200:
                    user_favourites = 1
                else:
                    user_favourites = 2
                # user_time_cha
                if user_time_cha  >=0 and user_time_cha  < 3600*2:
                    user_time_cha  = 0
                elif user_time_cha  >=3600*2 and user_time_cha  < 3600*6:
                    user_time_cha  = 1
                else:
                    user_time_cha  = 2
                # user_zhucetime_cha
                if user_zhucetime_cha  >=0 and user_zhucetime_cha  < 365:
                    user_zhucetime_cha  = 0
                elif user_zhucetime_cha  >=365 and user_zhucetime_cha  < 365*2:
                    user_zhucetime_cha  = 1
                else:
                    user_zhucetime_cha  = 2

                # user_list = [line[3], line[4], line[5], line[6], line[7], line[8], line[9], line[10]]
                user_list = [user_verified, user_geo, user_followers, user_friends, user_favourites, user_time_cha, user_zhucetime_cha, user_lichang]
                if node2 not in node_new_id.keys():
                    continue
                if node2 not in node_app_2.keys():
                    node_app_2[node2] = 0
                node_app_2[node2] += 1

                user_list = list(map(float, user_list))
                user_emb[node_new_id[node2], :8] = np.array(user_list)

            g_new = nx.DiGraph()
            for (s, t) in list(nx.edges(g)):
                g_new.add_edge(node_new_id[s], node_new_id[t])
            g_new.remove_edges_from(list(nx.selfloop_edges(g_new)))
            g_adj = nx.adjacency_matrix(g_new).todense()

            # print(g_adj)
            edge_index_temp = sp.coo_matrix(g_adj)
            # print(edge_index_temp)

            values = edge_index_temp.data  # 边上对应权重值weight
            indices = np.vstack((edge_index_temp.row, edge_index_temp.col))  # 我们真正需要的coo形式
            # edge_index_A = torch.LongTensor(indices)  # 我们真正需要的coo形式

            X = np.array(user_content)
            # X = torch.FloatTensor(X)

            user_emb = np.array(user_emb)
            urll = 'D:\zys\pheme-rnr-dataset\pheme'
            file = open(os.path.join(urll, 'all_data_emb', dataset,
                                     str(len_data) + '_' + type2id[type] + '_' + dataset + '_N50T6.pkl'), 'wb')
            pickle.dump((X, indices, label, user_emb), file)
            len_data += 1
