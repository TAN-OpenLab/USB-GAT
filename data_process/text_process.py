# encoding= 'utf-8'
# pheme-rnr-dataset
import json
import pprint
import os
import glob
import pickle
from datetime import datetime
# from nltk.corpus import stopwords
# from nltk.tokenize import word_tokenize
import re
import sys

sys.setrecursionlimit(10000)


def add(structure, content_dict, replies, s):
    for t in replies[s]:
        if t in structure.keys():
            structure = add(structure, replies, t)
            structure[s]['cascade'].extend(structure[t]['cascade'])
            content_dict[s].extend(content_dict[t])
            del structure[t]
            del content_dict[t]
    return structure, content_dict


if __name__ == '__main__':

    url = r'F:\Dataset\pheme-rnr-dataset'
    datasets = ['germanwings-crash', 'charliehebdo', 'ferguson', 'ottawashooting', 'sydneysiege']  #
    types = ['rumours', 'non-rumours']
    # 1:谣言 0:非谣言
    for dataset in datasets:
        for type in types:
            i = 1
            if type == 'rumours':
                label = 1
            else:
                label = 0

            global structure
            structure = {}
            content_dict = {}
            global replies
            replies = {}
            # 加载一个推文事件
            for tweets in glob.glob(os.path.join(url, dataset, type, '*')):
                # 每个推文事件 写入一个txt 每行是  文本id 用户id 用户特征...
                # 传播图
                # 读取source json 文件
                source = tweets + "\\source-tweet\\*.json"
                sourceData = json.load(open(glob.glob(source)[0]))
                # 文本id
                sourceWId = sourceData['id_str']
                # 用户id
                sourceUId = sourceData['user']['id_str']
                structure[sourceWId] = {}
                content_dict[sourceWId] = []
                # publication time
                time_pub = sourceData['created_at']
                t = time_pub.split(' ')
                t = t[1] + " " + t[2] + " " + t[5] + " " + t[3]
                time_pub = datetime.strptime(t, '%b %d %Y %H:%M:%S')
                time_pub = int(datetime.timestamp(time_pub))
                structure[sourceWId]['time'] = time_pub
                structure[sourceWId]['userId'] = sourceUId
                structure[sourceWId]['cascade'] = []

                reactions = tweets + "\\reactions\\*.json"
                for reaction in glob.glob(reactions):
                    reactionData = json.load(open(reaction))
                    reactionWId = reactionData["id_str"]
                    if reactionWId == sourceWId:
                        continue
                    replyTo = reactionData["in_reply_to_user_id_str"]
                    reactionUId = reactionData['user']['id_str']
                    # content process
                    txt = reactionData['text']
                    txt = re.sub(r'(@.*?)[\s]+', ' ', txt)
                    # Replace '&amp;' with '&'
                    txt = re.sub(r'&amp;', '&', txt)
                    txt = re.sub(r'\
                    n+', ' ', txt)
                    # Remove trailing whitespace 删除空格
                    txt = re.sub(r'\s+', ' ', txt).strip()
                    p = re.compile(r'http[s]?://(?:[a-zA-Z]|[0-9]|[$-_@.&+]|[!*\(\),]|(?:%[0-9a-fA-F][0-9a-fA-F]))+',
                                   re.S)
                    txt = re.sub(p, '', txt)
                    reactiontxt = txt
                    # public time
                    time = reactionData['created_at']  # Sun Nov 06 21:21:26 +0000 2011
                    t = time.split(' ')
                    t = t[1] + " " + t[2] + " " + t[5] + " " + t[3]
                    time = datetime.strptime(t, '%b %d %Y %H:%M:%S')
                    time = int(datetime.timestamp(time))
                    if sourceWId not in replies.keys():
                        replies[sourceWId] = []
                    replies[sourceWId].append(reactionWId)
                    if replyTo == None:
                        replyTo = sourceUId
                    structure[sourceWId]['cascade'].append([replyTo, reactionUId, time])
                    content_dict[sourceWId].append(str(reactionWId) + '\t' + str(reactionUId) + '\t' + reactiontxt)

            for s in replies.keys():
                for t in replies[s]:
                    if t in structure.keys():
                        structure, content_dict = add(structure, content_dict, replies, t)

            n = 1
            result_dir = os.path.join(url, 'pro_cascades', dataset + '_' + type + "_cascade.txt")

            with open(result_dir, 'a', encoding='utf-8') as fp:
                for key, value in structure.items():
                    c = []

                    if len(structure[key]['cascade']) == 0:
                        continue
                    for i in range(len(structure[key]['cascade'])):
                        c.append(structure[key]['cascade'][i][0] + '/' + structure[key]['cascade'][i][1] + ':' + str(
                            structure[key]['cascade'][i][2] - structure[key]['time']))

                    cascade = str(n) + '\t' + str(key) + '\t' + str(structure[key]['userId']) + '\t' + str(
                        structure[key]['time']) + '\t' + str(
                        len(c) + 1) + '\t' + str(structure[key]['userId']) + ':' + str(0) + ' ' + ' '.join(
                        c) + '\t' + str(label) + '\n'  # str(node_time.get(node)
                    fp.write(cascade)
                    content_dir = os.path.join(url, 'pro_content', dataset, type, str(key) + ".txt")
                    con_f = open(content_dir, 'a', encoding='utf-8')
                    con_f.write('\n'.join(content_dict[key]))
                    n += 1
