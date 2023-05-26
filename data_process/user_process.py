# encoding= 'utf-8'
import json
import os
import glob
from datetime import datetime
import sys
from transformers import AutoModelForSequenceClassification
from transformers import TFAutoModelForSequenceClassification
from transformers import AutoTokenizer, AutoConfig
import numpy as np
from scipy.special import softmax
# Preprocess text (username and link placeholders)

sys.setrecursionlimit(10000)

def preprocess(text):
    new_text = []
    for t in text.split(" "):
        t = '@user' if t.startswith('@') and len(t) > 1 else t
        t = 'http' if t.startswith('http') else t
        new_text.append(t)
    return " ".join(new_text)

MODEL = f"cardiffnlp/twitter-roberta-base-sentiment-latest"
tokenizer = AutoTokenizer.from_pretrained(MODEL)
config = AutoConfig.from_pretrained(MODEL)
# PT
model = AutoModelForSequenceClassification.from_pretrained(MODEL)

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
    # datasets = ['germanwings-crash']
    datasets = ['germanwings-crash', 'charliehebdo', 'ferguson', 'ottawashooting', 'sydneysiege']
    types = ['rumours', 'non-rumours']
    # 1:谣言 0:非谣言
    for dataset in datasets:
        for type in types:
            i = 1
            print(dataset + '  ' + type + '     start')
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
                # screen_name profile_image guanzhu  时间差 账户创建时间
                # 传播图
                # 读取source json 文件
                source = tweets + "\\source-tweet\\*.json"
                sourceData = json.load(open(glob.glob(source)[0]))
                # 文本id
                sourceWId = sourceData['id_str']
                # 用户id
                sourceUId = sourceData['user']['id_str']
                # structure[sourceWId] = {}
                # content_dict[sourceWId] = []
                user_name = sourceData['user']['screen_name']
                if sourceData['user']['verified']:
                    user_verified = 1
                else:
                    user_verified = 0
                if sourceData['geo']:
                    user_geo = 1
                else:
                    user_geo = 0
                user_followers = sourceData['user']['followers_count']
                user_friends = sourceData['user']['friends_count']
                user_favourites = sourceData['user']['favourites_count']
                time_pub = sourceData['created_at']
                t = time_pub.split(' ')
                t = t[1] + " " + t[2] + " " + t[5] + " " + t[3]
                time_pub = datetime.strptime(t, '%b %d %Y %H:%M:%S')
                time_pub_temp = time_pub
                time_pub = int(datetime.timestamp(time_pub))
                time_cha = 0

                user_time = sourceData['user']['created_at']
                t1 = user_time.split(' ')
                t1 = t1[1] + " " + t1[2] + " " + t1[5] + " " + t1[3]
                user_time = datetime.strptime(t1, '%b %d %Y %H:%M:%S')
                # user_time = int(datetime.timestamp(user_time))
                user_time_cha = (time_pub_temp - user_time).days

                f_w = open(os.path.join(url, 'pro_user_emb', dataset, type, str(sourceWId))+'.txt', 'a')
                f_w.write(
                    str(sourceWId) + '\t' + str(sourceUId) + '\t' + user_name + '\t' + str(user_verified) + '\t' + str(
                        user_geo) + '\t' +
                    str(user_followers) + '\t' + str(user_friends) + '\t' + str(user_favourites) + '\t' + str(
                        time_cha) + '\t' + str(user_time_cha) + '\t' + str(label) +'\n')
                reactions = tweets + "\\reactions\\*.json"
                for reaction in glob.glob(reactions):
                    reactionData = json.load(open(reaction))
                    reactionWId = reactionData["id_str"]
                    if reactionWId == sourceWId:
                        continue
                    replyTo = reactionData["in_reply_to_user_id_str"]
                    reactionUId = reactionData['user']['id_str']

                    reaction_name = reactionData['user']['screen_name']
                    if reactionData['user']['verified']:
                        reaction_verified = 1
                    else:
                        reaction_verified = 0
                    if reactionData['geo']:
                        reaction_geo = 1
                    else:
                        reaction_geo = 0

                    reaction_followers = reactionData['user']['followers_count']
                    reaction_friends = reactionData['user']['friends_count']
                    reaction_favourites = reactionData['user']['favourites_count']

                    reaction_time = reactionData['created_at']
                    react = reaction_time.split(' ')
                    react = react[1] + " " + react[2] + " " + react[5] + " " + react[3]
                    reaction_time = datetime.strptime(react, '%b %d %Y %H:%M:%S')
                    reaction_time = int(datetime.timestamp(reaction_time))
                    reaction_time_cha = reaction_time - time_pub

                    react_user_time = reactionData['user']['created_at']
                    react1 = react_user_time.split(' ')
                    react1 = react1[1] + " " + react1[2] + " " + react1[5] + " " + react1[3]
                    react_user_time = datetime.strptime(react1, '%b %d %Y %H:%M:%S')
                    react_user_time_cha = (time_pub_temp - react_user_time).days

                    user_text = reactionData['text']
                    text = preprocess(user_text)
                    encoded_input = tokenizer(text, return_tensors='pt')
                    output = model(**encoded_input)
                    scores = output[0][0].detach().numpy()
                    scores = softmax(scores)
                    user_lichang=np.argmax(scores)-1
                    #源id 转发id 转发名字 verified geo followers friends favourites time_cha user_time_cha lichang
                    f_w.write(
                        str(reactionWId) + '\t' + str(reactionUId) + '\t' + reaction_name + '\t' + str(
                            reaction_verified) + '\t' + str(reaction_geo) + '\t' +
                        str(reaction_followers) + '\t' + str(reaction_friends) + '\t' + str(
                            reaction_favourites) + '\t' + str(reaction_time_cha)
                        + '\t' + str(react_user_time_cha) + '\t' + str(user_lichang) + '\t' + str(replyTo) + '--->' + str(reactionUId) + '\n')
                f_w.close()
