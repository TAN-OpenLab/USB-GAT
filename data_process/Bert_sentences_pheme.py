import os
import torch
from transformers import BertTokenizer
from pytorch_pretrained_bert import BertModel

MODELNAME = 'bert-base-uncased'
tokenizer = BertTokenizer.from_pretrained(MODELNAME)  # 分词词
model = BertModel.from_pretrained(MODELNAME)  # 模型
model.eval()

url = r'D:\project\BiGCN-master\dataset\pheme-rnr-dataset'
datasets = ['germanwings-crash', 'charliehebdo', 'ferguson', 'ottawashooting', 'sydneysiege']  #
types = ['rumours', 'non-rumours']

if __name__ == '__main__':
    max_len = 0
    for dataset in datasets:
        print(dataset)
        for type in types:
            for _, _, filenames in os.walk(os.path.join(url, 'pro_content', dataset, type)):
                for file in filenames:
                    f = open(os.path.join(url, 'pro_content', dataset, type, file), 'r', encoding='utf-8')
                    f_w = open(os.path.join(url, 'pro_content_emb', dataset, type, str(file)), 'a')
                    for line in f.readlines():
                        line = line.strip('\n').split('\t')

                        with torch.no_grad():
                            input_ids = tokenizer.encode(
                                line[2],
                                add_special_tokens=True,  # 添加special tokens， 也就是CLS和SEP
                                # max_length=114,  # 设定最大文本长度
                                # padding = 'max_length',   # pad到最大的长度
                                return_tensors='pt'  # 返回的类型为pytorch tensor
                            )
                            encoded_layers, _ = model(input_ids)
                        sentence_vec = torch.mean(encoded_layers[11], 1).squeeze()
                        sentence_vec = sentence_vec.numpy().tolist()

                        input_ids = [str(x) for x in sentence_vec]
                        if len(input_ids) > max_len:
                            max_len = len(input_ids)
                        f_w.write(line[0] + '\t' + line[1] + '\t' + ' '.join(input_ids) + '\n')
            print('%s, %s, %d' % (dataset, type, max_len))
            max_len = 0





