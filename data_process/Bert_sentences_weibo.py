import torch
import os
from transformers import BertTokenizer
from pytorch_pretrained_bert import BertModel

MODELNAME = 'D:/zys/BiGCN/USB_GAT/bert-base-chinese'
tokenizer = BertTokenizer.from_pretrained(MODELNAME)  # 分词词
model = BertModel.from_pretrained(MODELNAME)  # 模型
model.eval()

url = r'D:\zys\pheme-rnr-dataset\rumdect'

if __name__ == '__main__':
    max_len = 0
    i = 0
    for _, _, filenames in os.walk(os.path.join(url, 'pro_text')):
        for file in filenames:
            print(str(i) + '\n')
            f = open(os.path.join(url, 'pro_text', file), 'r', encoding='utf-8')
            f_w = open(os.path.join(url, 'pro_text_emb_new', str(file)), 'a')
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
                sentence_vec = sentence_vec.detach().numpy().tolist()
                input_ids = [str(x) for x in sentence_vec]
                if len(input_ids) > max_len:
                    max_len = len(input_ids)
                f_w.write(line[0] + '\t' + line[1] + '\t' + ' '.join(input_ids) + '\n')
            i += 1
    print('%d', max_len)
    max_len = 0

