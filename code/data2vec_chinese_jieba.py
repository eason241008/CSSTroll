import torch
import jieba
from transformers import BertTokenizer, BertModel
from tqdm import tqdm
import os
import pandas as pd
import numpy as np

# 检测设备
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

# 文件路径
data_path = "/C24108/Wangsy/data/hupu_data.csv"
data_save_path = "/C24108/Wangsy/data2vec/"
X_w2v_file = os.path.join(data_save_path, "hupu_data_jieba.npy")
sentiment_dict_path = '/C24108/Wangsy/data/sentiment_dict.txt'

# 创建保存路径
if not os.path.exists(data_save_path):
    os.makedirs(data_save_path)

# 加载情感词典
def load_sentiment_dict(file_path):
    sentiment_dict = {}
    with open(file_path, 'r', encoding='utf-8') as f:
        for line in f:
            phrase, score = line.strip().split()
            sentiment_dict[phrase] = float(score)
    return sentiment_dict

sentiment_dict = load_sentiment_dict(sentiment_dict_path)

# 获取情感向量
def get_sentiment_vector(phrase):
    sentiment_score = sentiment_dict.get(phrase, None)
    if sentiment_score is None:
        return torch.zeros(3, device=device)  # 在 GPU 上创建全零向量

    if sentiment_score > 0:
        one_hot_vector = torch.tensor([1, 0, 0], dtype=torch.float32, device=device)  # 正面情感
    elif sentiment_score < -5:
        one_hot_vector = torch.tensor([0, 0, 1], dtype=torch.float32, device=device)  # 负面情感
    else:
        one_hot_vector = torch.tensor([0, 1, 0], dtype=torch.float32, device=device)  # 中性情感

    return one_hot_vector

# 加载 BERT 模型
model_path = "/C24108/Wangsy/chinese_bert"
tokenizer = BertTokenizer.from_pretrained(model_path)
bert_model = BertModel.from_pretrained(model_path).to(device)  # 将模型迁移到 GPU
print("BERT 模型加载成功")

# 文本转为 BERT 输入格式
def text_to_bert_input(text, tokenizer, max_length=128):
    inputs = tokenizer(text, return_tensors="pt", padding=True, truncation=True, max_length=max_length)
    return inputs.to(device)

# 获取短语的 BERT 向量
def phrase_vector_with_sentiment(phrase, model, tokenizer):
    inputs = text_to_bert_input(phrase, tokenizer)
    with torch.no_grad():
        outputs = model(**inputs)
    bert_vector = outputs.last_hidden_state.mean(dim=1).squeeze()  # 获取短语的 BERT 表示
    sentiment_vector = get_sentiment_vector(phrase)  # 获取情感向量
    combined_vector = torch.cat((bert_vector, sentiment_vector))  # 拼接 BERT 和情感向量
    return combined_vector

# 处理句子，将所有短语的组合向量拼接成句向量
def sentence_to_combined_vector(sentence, model, tokenizer):
    words = jieba.lcut(sentence)  # 使用 jieba 分词
    word_vectors = []
    
    for word in words:
        vector = phrase_vector_with_sentiment(word, model, tokenizer)
        word_vectors.append(vector)

    sentence_matrix = torch.stack(word_vectors)  # 将每个词的组合向量拼接成矩阵
    sentence_vector = sentence_matrix.mean(dim=0)  # 使用均值池化得到句子向量
    return sentence_vector

# 加载数据并处理每条数据
def process_and_save_data(data_path, model, tokenizer, save_path):
    data = pd.read_csv(data_path)
    sentences = data['reply'].dropna().tolist()  # 去掉空值，仅保留 reply 列
    
    # 存储所有句子向量
    all_sentence_vectors = []
    for sentence in tqdm(sentences, desc="Processing sentences"):
        sentence_vector = sentence_to_combined_vector(sentence, model, tokenizer)
        all_sentence_vectors.append(sentence_vector.cpu().numpy())  # 转回 CPU 保存
    
    # 转为 numpy 数组并保存
    all_sentence_vectors = np.array(all_sentence_vectors)
    np.save(save_path, all_sentence_vectors)
    print(f"所有数据已保存至 {save_path}")

# 执行数据处理和保存
process_and_save_data(data_path, bert_model, tokenizer, X_w2v_file)