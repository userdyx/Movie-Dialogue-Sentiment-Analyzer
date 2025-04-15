# -*- coding: utf-8 -*-
import pandas as pd
import numpy as np
import re
import nltk
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
from nltk.stem import PorterStemmer, WordNetLemmatizer
from sklearn.model_selection import train_test_split
import pickle
from tqdm import tqdm
import os

# 下载必要的NLTK资源
try:
    nltk.data.find('tokenizers/punkt')
except LookupError:
    nltk.download('punkt')
try:
    nltk.data.find('corpora/stopwords')
except LookupError:
    nltk.download('stopwords')
try:
    nltk.data.find('corpora/wordnet')
except LookupError:
    nltk.download('wordnet')

# 加载英文停用词
stop_words = set(stopwords.words('english'))
lemmatizer = WordNetLemmatizer()

def clean_text(text):
    """清洗英文文本：去除标点符号、HTML标签等"""
    # 去除HTML标签
    text = re.sub(r'<.*?>', ' ', text)
    # 去除URL
    text = re.sub(r'https?://\S+|www\.\S+', ' ', text)
    # 去除数字和标点符号
    text = re.sub(r'[^a-zA-Z\s]', ' ', text)
    # 转换为小写
    text = text.lower()
    # 去除多余的空格
    text = re.sub(r'\s+', ' ', text).strip()
    return text

def tokenize(text):
    """分词：将英文文本分割成词汇列表"""
    # 简单分词方式，避免punkt_tab资源问题
    tokens = text.split()
    # 去除停用词并进行词形还原
    tokens = [lemmatizer.lemmatize(word) for word in tokens if word not in stop_words and len(word) > 1]
    return tokens

def build_vocab(texts, max_vocab_size=20000):
    """构建词汇表"""
    print("Building vocabulary...")
    word_counts = {}
    
    # 统计词频
    for text in tqdm(texts):
        for word in tokenize(text):
            if word in word_counts:
                word_counts[word] += 1
            else:
                word_counts[word] = 1
    
    # 按词频排序
    sorted_vocab = sorted(word_counts.items(), key=lambda x: x[1], reverse=True)
    
    # 截取最常用的词汇
    if max_vocab_size > 0 and len(sorted_vocab) > max_vocab_size:
        sorted_vocab = sorted_vocab[:max_vocab_size]
    
    # 创建词汇到索引的映射
    word_to_idx = {'<PAD>': 0, '<UNK>': 1}
    for i, (word, _) in enumerate(sorted_vocab):
        word_to_idx[word] = i + 2  # +2是因为已经有<PAD>和<UNK>
    
    print(f"Vocabulary size: {len(word_to_idx)}")
    return word_to_idx

def texts_to_sequences(texts, word_to_idx, max_length=100):
    """将文本转换为数字序列"""
    sequences = []
    
    for text in tqdm(texts, desc="Converting text to sequences"):
        words = tokenize(text)
        seq = [word_to_idx.get(word, word_to_idx['<UNK>']) for word in words]
        
        # 截断或填充序列
        if len(seq) > max_length:
            seq = seq[:max_length]
        else:
            seq = seq + [word_to_idx['<PAD>']] * (max_length - len(seq))
        
        sequences.append(seq)
    
    return np.array(sequences)

def load_imdb_data(file_path='IMDB Dataset.csv', limit=None):
    """加载IMDb电影评论数据
    
    Args:
        file_path: CSV文件路径
        limit: 如果指定，则只加载前limit条记录；如果为None，加载全部数据
    """
    try:
        # 加载CSV文件
        if limit is not None:
            df = pd.read_csv(file_path, nrows=limit)
            print(f"Data loaded successfully with limit {limit}, {len(df)} records in total")
        else:
            df = pd.read_csv(file_path)
            print(f"Complete dataset loaded successfully, {len(df)} records in total")
        
        # 确保列名正确
        if 'review' not in df.columns or 'sentiment' not in df.columns:
            print("Warning: Column names don't match, expected 'review' and 'sentiment'")
            if len(df.columns) == 2:
                df.columns = ['review', 'sentiment']
        
        # 将情感标签转换为数值 (0 = negative, 1 = positive)
        df['sentiment'] = df['sentiment'].apply(lambda x: 1 if x == 'positive' else 0)
        
        # 确保数据集中包含正负样本
        positive_count = sum(df['sentiment'] == 1)
        negative_count = sum(df['sentiment'] == 0)
        print(f"Positive reviews: {positive_count}, Negative reviews: {negative_count}")
        
        return df
    except Exception as e:
        print(f"Failed to load data: {e}")
        # 如果加载真实数据失败，创建一些示例数据
        print("Creating sample data...")
        # 创建一些正面评论
        positive_reviews = [
            "This movie was great! The acting was wonderful and the story was captivating.",
            "I really enjoyed this film. The plot was interesting and the characters were well-developed.",
            "This is one of the best movies I have seen in a long time.",
            "The directing was superb and the cinematography was beautiful.",
            "An excellent film with amazing performances from the entire cast."
        ] * 10  # 重复数据以增加数量
        
        # 创建一些负面评论
        negative_reviews = [
            "This movie was terrible. I wasted my time watching it.",
            "The plot was confusing and the acting was poor.",
            "I was very disappointed with this film and would not recommend it.",
            "This is one of the worst movies I have ever seen.",
            "The director clearly had no idea what they were doing, and the story made no sense."
        ] * 10  # 重复数据以增加数量
        
        # 合并数据
        reviews = positive_reviews + negative_reviews
        labels = [1] * len(positive_reviews) + [0] * len(negative_reviews)
        
        # 打乱数据
        indices = np.random.permutation(len(reviews))
        reviews = [reviews[i] for i in indices]
        labels = [labels[i] for i in indices]
        
        df = pd.DataFrame({
            'review': reviews,
            'sentiment': labels
        })
        
        return df

def prepare_data(file_path='IMDB Dataset.csv', test_size=0.2, val_size=0.1, max_length=100, max_vocab_size=20000, limit=None):
    """准备数据：加载、清洗、分词、转换序列"""
    # 加载数据
    df = load_imdb_data(file_path, limit=limit)
    
    # 清洗文本
    print("Cleaning text...")
    df['clean_review'] = df['review'].apply(clean_text)
    
    # 划分训练集、验证集和测试集
    train_df, test_df = train_test_split(df, test_size=test_size, random_state=42, stratify=df['sentiment'])
    train_df, val_df = train_test_split(train_df, test_size=val_size/(1-test_size), random_state=42, stratify=train_df['sentiment'])
    
    print(f"Training set size: {len(train_df)}")
    print(f"Validation set size: {len(val_df)}")
    print(f"Test set size: {len(test_df)}")
    
    # 构建词汇表
    word_to_idx = build_vocab(train_df['clean_review'], max_vocab_size)
    
    # 保存词汇表
    with open('word_to_idx.pkl', 'wb') as f:
        pickle.dump(word_to_idx, f)
    print("Vocabulary saved to word_to_idx.pkl")
    
    # 转换文本为序列
    X_train = texts_to_sequences(train_df['clean_review'], word_to_idx, max_length)
    X_val = texts_to_sequences(val_df['clean_review'], word_to_idx, max_length)
    X_test = texts_to_sequences(test_df['clean_review'], word_to_idx, max_length)
    
    # 获取标签
    y_train = train_df['sentiment'].values
    y_val = val_df['sentiment'].values
    y_test = test_df['sentiment'].values
    
    # 创建data目录
    os.makedirs('data', exist_ok=True)
    
    # 保存处理后的数据
    np.save('data/X_train.npy', X_train)
    np.save('data/y_train.npy', y_train)
    np.save('data/X_val.npy', X_val)
    np.save('data/y_val.npy', y_val)
    np.save('data/X_test.npy', X_test)
    np.save('data/y_test.npy', y_test)
    
    print("Preprocessing complete, data saved to data directory.")
    
    return X_train, y_train, X_val, y_val, X_test, y_test, word_to_idx

if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description='Data Preprocessing')
    parser.add_argument('--data_path', type=str, default='IMDB Dataset.csv', help='Data file path')
    parser.add_argument('--max_length', type=int, default=100, help='Maximum sequence length')
    parser.add_argument('--max_vocab_size', type=int, default=20000, help='Maximum vocabulary size')
    parser.add_argument('--sample_size', type=int, default=None, help='Sample size to use, default is None (all data)')
    
    args = parser.parse_args()
    
    prepare_data(
        file_path=args.data_path,
        max_length=args.max_length,
        max_vocab_size=args.max_vocab_size,
        limit=args.sample_size
    ) 