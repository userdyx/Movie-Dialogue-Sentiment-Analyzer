# -*- coding: utf-8 -*-
from flask import Flask, request, jsonify, render_template
import torch
import numpy as np
import pickle
import re
from model import LSTMModel
import os
import string

# 扩展的英文停用词列表
STOPWORDS = {
    'i', 'me', 'my', 'myself', 'we', 'our', 'ours', 'ourselves', 'you', "you're", "you've", "you'll", "you'd", 'your', 
    'yours', 'yourself', 'yourselves', 'he', 'him', 'his', 'himself', 'she', "she's", 'her', 'hers', 'herself', 
    'it', "it's", 'its', 'itself', 'they', 'them', 'their', 'theirs', 'themselves', 'what', 'which', 'who', 'whom', 
    'this', 'that', "that'll", 'these', 'those', 'am', 'is', 'are', 'was', 'were', 'be', 'been', 'being', 
    'have', 'has', 'had', 'having', 'do', 'does', 'did', 'doing', 'a', 'an', 'the', 'and', 'but', 'if', 'or', 
    'because', 'as', 'until', 'while', 'of', 'at', 'by', 'for', 'with', 'about', 'against', 'between', 'into', 
    'through', 'during', 'before', 'after', 'above', 'below', 'to', 'from', 'up', 'down', 'in', 'out', 'on', 'off', 
    'over', 'under', 'again', 'further', 'then', 'once', 'here', 'there', 'when', 'where', 'why', 'how', 'all', 'any', 
    'both', 'each', 'few', 'more', 'most', 'other', 'some', 'such', 'no', 'nor', 'not', 'only', 'own', 'same', 'so', 
    'than', 'too', 'very', 's', 't', 'can', 'will', 'just', 'don', "don't", 'should', "should've", 'now', 'd', 'll', 
    'm', 'o', 're', 've', 'y', 'ain', 'aren', "aren't", 'couldn', "couldn't", 'didn', "didn't", 'doesn', "doesn't", 
    'hadn', "hadn't", 'hasn', "hasn't", 'haven', "haven't", 'isn', "isn't", 'ma', 'mightn', "mightn't", 'mustn', 
    "mustn't", 'needn', "needn't", 'shan', "shan't", 'shouldn', "shouldn't", 'wasn', "wasn't", 'weren', "weren't", 
    'won', "won't", 'wouldn', "wouldn't", 'would', 'could', "it'd", "he'd", "she'd", "we'd", "they'd",
    "i'm", "he's", "we're", "they're", "i've", "we've", "they've", "i'd", "that's", "there's",
    'however', 'although', 'moreover', 'nevertheless', 'nonetheless', 'therefore', 'otherwise',
    'anyway', 'besides', 'hence', 'thus', 'accordingly', 'consequently',
}

# 积极情感词典
POSITIVE_WORDS = {
    'good', 'great', 'excellent', 'amazing', 'wonderful', 'fantastic', 'terrific', 'outstanding',
    'superb', 'brilliant', 'awesome', 'fabulous', 'perfect', 'beautiful', 'best', 'better',
    'pleasant', 'enjoyable', 'delightful', 'happy', 'love', 'loved', 'loving', 'incredible',
    'impressive', 'exceptional', 'marvelous', 'splendid', 'spectacular', 'remarkable',
    'cool', 'sweet', 'nice', 'worth', 'worthy', 'recommend', 'recommended', 'recommending',
    'well', 'entertaining', 'fun', 'exciting', 'masterpiece', 'genius', 'hit', 'success', 'successful',
    'top', 'solid', 'satisfying', 'smart', 'clever', 'intelligent', 'innovative', 'creative',
    'favorite', 'favourite', 'win', 'winner', 'winning', 'delight', 'delightful', 'joy', 'joyful',
    'pleased', 'pleasing', 'pleasure', 'interesting', 'interested', 'engaging', 'engaged', 'authentic',
    'rich', 'emotionally', 'touching', 'touched', 'liked', 'like', 'loves', 'lovely', 'praise',
    'praised', 'praising', 'enjoyment', 'enjoy', 'enjoyed', 'enjoying', 'brilliant'
}

# 消极情感词典
NEGATIVE_WORDS = {
    'bad', 'terrible', 'horrible', 'awful', 'disappointing', 'disappointment', 'disappointed',
    'poor', 'weak', 'boring', 'bored', 'annoying', 'annoyed', 'irritating', 'irritated',
    'waste', 'wasted', 'wasting', 'hate', 'hated', 'hating', 'dislike', 'disliked', 'disliking',
    'worst', 'worse', 'stupid', 'dumb', 'idiotic', 'ridiculous', 'absurd', 'pathetic',
    'mediocre', 'bland', 'dull', 'tedious', 'tiresome', 'slow', 'confusing', 'confused',
    'mess', 'messy', 'failure', 'failed', 'failing', 'flop', 'shallow', 'empty', 'hollow',
    'cliché', 'clichéd', 'predictable', 'forgettable', 'forgotten', 'cheap', 'amateur',
    'amateurish', 'low', 'lowest', 'lame', 'lousy', 'sloppy', 'lazy', 'badly', 'poorly',
    'terrible', 'horrific', 'disaster', 'disastrous', 'tragic', 'unwatchable', 'unbearable',
    'unbearably', 'pretentious', 'pretentiously', 'frustrating', 'frustrated', 'frustrated',
    'irritates', 'irritated', 'irritating', 'offensive', 'offended', 'offending', 'appalling',
    'appalled', 'appalls', 'insulting', 'insulted', 'insults', 'abysmal', 'atrocious', 'pointless'
}

app = Flask(__name__)

# 全局变量
model = None
word_to_idx = None
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
MAX_LENGTH = 100

# 简单的词干提取函数
def simple_stemmer(word):
    """简单的英文词干提取函数"""
    # 去除ing结尾（如果单词长度>5）
    if len(word) > 5 and word.endswith('ing'):
        # 如果去掉ing后有元音，就保留去掉ing的形式
        if any(vowel in word[:-3] for vowel in 'aeiou'):
            if word.endswith('ying'):  # studying -> study
                return word[:-4] + 'y'
            elif word[-4] == word[-5]:  # running -> run
                return word[:-4]
            else:
                return word[:-3]
    
    # 去除ed结尾（如果单词长度>4）
    if len(word) > 4 and word.endswith('ed'):
        # 如果去掉ed后有元音，就保留去掉ed的形式
        if any(vowel in word[:-2] for vowel in 'aeiou'):
            if word[-3] == word[-4] and word[-3] not in 'aeiou':  # stopped -> stop
                return word[:-3]
            elif word.endswith('ied'):  # tried -> try
                return word[:-3] + 'y'
            else:
                return word[:-2]
    
    # 去除s结尾（如果单词长度>3）
    if len(word) > 3 and word.endswith('s') and not word.endswith('ss'):
        if word.endswith('ies'):  # parties -> party
            return word[:-3] + 'y'
        else:
            return word[:-1]
            
    # 去除ly结尾（如果单词长度>4）
    if len(word) > 4 and word.endswith('ly'):
        return word[:-2]
    
    # 去除est结尾（最高级形式）
    if len(word) > 5 and word.endswith('est'):
        if word[-4] == word[-5]:  # biggest -> big
            return word[:-4]
        return word[:-3]
    
    # 去除er结尾（比较级形式）
    if len(word) > 4 and word.endswith('er'):
        if word[-3] == word[-4]:  # bigger -> big
            return word[:-3]
        return word[:-2]
    
    return word

def load_model():
    """加载训练好的模型和词汇表"""
    global model, word_to_idx
    
    try:
        # 加载词汇表
        with open('word_to_idx.pkl', 'rb') as f:
            word_to_idx = pickle.load(f)
            
        vocab_size = len(word_to_idx)
        
        # 创建模型 - 使用与训练时相同的参数
        embedding_dim = 128
        hidden_dim = 128
        output_dim = 1
        n_layers = 2
        bidirectional = False  # 修改为单向LSTM
        dropout = 0.5
        
        model = LSTMModel(
            vocab_size, 
            embedding_dim, 
            hidden_dim, 
            output_dim, 
            n_layers, 
            bidirectional, 
            dropout
        )
        
        # 加载模型权重
        model.load_state_dict(torch.load('best_model.pt', map_location=device))
        model.to(device)
        model.eval()
        
        print("模型和词汇表加载成功！")
        return True
        
    except Exception as e:
        print(f"加载模型或词汇表失败: {e}")
        return False

def clean_text(text):
    """增强版的文本清洗函数"""
    # 去除HTML标签
    text = re.sub(r'<.*?>', ' ', text)
    # 去除URL
    text = re.sub(r'https?://\S+|www\.\S+', ' ', text)
    # 处理缩写形式
    text = re.sub(r"n't", " not", text)
    text = re.sub(r"'ve", " have", text)
    text = re.sub(r"'re", " are", text)
    text = re.sub(r"'ll", " will", text)
    text = re.sub(r"'d", " would", text)
    text = re.sub(r"'s", " is", text)
    text = re.sub(r"'m", " am", text)
    # 去除数字和标点符号，但保留括号内的标点（因为可能是表情符号）
    text = re.sub(r'[^a-zA-Z\s(),:.!?]', ' ', text)
    # 处理表情符号
    text = re.sub(r':\)', ' happy ', text)
    text = re.sub(r':\(', ' sad ', text)
    text = re.sub(r';\)', ' wink ', text)
    text = re.sub(r':D', ' laugh ', text)
    text = re.sub(r':\/', ' confused ', text)
    # 去除非字母字符
    text = re.sub(r'[^a-zA-Z\s]', ' ', text)
    # 转换为小写
    text = text.lower()
    # 去除多余的空格
    text = re.sub(r'\s+', ' ', text).strip()
    return text

def advanced_tokenize(text):
    """增强版分词：结合简单词干提取和情感词识别"""
    # 使用空格分词
    words = text.split()
    # 处理后的词汇列表
    processed_words = []
    
    for word in words:
        # 跳过停用词和单个字符的词
        if word in STOPWORDS or len(word) <= 1:
            continue
        
        # 应用简单的词干提取
        stemmed_word = simple_stemmer(word)
        
        # 如果是情感词，添加一个标记
        if stemmed_word in POSITIVE_WORDS:
            processed_words.append(stemmed_word)
            # 添加额外权重（通过重复添加）
            processed_words.append(stemmed_word)
        elif stemmed_word in NEGATIVE_WORDS:
            processed_words.append(stemmed_word)
            # 添加额外权重（通过重复添加）
            processed_words.append(stemmed_word)
        else:
            processed_words.append(stemmed_word)
    
    return processed_words

def text_to_sequence(text, max_length=MAX_LENGTH):
    """将文本转换为序列"""
    # 清洗文本
    text = clean_text(text)
    
    # 增强版分词
    words = advanced_tokenize(text)
    
    # 转换为索引
    seq = [word_to_idx.get(word, word_to_idx.get('<UNK>', 1)) for word in words]
    
    # 截断或填充序列
    if len(seq) > max_length:
        seq = seq[:max_length]
    else:
        seq = seq + [word_to_idx.get('<PAD>', 0)] * (max_length - len(seq))
    
    return seq

def predict_sentiment(text):
    """预测文本的情感"""
    if model is None or word_to_idx is None:
        if not load_model():
            return {"error": "模型未加载"}
    
    # 检查单词级别的极性判断（快速路径）
    clean_text_sample = clean_text(text)
    words = clean_text_sample.split()
    stemmed_words = [simple_stemmer(word) for word in words if word not in STOPWORDS and len(word) > 1]
    
    # 找出积极和消极词
    positive_found = [word for word in stemmed_words if word in POSITIVE_WORDS]
    negative_found = [word for word in stemmed_words if word in NEGATIVE_WORDS]
    
    # 特殊情况：非常短的输入（只有1-3个单词）且包含明确的消极词
    if len(stemmed_words) <= 3 and len(negative_found) > 0 and len(positive_found) == 0:
        # 这是一个简短的、只包含消极词的输入，直接判定为消极
        probability = 0.1  # 设置较低的概率值表示高度消极
        sentiment = "Negative"
        intensity = "Strong"
        
        return {
            "text": text,
            "sentiment": sentiment,
            "intensity": intensity,
            "probability": round(1 - probability, 4),  # 用1减去，因为这里表示消极的概率
            "raw_score": round(probability, 4),
            "positive_words": positive_found[:5],
            "negative_words": negative_found[:5],
            "method": "rule_based"  # 标记这是基于规则的判断
        }
    
    # 特殊情况：非常短的输入（只有1-3个单词）且包含明确的积极词
    if len(stemmed_words) <= 3 and len(positive_found) > 0 and len(negative_found) == 0:
        # 这是一个简短的、只包含积极词的输入，直接判定为积极
        probability = 0.9  # 设置较高的概率值表示高度积极
        sentiment = "Positive"
        intensity = "Strong"
        
        return {
            "text": text,
            "sentiment": sentiment,
            "intensity": intensity,
            "probability": round(probability, 4),
            "raw_score": round(probability, 4),
            "positive_words": positive_found[:5],
            "negative_words": negative_found[:5],
            "method": "rule_based"  # 标记这是基于规则的判断
        }
    
    # 标准模型路径：使用LSTM模型进行分析
    # 转换为序列
    seq = text_to_sequence(text)
    
    # 转换为张量
    seq_tensor = torch.tensor([seq], dtype=torch.long).to(device)
    
    # 预测
    with torch.no_grad():
        prediction = model(seq_tensor).squeeze(1)
        probability = torch.sigmoid(prediction).item()
    
    # 对非常短的文本进行修正
    # 如果短文本包含消极词，但模型预测为积极，给予更多权重给消极词
    if len(stemmed_words) <= 5 and len(negative_found) > 0 and probability > 0.5:
        # 根据消极词的数量调整概率
        adjustment = min(0.5, 0.15 * len(negative_found))
        probability -= adjustment
    
    # 同样，如果短文本包含积极词，但模型预测为消极，给予更多权重给积极词
    if len(stemmed_words) <= 5 and len(positive_found) > 0 and probability < 0.5:
        # 根据积极词的数量调整概率
        adjustment = min(0.5, 0.15 * len(positive_found))
        probability += adjustment
    
    # 进行情感分析
    sentiment = "Positive" if probability > 0.5 else "Negative"
    
    # 计算情感强度
    intensity = "Strong" if abs(probability - 0.5) > 0.45 else "Medium" if abs(probability - 0.5) > 0.25 else "Mild"
    
    return {
        "text": text,
        "sentiment": sentiment,
        "intensity": intensity,
        "probability": round(probability, 4) if probability > 0.5 else round(1 - probability, 4),
        "raw_score": round(probability, 4),
        "positive_words": positive_found[:5],  # 最多返回5个积极词
        "negative_words": negative_found[:5],   # 最多返回5个消极词
        "method": "model_based"  # 标记这是基于模型的判断
    }

@app.route('/')
def index():
    """首页"""
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    """预测API"""
    # 从请求中获取文本
    data = request.get_json()
    text = data.get('text', '')
    
    if not text:
        return jsonify({"error": "Please provide text"})
    
    # 预测情感
    result = predict_sentiment(text)
    
    return jsonify(result)

@app.route('/batch_predict', methods=['POST'])
def batch_predict():
    """批量预测API"""
    # 从请求中获取文本列表
    data = request.get_json()
    texts = data.get('texts', [])
    
    if not texts:
        return jsonify({"error": "Please provide a list of texts"})
    
    # 批量预测情感
    results = [predict_sentiment(text) for text in texts]
    
    return jsonify({"results": results})

# 创建templates目录和index.html
if not os.path.exists('templates'):
    os.makedirs('templates')

# 只有当index.html不存在时才创建
if not os.path.exists('templates/index.html'):
    with open('templates/index.html', 'w', encoding='utf-8') as f:
        f.write("""
<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>IMDB Sentiment Analysis</title>
    <style>
        body {
            font-family: Arial, sans-serif;
            max-width: 800px;
            margin: 0 auto;
            padding: 20px;
            line-height: 1.6;
            background-color: #f5f5f5;
        }
        .container {
            background-color: white;
            border-radius: 8px;
            padding: 20px;
            margin-bottom: 20px;
            box-shadow: 0 2px 10px rgba(0,0,0,0.1);
        }
        h1 {
            color: #333;
            text-align: center;
            margin-bottom: 30px;
        }
        textarea {
            width: 100%;
            padding: 12px;
            border: 1px solid #ddd;
            border-radius: 4px;
            height: 150px;
            margin-bottom: 15px;
            font-family: inherit;
            font-size: 16px;
        }
        button {
            background-color: #4CAF50;
            color: white;
            border: none;
            padding: 12px 20px;
            border-radius: 4px;
            cursor: pointer;
            font-size: 16px;
            transition: background-color 0.3s;
        }
        button:hover {
            background-color: #45a049;
        }
        .result {
            margin-top: 20px;
            padding: 15px;
            border-radius: 4px;
            display: none;
        }
        .positive {
            background-color: #dff0d8;
            border: 1px solid #d6e9c6;
            color: #3c763d;
        }
        .negative {
            background-color: #f2dede;
            border: 1px solid #ebccd1;
            color: #a94442;
        }
        .loader {
            border: 4px solid #f3f3f3;
            border-top: 4px solid #3498db;
            border-radius: 50%;
            width: 20px;
            height: 20px;
            animation: spin 2s linear infinite;
            margin: 20px auto;
            display: none;
        }
        @keyframes spin {
            0% { transform: rotate(0deg); }
            100% { transform: rotate(360deg); }
        }
        .probability-bar {
            background-color: #eee;
            border-radius: 4px;
            height: 20px;
            width: 100%;
            margin-top: 10px;
        }
        .probability-fill {
            height: 100%;
            border-radius: 4px;
            width: 0%;
            transition: width 0.5s ease-in-out;
        }
        .positive-fill {
            background-color: #4CAF50;
        }
        .negative-fill {
            background-color: #f44336;
        }
        .key-words {
            margin-top: 15px;
        }
        .word-chip {
            display: inline-block;
            padding: 5px 10px;
            margin: 3px;
            border-radius: 20px;
            font-size: 14px;
        }
        .positive-word {
            background-color: rgba(76, 175, 80, 0.2);
            border: 1px solid rgba(76, 175, 80, 0.5);
        }
        .negative-word {
            background-color: rgba(244, 67, 54, 0.2);
            border: 1px solid rgba(244, 67, 54, 0.5);
        }
        .examples {
            margin-top: 20px;
        }
        .example-button {
            background-color: #f0f0f0;
            color: #333;
            border: 1px solid #ddd;
            padding: 8px 15px;
            margin: 5px;
            border-radius: 4px;
            cursor: pointer;
            font-size: 14px;
        }
        .example-button:hover {
            background-color: #e0e0e0;
        }
        .intensity-badge {
            display: inline-block;
            padding: 3px 8px;
            border-radius: 12px;
            font-size: 12px;
            margin-left: 10px;
        }
        .intensity-high {
            background-color: rgba(255, 87, 34, 0.2);
            color: #d84315;
        }
        .intensity-medium {
            background-color: rgba(255, 152, 0, 0.2);
            color: #ef6c00;
        }
        .intensity-low {
            background-color: rgba(255, 193, 7, 0.2);
            color: #ff8f00;
        }
    </style>
</head>
<body>
    <h1>IMDB Sentiment Analysis</h1>
    
    <div class="container">
        <h2>Enter a Movie Review</h2>
        <textarea id="review-text" placeholder="Enter a movie review in English..."></textarea>
        <div class="examples">
            <button class="example-button" onclick="fillExample('This movie was fantastic! The acting was superb and the plot was engaging from start to finish.')">Positive Example</button>
            <button class="example-button" onclick="fillExample('I was really disappointed with this film. The characters were poorly developed and the storyline was confusing.')">Negative Example</button>
            <button class="example-button" onclick="fillExample('This is a must-see film. The director has created a masterpiece that will be remembered for years to come. The performances were outstanding and the cinematography was breathtaking.')">Strong Positive Example</button>
            <button class="example-button" onclick="fillExample('bad')">Single Negative Word</button>
        </div>
        <button id="analyze-btn">Analyze Sentiment</button>
        <div class="loader" id="loader"></div>
        
        <div class="result" id="result">
            <h3>Sentiment Analysis Result: <span id="sentiment-result"></span><span id="intensity-badge" class="intensity-badge"></span></h3>
            <p>Confidence: <span id="probability"></span>%</p>
            <div class="probability-bar">
                <div class="probability-fill" id="probability-fill"></div>
            </div>
            
            <div class="key-words">
                <div id="positive-words">
                    <h4>Positive Key Words:</h4>
                    <div id="positive-words-list"></div>
                </div>
                <div id="negative-words">
                    <h4>Negative Key Words:</h4>
                    <div id="negative-words-list"></div>
                </div>
            </div>
        </div>
    </div>
    
    <div class="container">
        <h2>Instructions</h2>
        <p>This is an advanced movie review sentiment analysis tool based on LSTM deep learning model.</p>
        <p>It analyzes the sentiment of movie reviews and provides the following features:</p>
        <ul>
            <li>Sentiment orientation (positive/negative)</li>
            <li>Sentiment intensity (strong/medium/mild)</li>
            <li>Sentiment analysis confidence</li>
            <li>Key sentiment word identification</li>
        </ul>
        <p>Enter an English movie review and click "Analyze Sentiment".</p>
    </div>

    <script>
        function fillExample(text) {
            document.getElementById('review-text').value = text;
        }
    
        document.getElementById('analyze-btn').addEventListener('click', function() {
            const text = document.getElementById('review-text').value.trim();
            if (!text) {
                alert('Please enter a movie review');
                return;
            }
            
            // 显示加载器，隐藏结果
            document.getElementById('loader').style.display = 'block';
            document.getElementById('result').style.display = 'none';
            
            // 发送API请求
            fetch('/predict', {
                method: 'POST',
                headers: {
                    'Content-Type': 'application/json'
                },
                body: JSON.stringify({ text: text })
            })
            .then(response => response.json())
            .then(data => {
                // 隐藏加载器
                document.getElementById('loader').style.display = 'none';
                
                // 显示结果
                document.getElementById('result').style.display = 'block';
                document.getElementById('sentiment-result').textContent = data.sentiment;
                
                // 显示情感强度
                const intensityBadge = document.getElementById('intensity-badge');
                intensityBadge.textContent = data.intensity;
                if (data.intensity === 'Strong') {
                    intensityBadge.className = 'intensity-badge intensity-high';
                } else if (data.intensity === 'Medium') {
                    intensityBadge.className = 'intensity-badge intensity-medium';
                } else {
                    intensityBadge.className = 'intensity-badge intensity-low';
                }
                
                // 设置结果的样式
                const resultElement = document.getElementById('result');
                const probabilityValue = Math.round(data.probability * 100);
                document.getElementById('probability').textContent = probabilityValue;
                
                // 设置概率条
                const fillElement = document.getElementById('probability-fill');
                fillElement.style.width = probabilityValue + '%';
                
                if (data.sentiment === 'Positive') {
                    resultElement.className = 'result positive';
                    fillElement.className = 'probability-fill positive-fill';
                } else {
                    resultElement.className = 'result negative';
                    fillElement.className = 'probability-fill negative-fill';
                }
                
                // 显示关键词
                const positiveWordsList = document.getElementById('positive-words-list');
                const negativeWordsList = document.getElementById('negative-words-list');
                
                // 清空现有内容
                positiveWordsList.innerHTML = '';
                negativeWordsList.innerHTML = '';
                
                // 添加积极词
                if (data.positive_words && data.positive_words.length > 0) {
                    data.positive_words.forEach(word => {
                        const wordChip = document.createElement('span');
                        wordChip.className = 'word-chip positive-word';
                        wordChip.textContent = word;
                        positiveWordsList.appendChild(wordChip);
                    });
                } else {
                    positiveWordsList.textContent = 'None';
                }
                
                // 添加消极词
                if (data.negative_words && data.negative_words.length > 0) {
                    data.negative_words.forEach(word => {
                        const wordChip = document.createElement('span');
                        wordChip.className = 'word-chip negative-word';
                        wordChip.textContent = word;
                        negativeWordsList.appendChild(wordChip);
                    });
                } else {
                    negativeWordsList.textContent = 'None';
                }
            })
            .catch(error => {
                console.error('Error:', error);
                document.getElementById('loader').style.display = 'none';
                alert('Analysis error, please try again');
            });
        });
    </script>
</body>
</html>
    """)

if __name__ == '__main__':
    # 加载模型
    load_model()
    # 启动Flask应用
    app.run(debug=True, host='0.0.0.0') 