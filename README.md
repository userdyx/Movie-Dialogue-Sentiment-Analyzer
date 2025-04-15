# 基于LSTM的电影评论情感分析

## 简介

这是一个基于LSTM（长短期记忆网络）的情感分析系统，旨在分析电影评论的情感倾向（积极或消极）。项目涵盖了数据预处理、模型训练以及Web应用的完整流程。

## 项目结构

- `data_preprocessing.py`：负责数据的清洗、分词、词汇表创建和序列转换。
- `app.py`：Flask Web应用，提供不依赖于NLTK的增强文本处理功能。
- `model.py`：定义和训练LSTM模型的代码。
- `templates/index.html`：Web应用的前端页面。
- `requirements.txt`：项目所需的依赖包列表。

## 技术栈

- **Python 3.6+**：项目的编程语言。
- **PyTorch**：用于深度学习的框架。
- **Flask**：用于构建Web应用的框架。
- **NumPy, Pandas**：用于数据处理。
- **Matplotlib, Seaborn**：用于数据可视化。

## 安装步骤

1. 克隆项目代码到本地。

2. 创建虚拟环境（推荐）
   ```
   python -m venv venv
   source venv/bin/activate  # Linux/Mac
   venv\Scripts\activate  # Windows
   ```

3. 安装项目依赖
   ```
   pip install -r requirements.txt
   ```

## 运行项目

您可以通过以下命令运行整个项目：

```
python run.py
```

或者分别运行每个步骤：

### 1. 数据预处理

```
python data_preprocessing.py
```

此步骤将执行以下操作：
- 加载或创建示例数据。
- 清洗文本数据，去除噪声。
- 对文本进行分词处理。
- 构建词汇表以便于后续处理。
- 将文本转换为数值序列。
- 保存处理后的数据。

### 2. 模型训练

```
python model.py
```

此步骤将执行以下操作：
- 加载预处理后的数据。
- 创建并训练LSTM模型。
- 评估模型在测试集上的性能。
- 保存训练好的模型。
- 生成训练曲线和混淆矩阵以供分析。

### 3. 启动Web应用

```
python app.py
```

然后在浏览器中访问 http://localhost:5000 使用情感分析的Web界面。

## 模型结构

该情感分析器的LSTM模型结构包括：
1. **嵌入层**：将词汇表中的单词映射到向量空间。
2. **LSTM层**：捕获文本的上下文信息，支持双向LSTM。
3. **全连接层**：将LSTM的输出映射到情感类别。

## 性能评估

模型在测试集上的性能指标包括：
- **准确率**：正确预测的比例。
- **精确率**：在所有预测为正的样本中，真正为正的比例。
- **召回率**：在所有真正为正的样本中，正确预测为正的比例。
- **F1分数**：精确率和召回率的调和平均数。

## Web应用

Web应用提供了一个直观的用户界面，用户可以输入电影评论，系统将实时分析并显示情感方向和置信度。应用还包括以下高级功能：

1. **情感强度分析**：分析情感的强、中、弱程度。
2. **关键情感词识别**：识别评论中的关键情感词。
3. **简单词干提取**：处理不同的词形变化。
4. **处理表情符号和常见缩写**：增强文本处理能力。

### API端点

1. **单条文本预测**：
   ```
   POST /predict
   Content-Type: application/json
   
   {
       "text": "这部电影太棒了！"
   }
   ```

2. **批量文本预测**：
   ```
   POST /batch_predict
   Content-Type: application/json
   
   {
       "texts": ["这部电影太棒了！", "这部电影太糟糕了。"]
   }
   ```

## 最近的改进

1. **增强文本处理**：增加了对缩写、表情符号和特殊字符的处理能力。
2. **情感分析增强**：增加了情感强度分析和关键情感词识别功能。
3. **用户界面改进**：重新设计了Web界面，使其更直观和信息丰富。
4. **英文文本支持**：所有输出、图表和界面元素现在都支持正确的英文显示。

## References

- IMDB 数据集来源:
  @InProceedings{maas-EtAl:2011:ACL-HLT2011,
  author    = {Maas, Andrew L.  and  Daly, Raymond E.  and  Pham, Peter T.  and  Huang, Dan  and  Ng, Andrew Y.  and  Potts, Christopher},
  title     = {Learning Word Vectors for Sentiment Analysis},
  booktitle = {Proceedings of the 49th Annual Meeting of the Association for Computational Linguistics: Human Language Technologies},
  month     = {June},
  year      = {2011},
  address   = {Portland, Oregon, USA},
  publisher = {Association for Computational Linguistics},
  pages     = {142--150},
  url       = {http://www.aclweb.org/anthology/P11-1015}
} 