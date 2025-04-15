# ����LSTM�ĵ�Ӱ������з���

## ���

����һ������LSTM�������ڼ������磩����з���ϵͳ��ּ�ڷ�����Ӱ���۵�������򣨻���������������Ŀ����������Ԥ����ģ��ѵ���Լ�WebӦ�õ��������̡�

## ��Ŀ�ṹ

- `data_preprocessing.py`���������ݵ���ϴ���ִʡ��ʻ����������ת����
- `app.py`��Flask WebӦ�ã��ṩ��������NLTK����ǿ�ı������ܡ�
- `model.py`�������ѵ��LSTMģ�͵Ĵ��롣
- `templates/index.html`��WebӦ�õ�ǰ��ҳ�档
- `requirements.txt`����Ŀ������������б�

## ����ջ

- **Python 3.6+**����Ŀ�ı�����ԡ�
- **PyTorch**���������ѧϰ�Ŀ�ܡ�
- **Flask**�����ڹ���WebӦ�õĿ�ܡ�
- **NumPy, Pandas**���������ݴ���
- **Matplotlib, Seaborn**���������ݿ��ӻ���

## ��װ����

1. ��¡��Ŀ���뵽���ء�

2. �������⻷�����Ƽ���
   ```
   python -m venv venv
   source venv/bin/activate  # Linux/Mac
   venv\Scripts\activate  # Windows
   ```

3. ��װ��Ŀ����
   ```
   pip install -r requirements.txt
   ```

## ������Ŀ

������ͨ��������������������Ŀ��

```
python run.py
```

���߷ֱ�����ÿ�����裺

### 1. ����Ԥ����

```
python data_preprocessing.py
```

�˲��轫ִ�����²�����
- ���ػ򴴽�ʾ�����ݡ�
- ��ϴ�ı����ݣ�ȥ��������
- ���ı����зִʴ���
- �����ʻ���Ա��ں�������
- ���ı�ת��Ϊ��ֵ���С�
- ���洦�������ݡ�

### 2. ģ��ѵ��

```
python model.py
```

�˲��轫ִ�����²�����
- ����Ԥ���������ݡ�
- ������ѵ��LSTMģ�͡�
- ����ģ���ڲ��Լ��ϵ����ܡ�
- ����ѵ���õ�ģ�͡�
- ����ѵ�����ߺͻ��������Թ�������

### 3. ����WebӦ��

```
python app.py
```

Ȼ����������з��� http://localhost:5000 ʹ����з�����Web���档

## ģ�ͽṹ

����з�������LSTMģ�ͽṹ������
1. **Ƕ���**�����ʻ���еĵ���ӳ�䵽�����ռ䡣
2. **LSTM��**�������ı�����������Ϣ��֧��˫��LSTM��
3. **ȫ���Ӳ�**����LSTM�����ӳ�䵽������

## ��������

ģ���ڲ��Լ��ϵ�����ָ�������
- **׼ȷ��**����ȷԤ��ı�����
- **��ȷ��**��������Ԥ��Ϊ���������У�����Ϊ���ı�����
- **�ٻ���**������������Ϊ���������У���ȷԤ��Ϊ���ı�����
- **F1����**����ȷ�ʺ��ٻ��ʵĵ���ƽ������

## WebӦ��

WebӦ���ṩ��һ��ֱ�۵��û����棬�û����������Ӱ���ۣ�ϵͳ��ʵʱ��������ʾ��з�������Ŷȡ�Ӧ�û��������¸߼����ܣ�

1. **���ǿ�ȷ���**��������е�ǿ���С����̶ȡ�
2. **�ؼ���д�ʶ��**��ʶ�������еĹؼ���дʡ�
3. **�򵥴ʸ���ȡ**������ͬ�Ĵ��α仯��
4. **���������źͳ�����д**����ǿ�ı�����������

### API�˵�

1. **�����ı�Ԥ��**��
   ```
   POST /predict
   Content-Type: application/json
   
   {
       "text": "�ⲿ��Ӱ̫���ˣ�"
   }
   ```

2. **�����ı�Ԥ��**��
   ```
   POST /batch_predict
   Content-Type: application/json
   
   {
       "texts": ["�ⲿ��Ӱ̫���ˣ�", "�ⲿ��Ӱ̫����ˡ�"]
   }
   ```

## ����ĸĽ�

1. **��ǿ�ı�����**�������˶���д��������ź������ַ��Ĵ���������
2. **��з�����ǿ**�����������ǿ�ȷ����͹ؼ���д�ʶ���ܡ�
3. **�û�����Ľ�**�����������Web���棬ʹ���ֱ�ۺ���Ϣ�ḻ��
4. **Ӣ���ı�֧��**�����������ͼ��ͽ���Ԫ�����ڶ�֧����ȷ��Ӣ����ʾ��

## References

- IMDB ���ݼ���Դ:
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