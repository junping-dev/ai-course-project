# ai-course-project

# 📘 Text Sentiment Analysis with Self-Attention

## 📖 项目简介
本项目实现了一个基于 **IMDB Large Movie Review Dataset** 的文本情感分析模型。  
在传统的 RNN/LSTM 架构基础上，我们引入了 **Self-Attention 机制**（参考 *Attention is All You Need* 论文），以提升模型对长文本的理解能力，从而更好地区分电影评论中的 **正面** 与 **负面** 情感。

---
## 🗂 数据集
- **名称**: IMDB Large Movie Review Dataset  
- **规模**: 50,000 条电影评论（25,000 正面，25,000 负面）  
- **划分**:  
  - 训练集：25,000 条  
  - 测试集：25,000 条  
- **特点**: 预处理后，保留了标点和句子结构，适合 NLP 模型输入。  

下载地址：[IMDB Dataset](https://ai.stanford.edu/~amaas/data/sentiment/)

---

## ⚙️ 方法与模型设计
1. **文本表示**  
   - 使用 **词嵌入 (Word Embedding)** 将句子映射为向量序列。  
   - 可选：使用 **预训练词向量 (GloVe)** 初始化。  

2. **基线模型**  
   - LSTM 作为序列编码器。  
   - 最后一层隐藏状态用于情感分类。  

3. **改进点：Self-Attention 机制**  
   - 在序列表示中加入 **Scaled Dot-Product Attention**：  
     $$
     \text{Attention}(Q,K,V) = \text{softmax}\left(\frac{QK^T}{\sqrt{d_k}}\right)V
     $$
   - 允许模型在分类时关注关键情感词语，提升对长评论的建模能力。  

4. **分类层**  
   - 使用全连接层 + Sigmoid 激活完成二分类。  

5. **损失函数与优化器**  
   - Binary Cross Entropy (BCE)  
   - Adam 优化器  

---

## 🖥 环境依赖
- Python 3.9+  
- PyTorch 2.0+  
- NumPy  
- Scikit-learn  
- Matplotlib（绘制训练曲线）  
