"""
题目: 基于历史窗口的TF-IDF新闻检索系统

核心概念:
1. 历史窗口: 只考虑查询时间点t之前的最近K篇文档
2. 动态权重: 窗口内第j篇文档权重 = (K - j + 1) / K
3. TF-IDF向量化:
   - 查询向量: qi = TF(wi, q) × IDF(wi)
   - 文档向量: di = TF(wi, doc) × IDF(wi) × weight
4. 余弦相似度筛选: 返回相似度>=0.6且最高的文档编号,否则返回-1

输入格式:
- 第1行: 文档总数N
- 第2~N+1行: 各文档内容
- 第N+2行: 窗口大小K
- 第N+3行: 查询次数P
- 第N+4~N+3+P行: 每行为"时间点t 查询内容q"
"""

import math
from collections import Counter
from typing import List, Dict, Tuple


def tokenize(text: str) -> List[str]:
    """分词：简单按空格分割"""
    return text.strip().split()


def calculate_tf(words: List[str]) -> Dict[str, float]:
    """
    计算词频TF
    TF(word) = word在文档中出现次数 / 文档总词数
    """
    if not words:
        return {}
    
    word_count = Counter(words)
    total_words = len(words)
    
    return {word: count / total_words for word, count in word_count.items()}


def calculate_idf(documents: List[List[str]]) -> Dict[str, float]:
    """
    计算逆文档频率IDF
    IDF(word) = log(文档总数 / 包含该词的文档数)
    """
    if not documents:
        return {}
    
    num_docs = len(documents)
    word_doc_count = Counter()
    
    # 统计每个词出现在多少个文档中
    for doc in documents:
        unique_words = set(doc)
        for word in unique_words:
            word_doc_count[word] += 1
    
    # 计算IDF
    idf = {}
    for word, doc_count in word_doc_count.items():
        idf[word] = math.log(num_docs / doc_count)
    
    return idf


def calculate_tfidf_vector(tf: Dict[str, float], idf: Dict[str, float], 
                           vocabulary: set, weight: float = 1.0) -> Dict[str, float]:
    """
    计算TF-IDF向量
    
    参数:
        tf: 词频字典
        idf: 逆文档频率字典
        vocabulary: 词汇表
        weight: 文档权重（用于窗口内文档）
    
    返回:
        TF-IDF向量字典
    """
    vector = {}
    for word in vocabulary:
        tf_value = tf.get(word, 0.0)
        idf_value = idf.get(word, 0.0)
        vector[word] = tf_value * idf_value * weight
    
    return vector


def cosine_similarity(vec_a: Dict[str, float], vec_b: Dict[str, float]) -> float:
    """
    计算两个向量的余弦相似度
    cos(A, B) = A·B / (||A|| ||B||)
    """
    # 计算点积
    dot_product = sum(vec_a.get(word, 0) * vec_b.get(word, 0) for word in set(vec_a) | set(vec_b))
    
    # 计算向量模长
    norm_a = math.sqrt(sum(v ** 2 for v in vec_a.values()))
    norm_b = math.sqrt(sum(v ** 2 for v in vec_b.values()))
    
    # 避免除零
    if norm_a == 0 or norm_b == 0:
        return 0.0
    
    return dot_product / (norm_a * norm_b)


def search_in_window(query: str, time_point: int, window_size: int, 
                     documents: List[str]) -> int:
    """
    在历史窗口中搜索最相关的文档
    
    参数:
        query: 查询内容
        time_point: 查询时间点t
        window_size: 窗口大小K
        documents: 所有文档列表
    
    返回:
        最相关文档的编号（相似度>=0.6），否则返回-1
    """
    # 确定窗口范围：时间点t之前的K篇文档
    # 窗口为 [max(0, t-K+1), t]，即文档编号从 t-K+1 到 t
    window_start = max(0, time_point - window_size + 1)
    window_end = time_point
    
    # 如果窗口为空（时间点为0且窗口大小为1）
    if window_start > window_end:
        return -1
    
    # 提取窗口内的文档
    window_docs = documents[window_start:window_end + 1]
    window_doc_indices = list(range(window_start, window_end + 1))
    
    # 分词
    query_words = tokenize(query)
    window_docs_words = [tokenize(doc) for doc in window_docs]
    
    # 构建词汇表（窗口内所有文档+查询）
    vocabulary = set(query_words)
    for doc_words in window_docs_words:
        vocabulary.update(doc_words)
    
    # 计算窗口内文档的IDF
    idf = calculate_idf(window_docs_words)
    
    # 计算查询向量的TF
    query_tf = calculate_tf(query_words)
    
    # 计算查询向量（权重为1）
    query_vector = calculate_tfidf_vector(query_tf, idf, vocabulary, weight=1.0)
    
    # 计算窗口内每篇文档的TF-IDF向量并计算相似度
    best_similarity = -1.0
    best_doc_idx = -1
    
    for i, doc_words in enumerate(window_docs_words):
        # 计算文档在窗口中的位置（从1开始）
        position_in_window = i + 1
        num_docs_in_window = len(window_docs_words)
        
        # 动态权重：第j篇文档权重 = (K - j + 1) / K
        # position_in_window = j，所以 weight = (K - j + 1) / K
        weight = (num_docs_in_window - position_in_window + 1) / num_docs_in_window
        
        # 计算文档TF
        doc_tf = calculate_tf(doc_words)
        
        # 计算文档向量
        doc_vector = calculate_tfidf_vector(doc_tf, idf, vocabulary, weight)
        
        # 计算余弦相似度
        similarity = cosine_similarity(query_vector, doc_vector)
        
        # 更新最佳结果（相似度>=0.6，且更高；相同时保留最早的）
        if similarity >= 0.6:
            if similarity > best_similarity:
                best_similarity = similarity
                best_doc_idx = window_doc_indices[i]
    
    return best_doc_idx


def main():
    """主函数"""
    # 读取文档总数
    N = int(input().strip())
    
    # 读取所有文档
    documents = []
    for _ in range(N):
        documents.append(input().strip())
    
    # 读取窗口大小
    K = int(input().strip())
    
    # 读取查询次数
    P = int(input().strip())
    
    # 处理每个查询
    results = []
    for _ in range(P):
        line = input().strip()
        parts = line.split(maxsplit=1)
        t = int(parts[0])
        q = parts[1] if len(parts) > 1 else ""
        
        # 在窗口中搜索
        result = search_in_window(q, t, K, documents)
        results.append(result)
    
    # 输出结果
    for result in results:
        print(result)


if __name__ == '__main__':
    main()


