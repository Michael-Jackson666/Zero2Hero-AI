'''答案'''

import sys, math
from collections import Counter, defaultdict

def tok(s:str):
    ''''将字符串转小写并按非字母数字切分为词'''
    s = s.lower()
    out, cur = [], []
    for ch in s:
        if ch.isalnum():
            cur.append(ch)
        else:
            if cur:
                out.append(''.join(cur))
                cur = []

    if cur:
        out.append(''.join(cur))
    return out

def read_input():
    data = sys.stdin.read().splitlines()
    it = iter(data)
    N = int(next(it).strip())
    docs_raw = []
    for _ in range(N):
        docs_raw.append(next(it) if True else "")
    K = int(next(it).strip())
    P = int(next(it).strip())
    queries = []
    for _ in range(P):
        line = next(it).rstrip('\n')
        line = line.strip()
        sp = line.find(' ')
        if sp == -1:
            t = int(line)
            q = ""
        else:
            t = int(line[:sp].strip())
            q = line[sp+1:].strip()
        queries.append((t,q))
    return N ,docs_raw, K , P, queries

def main():
    N, docs_raw, K, P, queries = read_input()

    # 预处理分词和词频
    doc_tokens = [tok(s) for s in docs_raw]
    doc_lens = [len(tks) for tks in doc_tokens]
    doc_cnt = [Counter(tks) for tks in doc_tokens]

    ans = []

    
    for t, qstr in queries:
        # 构造时间窗口 [L, R]
        R = min(max(0, t), N - 1)
        L = max(0, R - K + 1)
        M = R - L + 1 # 窗口文档数目
        
        q_words = tok(qstr)
        if not q_words or M <= 0:
            ans.append(-1)
            continue

        q_cnt = Counter(q_words)
        q_len = len(q_words)

        # 计算窗口内查询词的 df 与 idf
        df = defaultdict(int)
        q_set = set(q_cnt.keys())
        for i in range(L, R + 1):
            # 每篇文档只记一次（是否出现）
            for w in q_set:
                if doc_cnt[i].get(w, 0) > 0:
                    df[w] += 1
        idf = {}
        for w in q_set:
            idf[w] = math.log((M + 1.0) / (df[w] + 1.0)) + 1.0 # 平滑IDF

        # 预计算查询向量的范数平方
        q_norm_sq = 0.0
        for w, c in q_cnt.items():
            tfq = c / q_len
            x = tfq * idf[w]
            q_norm_sq += x * x
        q_norm = math.sqrt(q_norm_sq) if q_norm_sq > 0 else 1.0

        # 初始化最佳匹配: (相似度, 文档索引)
        best = (0.0, -1)
        
        # 从新到旧遍历窗口内的文档
        for idx in range(R, L - 1, -1):
            dl = doc_lens[idx] if doc_lens[idx] > 0 else 1
            
            # 计算文档在窗口中的动态权重
            # j表示倒数第几篇: 倒数第1篇(idx=R,最新), 倒数第M篇(idx=L,最旧)
            j = (R - idx) + 1
            # 窗口内第j篇文档的权重 = (M - j + 1) / M
            # j=1(最新,idx=R): weight = (M-1+1)/M = M/M = 1.0 ✓
            # j=M(最旧,idx=L): weight = (M-M+1)/M = 1/M ✓
            weight = (M - j + 1) / M
            
            dot = 0.0  # 错误1修复: 变量名拼写错误 doc -> dot
            d_norm_sq = 0.0  # 错误2修复: 变量名拼写错误 d_norw_sq -> d_norm_sq
            
            # 计算加权后的文档向量与查询向量的点积和范数
            # 根据题目: di = TF(wi, doc) × IDF(wi) × weight
            for w, qc in q_cnt.items():
                tfd = doc_cnt[idx].get(w, 0) / dl
                tfq = qc / q_len
                dq = tfq * idf[w]  # 查询向量在该维度的值
                dd = tfd * idf[w] * weight  # 错误4修复: 文档向量需要乘以权重
                dot += dq * dd
                d_norm_sq += dd * dd

            # 计算余弦相似度
            if d_norm_sq == 0.0:
                sim = 0.0
            else:
                d_norm = math.sqrt(d_norm_sq)
                sim = dot / (q_norm * d_norm)

            # 更新最佳匹配(相似度>=0.6)
            if sim >= 0.6 - 1e-12:
                # 如果当前相似度更高,或者相似度相同但文档索引更小(更早)
                if sim > best[0] + 1e-12:
                    best = (sim, idx)
                elif abs(sim - best[0]) <= 1e-12 and (best[1] == -1 or idx < best[1]):
                    # 相似度相同时,选择更早的文档
                    best = (sim, idx)

        ans.append(best[1] if best[1] != -1 else -1)

    print(' '.join(str(x) for x in ans))

if __name__ == '__main__':
    main()