"""
Hugging Face Transformers 完整使用示例
包含多种常见NLP任务的实现
"""

from transformers import pipeline
import warnings
warnings.filterwarnings('ignore')

print("🤗 Hugging Face Transformers 示例集合\n")

# ==================== 1. 文本生成 ====================
print("=" * 60)
print("1️⃣ 文本生成 (Text Generation)")
print("=" * 60)

generator = pipeline('text-generation', model='gpt2')
prompt = "The future of artificial intelligence is"
result = generator(
    prompt,
    max_length=80,
    num_return_sequences=2,
    temperature=0.8
)

for i, gen in enumerate(result, 1):
    print(f"\n生成 {i}:")
    print(gen['generated_text'])

# ==================== 2. 情感分析 ====================
print("\n" + "=" * 60)
print("2️⃣ 情感分析 (Sentiment Analysis)")
print("=" * 60)

sentiment = pipeline('sentiment-analysis')
texts = [
    "I love this product! It's amazing!",
    "This is terrible, worst experience ever.",
    "It's okay, nothing special."
]

for text in texts:
    result = sentiment(text)[0]
    print(f"\n文本: {text}")
    print(f"情感: {result['label']} (置信度: {result['score']:.2%})")

# ==================== 3. 问答系统 ====================
print("\n" + "=" * 60)
print("3️⃣ 问答系统 (Question Answering)")
print("=" * 60)

qa = pipeline('question-answering')
context = """
Hugging Face is a company that develops tools for building applications using 
machine learning. It is most notable for its Transformers library built for 
natural language processing applications and its platform that allows users to 
share machine learning models and datasets.
"""

questions = [
    "What is Hugging Face?",
    "What is the Transformers library used for?",
]

for question in questions:
    result = qa(question=question, context=context)
    print(f"\n问题: {question}")
    print(f"答案: {result['answer']}")
    print(f"置信度: {result['score']:.2%}")

# ==================== 4. 文本摘要 ====================
print("\n" + "=" * 60)
print("4️⃣ 文本摘要 (Summarization)")
print("=" * 60)

summarizer = pipeline('summarization', model='facebook/bart-large-cnn')
article = """
The tower is 324 metres (1,063 ft) tall, about the same height as an 81-storey 
building, and the tallest structure in Paris. Its base is square, measuring 
125 metres (410 ft) on each side. During its construction, the Eiffel Tower 
surpassed the Washington Monument to become the tallest man-made structure in 
the world, a title it held for 41 years until the Chrysler Building in New York 
City was finished in 1930.
"""

summary = summarizer(article, max_length=50, min_length=25, do_sample=False)
print(f"\n原文: {article[:100]}...")
print(f"\n摘要: {summary[0]['summary_text']}")

# ==================== 5. 命名实体识别 ====================
print("\n" + "=" * 60)
print("5️⃣ 命名实体识别 (Named Entity Recognition)")
print("=" * 60)

ner = pipeline('ner', grouped_entities=True)
text = "Apple Inc. was founded by Steve Jobs in Cupertino, California."

entities = ner(text)
print(f"\n文本: {text}\n")
print("识别的实体:")
for entity in entities:
    print(f"  - {entity['word']}: {entity['entity_group']} (置信度: {entity['score']:.2%})")

# ==================== 6. 翻译 ====================
print("\n" + "=" * 60)
print("6️⃣ 翻译 (Translation)")
print("=" * 60)

# 英译法
en_to_fr = pipeline('translation_en_to_fr', model='t5-small')
en_text = "Hello, how are you today?"
fr_result = en_to_fr(en_text)
print(f"\n英语 → 法语:")
print(f"  原文: {en_text}")
print(f"  译文: {fr_result[0]['translation_text']}")

# ==================== 7. 零样本分类 ====================
print("\n" + "=" * 60)
print("7️⃣ 零样本分类 (Zero-Shot Classification)")
print("=" * 60)

classifier = pipeline('zero-shot-classification')
text = "This is a tutorial about using transformers for NLP tasks."
candidate_labels = ['education', 'politics', 'sports', 'technology']

result = classifier(text, candidate_labels)
print(f"\n文本: {text}\n")
print("分类结果:")
for label, score in zip(result['labels'], result['scores']):
    print(f"  {label}: {score:.2%}")

# ==================== 8. 填充遮罩词 ====================
print("\n" + "=" * 60)
print("8️⃣ 填充遮罩词 (Fill-Mask)")
print("=" * 60)

fill_mask = pipeline('fill-mask')
text = "Artificial intelligence will <mask> the future of technology."

results = fill_mask(text, top_k=3)
print(f"\n句子: {text}\n")
print("可能的填充:")
for i, result in enumerate(results, 1):
    print(f"  {i}. {result['token_str']}: {result['score']:.2%}")

print("\n" + "=" * 60)
print("✅ 所有示例运行完成！")
print("=" * 60)

print("\n💡 提示:")
print("  - 首次运行会下载模型，需要一些时间")
print("  - 模型缓存在 ~/.cache/huggingface/")
print("  - 您的Mac支持MPS加速，训练会很快！")
print("  - 更多模型请访问: https://huggingface.co/models")
