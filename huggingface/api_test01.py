"""
Hugging Face 模型使用示例
使用 transformers 库直接加载模型（推荐方式）
"""

from transformers import pipeline

print("=" * 60)
print("方式1: 使用 transformers pipeline（推荐）")
print("=" * 60)

# 文本生成
print("\n1️⃣ 文本生成示例 (GPT-2)")
generator = pipeline('text-generation', model='gpt2')
result = generator(
    "Explain the theory of relativity in simple terms.",
    max_length=100,
    num_return_sequences=1
)
print(f"生成的文本: {result[0]['generated_text']}")

# 对话模型 - 使用 text2text-generation
print("\n2️⃣ 对话模型示例 (FLAN-T5)")
conversational = pipeline('text2text-generation', model='google/flan-t5-small')
result = conversational("Answer this question: What is artificial intelligence?")
print(f"回复: {result[0]['generated_text']}")

# 翻译
print("\n3️⃣ 翻译示例 (英译法)")
translator = pipeline('translation_en_to_fr', model='t5-small')
result = translator("Hello, how are you?")
print(f"翻译结果: {result[0]['translation_text']}")

print("\n" + "=" * 60)
print("✅ 所有测试完成！")
print("=" * 60)