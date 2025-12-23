# DSPy 介绍与使用指南

## 1. DSPy 是什么？

**DSPy** (Declarative Self-improving Language Programs) 是斯坦福大学开发的一个用于构建语言模型（LM）应用的框架。

它的核心理念是：**编程，而不是写提示词 (Programming, not Prompting)**。

在传统的开发中，你可能需要花费大量时间手动调整复杂的 Prompt（提示词字符串）。而在 DSPy 中，你通过编写 Python 代码来定义程序的逻辑（输入什么、输出什么），然后由 DSPy 的编译器（Optimizer）自动生成和优化最适合模型的 Prompt。

*   **项目地址**: [https://github.com/stanfordnlp/dspy]

## 2. 核心方法论

DSPy 将 LM 应用的开发拆解为三个部分：

1.  **签名 (Signatures)**：定义任务的输入和输出。告诉 DSPy “你要做什么”，而不是“怎么做”。
2.  **模块 (Modules)**：封装了特定的处理逻辑（如 `ChainOfThought` 思维链, `ReAct` 推理等）。
3.  **优化器 (Optimizers)**：这是 DSPy 的魔法所在。它能根据你提供的少量示例（Few-shot）或评估指标（Metric），自动“编译”你的代码，找出能让模型表现最好的 Prompt 组合。

## 3. 适用用户与场景

### 适用用户
*   **AI 工程师/开发者**：希望构建稳定、可维护的 AI 应用，而不是维护一堆脆弱的 Prompt 字符串。
*   **研究人员**：需要快速实验不同的模型和推理策略。
*   **想要“系统化”优化效果的人**：当你发现手动改 Prompt 已经无法提升效果时，DSPy 的自动优化能帮你突破瓶颈。

### 适用场景
*   **复杂的多步推理**：如“先搜索文档，再阅读，最后回答”（RAG 系统）。
*   **信息提取**：从非结构化文本中提取特定的结构化数据。
*   **文本分类与生成**：需要高准确率和一致性的任务。
*   **自动化 Prompt 优化**：当你有一个数据集，想让模型在这个数据集上表现达到最优。

---

## 4. 用户指南：你需要在哪里输入要求？

在 DSPy 代码中，你主要关注 **Signature (签名)** 的定义。这是你告诉模型“任务规则”的地方。

打开你的 Python 脚本（例如 `intro.py`），找到类似下面的类定义：

```python
# 👇 这里的类名可以修改，比如改为 "WriteEmail" 或 "ExtractInfo"
class BasicQA(dspy.Signature):
    """
    👇 在这里写任务描述 (Docstring)
    这是最重要的部分！告诉模型这个任务的背景和目标。
    例如："根据提供的上下文回答问题" 或 "将用户输入翻译成莎士比亚风格的英语"
    """
    
    # 👇 定义输入字段 (Input Fields)
    # 变量名即为输入名称，desc 是给模型的补充说明
    question = dspy.InputField(desc="用户提出的问题")
    # context = dspy.InputField(desc="相关的背景知识") # 如果需要更多输入，可以加在这里
    
    # 👇 定义输出字段 (Output Fields)
    # 告诉模型你需要什么格式的结果
    answer = dspy.OutputField(desc="简短的事实性回答，通常在1-5个词之间")
```

### 如何修改？

1.  **修改任务描述**：在 `"""..."""` 中用自然语言清晰地描述你的目标。
2.  **定义输入**：使用 `dspy.InputField()` 定义用户会提供什么信息（如 `topic`, `email_draft`, `query`）。
3.  **定义输出**：使用 `dspy.OutputField()` 定义你希望模型生成什么（如 `summary`, `polished_email`, `json_data`）。

### 示例：改为“写诗助手”

如果你想把上面的 QA 机器人改成写诗助手，只需修改 Signature：

```python
class PoemWriter(dspy.Signature):
    """根据给定的主题和风格写一首短诗。"""
    
    topic = dspy.InputField(desc="诗歌的主题")
    style = dspy.InputField(desc="诗歌的风格，例如：悲伤、幽默、古风")
    
    poem = dspy.OutputField(desc="生成的诗歌，包含4行")

# 使用时
generate_poem = dspy.Predict(PoemWriter)
response = generate_poem(topic="月亮", style="古风")
print(response.poem)
```
