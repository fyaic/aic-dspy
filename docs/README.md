# DSPy 介绍与使用指南

## 1. DSPy 是什么？

**DSPy** (Declarative Self-improving Language Programs) 是斯坦福大学开发的一个用于构建语言模型（LM）应用的框架。

它的核心理念是：**编程，而不是写提示词 (Programming, not Prompting)**。

在传统的开发中，你可能需要花费大量时间手动调整复杂的 Prompt（提示词字符串）。而在 DSPy 中，你通过编写 Python 代码来定义程序的逻辑（输入什么、输出什么），然后由 DSPy 的编译器（Optimizer）自动生成和优化最适合模型的 Prompt。

*   **项目地址**: [https://github.com/stanfordnlp/dspy]
*   Obsidian笔记：https://publish.obsidian.md/aic/%E6%96%AF%E5%9D%A6%E7%A6%8FDSPy+-+%E5%A3%B0%E6%98%8E%E5%BC%8F%E8%87%AA%E6%88%91%E4%BC%98%E5%8C%96python%EF%BC%88%E6%8C%87%E4%BB%A4%E5%B7%A5%E7%A8%8B%EF%BC%89

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


# 作者原文
**If you're looking to understand the framework, please go to the [DSPy Docs at dspy.ai](https://dspy.ai)**

&nbsp;

--------

&nbsp;

The content below is focused on how to modify the documentation site.

&nbsp;

# Modifying the DSPy Documentation


This website is built using [Material for MKDocs](https://squidfunk.github.io/mkdocs-material/), a Material UI inspired theme for MKDocs.

## Building docs locally

To build and test the documentation locally:

1. Navigate to the `docs` directory:
   ```bash
   cd docs
   ```

2. Install the necessary dependencies:
   ```bash
   pip install -r requirements.txt
   ```

3. In docs/ directory, run the command below to generate the API docs and index them:
   ```bash
   python scripts/generate_api_docs.py
   python scripts/generate_api_summary.py
   ```

4. (Optional) On MacOS you may also need to install libraries for building the site
   ```bash
   brew install cairo freetype libffi libjpeg libpng zlib
   export DYLD_FALLBACK_LIBRARY_PATH=/opt/homebrew/lib
   ```

5. Run the build command:
   ```bash
   mkdocs build
   ```

This will generate a static build of the documentation site in the `site` directory. You can then serve this directory to view the site locally using:

```bash
mkdocs serve
```

If you see the build failing make sure to fix it before pushing.

## Continuous Integration (CI) Build Checks

We have automated build checks set up in our CI pipeline to ensure the documentation builds successfully before merging changes. These checks:

1. Run the `mkdocs build` command
2. Verify that the build completes without errors
3. Help catch potential issues early in the development process

If the CI build check fails, please review your changes and ensure the documentation builds correctly locally before pushing updates.

## Contributing to the `docs` Folder

This guide is for contributors looking to make changes to the documentation in the `dspy/docs` folder. 

1. **Pull the up-to-date version of the website**: Please pull the latest version of the live documentation site via cloning the dspy repo.  The current docs are in the `dspy/docs` folder.

2. **Push your new changes on a new branch**: Feel free to add or edit existing documentation and open a PR for your changes. Once your PR is reviewed and approved, the changes will be ready to merge into main. 

3. **Updating the website**: Once your changes are merged to main, the changes would be reflected on live websites usually in 5-15 mins.

## LLMs.txt

The build process generates an `/llms.txt` file for LLM consumption using [mkdocs-llmstxt](https://github.com/pawamoy/mkdocs-llmstxt). Configure sections in `mkdocs.yml` under the `llmstxt` plugin.

