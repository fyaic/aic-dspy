import dspy
import os
from dotenv import load_dotenv

# 1. 加载环境变量 (Load environment variables)
load_dotenv()

# 优先读取 DEEPSEEK_API_KEY
api_key = os.getenv("DEEPSEEK_API_KEY")

if not api_key:
    print("\n⚠️  警告: 未在 .env 文件中找到 DEEPSEEK_API_KEY。")
    exit(1)

# 2. 配置语言模型 (Configure LM)
# 这里使用 OpenAI 兼容模式连接到 DeepSeek
lm = dspy.LM(
    model='openai/deepseek-chat', 
    api_key=api_key, 
    api_base='https://api.deepseek.com'
)
dspy.configure(lm=lm)

# =========================================================================
# 3. 定义签名 (Signature) - ⚠️ 用户主要修改区域
# =========================================================================
# 签名定义了“做什么”：输入是什么，输出是什么。
class BasicQA(dspy.Signature):
    """
    在这里用自然语言描述任务目标。
    例如：'根据常识回答用户的问题，答案要简短'
    """
    # 👇 定义输入字段 (Input): 告诉模型你需要它处理什么
    question = dspy.InputField(desc="用户提出的问题")
    
    # 👇 定义输出字段 (Output): 告诉模型你需要它生成什么
    answer = dspy.OutputField(desc="生成的回答，通常在1-5个词之间")

# 4. 创建模块 (Create Module)
# ChainOfThought (思维链) 会让模型在回答前先进行推理，通常效果更好
generate_answer = dspy.ChainOfThought(BasicQA)

# 5. 运行模块 (Run)
my_question = "What is the capital of France?"
print(f"正在提问: {my_question}")

response = generate_answer(question=my_question)

# 6. 显示结果 (Show Result)
if hasattr(response, 'reasoning'):
    print(f"推理过程: {response.reasoning}")
print(f"最终答案: {response.answer}")

# 7. 调试：查看发送给模型的实际 Prompt (Inspect)
print("\n--- 发送给模型的最后一条 Prompt ---")
lm.inspect_history(n=1)

# 8. [新增] 自动保存 Prompt 到文件，方便查看
try:
    with open("debug_prompt.txt", "w", encoding="utf-8") as f:
        # 获取最后一次交互的 messages (聊天模型) 或 prompt (补全模型)
        last_item = lm.history[-1]
        if 'messages' in last_item:
            for msg in last_item['messages']:
                f.write(f"[{msg['role'].upper()}]\n{msg['content']}\n\n{'='*20}\n\n")
        else:
            f.write(last_item.get('prompt', 'No prompt found'))
            
    print("\n✅ 提示词已保存到文件: debug_prompt.txt (请在左侧文件列表打开查看)")
except Exception as e:
    print(f"\n⚠️ 保存 Prompt 文件失败: {e}")
