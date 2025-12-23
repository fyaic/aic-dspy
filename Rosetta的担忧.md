# DSPy 优化性能与超时处理指南

## 1. 优化过程会花费很久吗？

**简短回答**：是的，可能需要几分钟到几十分钟，具体取决于优化器类型、数据集大小和模型响应速度。

### 影响因素
1.  **优化器类型 (Optimizer Type)**
    *   **BootstrapFewShot**: 速度**最快**。它只是从你的训练集中挑选几个最好的例子作为 Few-shot。通常只需几分钟。
    *   **MIPRO / COPRO**: 速度**较慢**。它们需要多次调用模型来生成新的 Instruction（指令）并进行评估打分，涉及大量的 LLM API 调用。
2.  **数据集大小 (Dataset Size)**
    *   DSPy 建议训练集在 30~300 条左右。数据越多，评估一轮的时间越长。
    *   **技巧**：如果你只是想快速验证，可以限制 `max_bootstrapped_demos` 或使用 `trainset[:10]` 这样的小切片。
3.  **模型并发数 (Concurrency)**
    *   DSPy 内部通过多线程并行调用 LLM API。如果你的 API Key 有速率限制（Rate Limit，如每分钟只能请求 60 次），那么优化过程会被强制拖慢。

### 典型耗时参考
*   **简单优化 (BootstrapFewShot)**: 10~50 个样本，约 **2-5 分钟**。
*   **深度优化 (MIPRO/COPRO)**: 50+ 样本，多轮迭代，可能需要 **15-45 分钟**，甚至更久。
*   **成本**：一次典型的优化运行可能消耗 $0.1 ~ $2.0 USD 的 Token 费用（取决于模型价格）。

---

## 2. 如何解决超时与网络问题？

由于优化过程需要发起成百上千次 API 请求，网络波动或 API 超时是常见问题。

### 解决方案 A：配置请求超时参数 (Timeout)

在配置 LM 时，可以直接传递 `request_timeout` 参数（单位：秒）。

```python
# 在 intro.py 或你的代码中
lm = dspy.LM(
    model='openai/deepseek-chat', 
    api_key=api_key, 
    api_base='https://api.deepseek.com',
    timeout=60,  # ⚡ 设置单次请求超时为 60 秒
    max_retries=3 # ⚡ 设置失败重试次数
)
```

### 解决方案 B：使用并行加速与容错

在调用编译方法（compile）时，可以通过 `eval_kwargs` 控制并发数。减少并发数可以降低被 API 限流的风险，虽然会变慢，但更稳定。

```python
# 优化器配置
teleprompter = dspy.BootstrapFewShot(metric=your_metric, ...)

# 编译时传入参数
compiled_program = teleprompter.compile(
    student=your_program,
    trainset=trainset,
    # 👇 关键配置
    eval_kwargs={
        "num_threads": 4,       # 降低并发数 (默认可能是 8 或 16)
        "display_progress": True # 显示进度条
    }
)
```

### 解决方案 C：缓存机制 (Cache)

DSPy 默认开启了缓存。如果你运行一半报错了，**不要担心重头再来**。
DSPy 会把已经跑过的 API 结果缓存在本地（通常在 `~/.dspy_cache/` 或内存中）。
*   **好处**：再次运行代码时，已经成功的请求会直接从缓存读取，瞬间完成，直接跳到断点处继续。
*   **注意**：如果你修改了代码逻辑想强制重跑，可能需要手动清除缓存。

### 解决方案 D：处理 "Straggler" (拖后腿的任务)

对于极个别卡死的请求，DSPy 的并行执行器 (`ParallelExecutor`) 有一定的自动重试机制。
如果在某些极端网络环境下（如国内连接 OpenAI/DeepSeek 有时不稳定），建议：
1.  确保使用了稳定的代理或中转 API。
2.  在 `.env` 中配置好 HTTP_PROXY / HTTPS_PROXY。

---

## 3. 总结建议

1.  **从小开始**：先用 10 条数据跑通流程，确认没问题了再上全量数据。
2.  **用便宜模型优化**：可以用 `gpt-4o-mini` 或 `deepseek-chat` 这种便宜快速的模型来进行优化（作为 Teacher），得到 Prompt 后，再部署到更强的模型上（或者反过来）。
3.  **相信缓存**：遇到超时报错直接 `Ctrl+C` 停掉，调整参数（如减少线程数）后重新运行，它会接着跑，不会浪费之前的 Token。
