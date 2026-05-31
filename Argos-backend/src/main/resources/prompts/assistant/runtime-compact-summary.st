你负责对一次模型调用前的运行时上下文做临时压缩（短期、调用级上下文，不是长期记忆）。

这个 prompt 只负责“当前调用要带什么上下文”，不负责写入或维护 session memory。

输入：
- compact summary：
【compact-summary-begin】
{compactSummary}
【compact-summary-end】
- session memory：
【session-memory-begin】
{sessionMemory}
【session-memory-end】
- 最近消息窗口：
【recent-messages-begin】
{recentMessages}
【recent-messages-end】
- 当前问题：
【current-question-begin】
{currentQuestion}
【current-question-end】

输出要求：
- 只输出本轮调用可直接使用的压缩上下文
- 优先保留当前问题直接相关的信息，以及解决当前问题必须依赖的少量背景
- 重点选择：compact summary 里的关键结论、session memory 里的关键事实、最近消息里与当前问题直接相关的几轮、工具返回的关键结果
- 主动丢弃：无关历史、重复叙述、已经沉淀进 session memory 但本轮无需展开的细节
- 不要输出解释、标题外包装或 Markdown 代码块
