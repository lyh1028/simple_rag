from langchain_core.prompts import ChatPromptTemplate, PromptTemplate, MessagesPlaceholder

choose_tool_prompt = ChatPromptTemplate.from_messages([
    ("system", '''选择你可以使用的自定义检索工具来处理用户的查询，以获取更好的查询结果。
     1. 对于与论文相关的复杂推理的问题，使用decompose_search分解成子问题
     2. 对于比较抽象或宽泛的问题，使用rag_fusion_search, 生成多方面的查询
     3. 对于较为具体的问题，使用multi_query_search，生成多个措辞不同的查询，或者更宏观的查询，尽量覆盖更多的信息'''),
    ("user", "使用工具回答问题：{question}， 如果找不到合适的工具，不要回答"),
])
# 判断是否需要RAG的提示模板
judge_prompt = ChatPromptTemplate.from_messages([
    ("system", """你是一个论文阅读助手。请分析用户问题：
1. 如果问题仅凭你的已有知识无法作答，需要参考文献或其他信息，或者用户明确要求进行检索、使用RAG功能等 → 输出Y
2. 如果是闲聊、问候、仅凭已有知识或上下文能准确作答 → 输出N

只需输出单个字母（Y/N）"""),
    MessagesPlaceholder(variable_name="messages"),
])

# RAG回答提示模板
rag_prompt = ChatPromptTemplate.from_messages([
    MessagesPlaceholder(variable_name="messages"),
    ("human", """基于已知信息和以下文本回答上面的问题：
{context}

请用学术语言回答。"""),
])

# 直接回答提示模板
direct_prompt = ChatPromptTemplate.from_messages([
    ("system", "你是一个友好的科研助手，回答用户问题时确保回答的科学性、严谨性和准确性。"),
    MessagesPlaceholder(variable_name="messages"),
])