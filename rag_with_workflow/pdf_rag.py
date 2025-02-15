from langchain_community.chat_models import ChatZhipuAI
from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.messages import HumanMessage
from langgraph.checkpoint.memory import MemorySaver
from langgraph.graph import START, MessagesState, StateGraph, END
from typing import Annotated, TypedDict, Sequence
from langgraph.graph.message import add_messages
from langchain_core.messages import BaseMessage, HumanMessage, SystemMessage, AIMessage
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langgraph.checkpoint.memory import MemorySaver
from gen_retrieve import retriever
from langchain_community.document_transformers import LongContextReorder
from functools import partial
from env_var import *
import ipdb
# ===== 状态定义 =====
class State(TypedDict):
    messages: Annotated[Sequence[BaseMessage], add_messages]
    needs_rag: bool  # 标记是否需要RAG
    context: str      # 存储检索结果
    cur_node: str     # 记录当前节点名称

# ===== 组件定义 =====
# 判断是否需要RAG的提示模板
judge_prompt = ChatPromptTemplate.from_messages([
    ("system", """你是一个论文阅读助手。请分析用户问题：
1. 如果问题仅凭你的已有知识无法作答，需要参考文献或其他信息 → 输出Y
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

# 模型初始化
judge_llm = ChatZhipuAI(
    model="glm-4-plus",
    temperature=0,
)

if "glm" in answer_llm_name:
    answer_llm = ChatZhipuAI(
        model=answer_llm_name,
        temperature=0.1,
    )
else:
    answer_llm = ChatOpenAI(
        model=answer_llm_name,
        temperature=0.1,
    )   

# ===== 节点函数 =====
def judge_node(state: State):
    """判断是否需要RAG"""
    # 构造判断提示
    prompt = judge_prompt.format_messages(messages=state["messages"])
    response = judge_llm.invoke(prompt).content
    #state["needs_rag"] = True if "Y" in response.upper() else False
    need_rag = True if "Y" in response.upper() else False
    return {"needs_rag": need_rag, "cur_node":"judge_node"}


def retriever_node_wrapper(retriever):
    reordering = LongContextReorder()
    def retrieve_node(state: State, reordering):
        """模拟检索上下文（实际应替换为真实检索逻辑）"""
        print("\n=== 已有知识无法回答，正在检索pdf... ===")
        docs = retriever.invoke(state['messages'][-1].content)
        reordered_docs = reordering.transform_documents(docs)
        retrieve_context = ""
        for doc in reordered_docs:
            retrieve_context += doc.page_content
        return {'context':retrieve_context, "cur_node":"retrieve_node"}
    return partial(retrieve_node, reordering=reordering)

def answer_with_rag(state: State):
    """使用RAG生成回答"""
    prompt = rag_prompt.format_messages(
        context=state["context"],
        messages=state["messages"]
    )
    response = answer_llm.invoke(prompt)
    return {"messages": [AIMessage(response.content)], "cur_node":"answer_with_rag"}

def answer_directly(state: State):
    """直接生成回答"""
    prompt = direct_prompt.format_messages(messages=state["messages"])
    response = answer_llm.invoke(prompt)
    return {"messages": [AIMessage(response.content)], "cur_node":"answer_directly"}

def output_node(state: State):
    return {"cur_node":"end"}
# ===== 构建工作流 =====
workflow = StateGraph(State)

# 添加节点
retrieve_node = retriever_node_wrapper(retriever)
workflow.add_node("check_exit", judge_node)
workflow.add_node("retrieve", retrieve_node)
workflow.add_node("answer_rag", answer_with_rag)
workflow.add_node("answer_direct", answer_directly)
workflow.add_node("output", output_node)

# 设置边
workflow.add_edge(START, "check_exit")
#workflow.set_entry_point("check_exit")
# 条件分支
workflow.add_conditional_edges(
    "check_exit",
    lambda state: "output" if "再见！" in str(state["messages"][-1]) else "retrieve" if state["needs_rag"] else "answer_direct"
)

workflow.add_edge("retrieve", "answer_rag")
workflow.add_edge("answer_rag", "output")
workflow.add_edge("answer_direct", "output")
workflow.add_edge("output", END)

# 持久化记忆
memory = MemorySaver()


# ===== 对话循环 =====
def chat_loop():
    print("论文助手：您好！我是学术论文阅读助手，请输入您的问题（输入'退出'或'exit'结束）")
    
    thread_id = "user_123"  # 实际应使用用户唯一标识
    while True:
        user_input = input("\n用户：")
        
        # 初始化或加载对话状态
        config = {"configurable": {"thread_id": thread_id}}
        if user_input.strip().lower() == "退出" or user_input.strip().lower() == "exit":
            #app.invoke({"messages": [HumanMessage("退出")]}, config)
            print("论文助手：再见！")
            break
        else:
            # 调用工作流
            for event in app.stream(
                {"messages": [HumanMessage(user_input)]},
                config=config,
                stream_mode="values"
            ):
                if "cur_node" in event.keys() and (event["cur_node"] == "end"):
                    event["messages"][-1].pretty_print()
                
from IPython.display import Image, display
from PIL import Image as PILImage
from io import BytesIO

def gen_graph_img(app):
    try:
        # 获取图片的二进制数据
        png_data = app.get_graph().draw_mermaid_png()
        
        # 将二进制数据转换为PIL图像对象
        image = PILImage.open(BytesIO(png_data))
        
        # 保存图像到指定路径
        image.save(img_save_path)  # 替换为你想保存的路径

    except Exception as e:
        # 处理可能出现的错误（例如缺少依赖）
        print(f"An error occurred: {e}")

if __name__ == "__main__":
    # ===== 编译执行 =====
    app = workflow.compile(checkpointer=memory)
    gen_graph_img(app)
    chat_loop()
