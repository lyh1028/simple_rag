import getpass
import os
#使用langSmith追踪过程, 注意langSmith只有免费5k次调用/月
os.environ["LANGSMITH_TRACING"] = "true"
os.environ["LANGSMITH_API_KEY"] = 'xxx'
os.environ['ZHIPUAI_API_KEY'] = 'xxx.xxx'
from langchain_community.chat_models import ChatZhipuAI
from langchain_community.embeddings import ZhipuAIEmbeddings
import logging
import bs4
from langchain import hub
from langchain_community.document_loaders.base import BaseLoader
from langchain_core.documents import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langgraph.graph import START, StateGraph
from typing_extensions import List, TypedDict, Tuple, Annotated
from langchain_community.document_loaders.parsers import GrobidParser
from langchain_community.document_loaders.generic import GenericLoader
import hashlib
from langchain_community.vectorstores import FAISS
from langchain.retrievers.multi_vector import MultiVectorRetriever
from langgraph.graph import START, MessagesState, StateGraph, END
from typing import Annotated, TypedDict, Sequence
from langgraph.graph.message import add_messages
from langchain_core.messages import BaseMessage, HumanMessage, SystemMessage, AIMessage, ToolMessage
from prompts import judge_prompt, rag_prompt, direct_prompt, choose_tool_prompt
from langchain_community.document_transformers import LongContextReorder
from functools import partial
from langgraph.prebuilt import ToolNode, tools_condition
from custom_tools.multi_query import *
from custom_tools.rag_fusion import *
from custom_tools.decomposition import *
from langchain_core.tools import Tool, InjectedToolArg, tool
from langchain_core.runnables import RunnableLambda, Runnable
from langchain_core.vectorstores import VectorStoreRetriever
from typing import Callable
from custom_tools.common import docs2str, fill_messages
from copy import deepcopy
from uuid import uuid4
from langgraph.checkpoint.memory import MemorySaver
# 设置日志级别
logging.basicConfig(level=logging.INFO)


gen_model = ChatZhipuAI(
    model="glm-4-plus",
)
embedding_model = ZhipuAIEmbeddings(
    model = 'embedding-3',
    dimensions=2048
)

pdf_dir = '/Users/liyihang/Documents/大模型+微调'

loader = GenericLoader.from_filesystem(
            pdf_dir,
            glob="*",
            suffixes=[".pdf"],
            parser= GrobidParser(segment_sentences=False)
        )

def generate_docs(loader:BaseLoader, min_doc_size:int=30)->Tuple[List[Document], List[Document]]:
    docs = loader.load()
    filter_docs = [doc for doc in docs if len(doc.page_content.split(' '))<min_doc_size]
    # 提取docs metadata中的abstract， 作为一级摘要文档，方便查询，metadata为paper_title, pub_time
    prev_paper_title = ''
    abstract_docs = []
    for doc in docs:
        if doc.metadata['paper_title'] != prev_paper_title:
            #新的文章，提取abstract
            prev_paper_title = doc.metadata['paper_title']
            asb_doc = Document(
                page_content=prev_paper_title + '\n' + doc.page_content,  # 文本内容
                metadata={"paper_title": prev_paper_title, "pub_time":doc.metadata['pub_time']}  # 元数据
            )
            abstract_docs.append(asb_doc)
        else:
            continue  
    return filter_docs, abstract_docs

def split_docs(filter_docs:List[Document], chunk_size:int, overlap_size:int, single_max_chunk:int):
    text_splitter = RecursiveCharacterTextSplitter.from_tiktoken_encoder(encoding_name='cl100k_base', chunk_size=chunk_size, chunk_overlap=overlap_size)
    all_splits = text_splitter.split_documents(filter_docs)
    def split_array(arr:list[Document], single_array_size:int)->list[list[Document]]:
        # 使用列表切片按 single_array_size 拆分list
        # 因为zhipuAI embedding在调用add_documents时最大允许的文档数是64，所以要分批add
        return [arr[i:i + single_array_size] for i in range(0, len(arr), single_array_size)]
    return split_array(all_splits, 64)

def calculate_md5_from_filepaths(folder_path):
    """
    根据文件夹下所有文件的相对路径计算 MD5 码
    :param folder_path: 文件夹路径
    :return: MD5 码（字符串）
    """
    filepaths = []
    for root, dirs, files in os.walk(folder_path):
        for file in files:
            # 获取文件的相对路径
            relative_path = os.path.relpath(os.path.join(root, file), folder_path)
            filepaths.append(relative_path)
    
    # 按字母顺序排序文件路径
    filepaths.sort()
    
    # 将所有文件路径拼接成一个字符串
    filepaths_str = ''.join(filepaths)
    
    # 计算 MD5 值
    md5_hash = hashlib.md5(filepaths_str.encode('utf-8')).hexdigest()
    
    return md5_hash

def get_faiss_index(splits_docs, md5_value, index_path, embedding_model, index_name='faiss_index'):
    FAISS_INDEX_PATH = f"{index_path}/{md5_value}_{index_name}"
    #print(f"开始创建向量数据库：{FAISS_INDEX_PATH}")
    if os.path.exists(FAISS_INDEX_PATH):
        # 加载已存在的向量数据库
        vectordb = FAISS.load_local(
            FAISS_INDEX_PATH,
            embedding_model,
            allow_dangerous_deserialization=True  # 明确允许反序列化
        )
        print(f"已加载本地向量数据库{index_name}")
    else:
        #index
        for i, docs in enumerate(splits_docs):
            if i == 0:
                vectordb = FAISS.from_documents(documents=docs, embedding=embedding_model)
            else:
                vectordb.add_documents(documents=docs)
        vectordb.save_local(FAISS_INDEX_PATH)
    return vectordb


retriever_store = {}
def gen_title_filter(candidate_titles:Sequence[str]):
    return {"paper_title":{"$in":candidate_titles}}

@tool(parse_docstring=True)
def multi_query_tool(question:str, retriever_id: Annotated[str, InjectedToolArg]):
    '''为了缓解余弦相似度查询的局限性，生成多个语义类似但措辞不同的查询，以获取更全面的上下文

    Args:
        question (str): 用户提出的问题
        retriever_id (Annotated[str, InjectedToolArg]): 查询工具
    '''
    # 从全局字典中获取 retriever
    retriever = retriever_store.get(retriever_id)
    if retriever is None:
        raise ValueError(f"Invalid retriever_id: {retriever_id}")
    query_search_func = create_multi_query_chain(retriever)
    if isinstance(query_search_func, Runnable):
        return create_multi_query_chain(retriever).invoke({"question":question})
    elif isinstance(query_search_func, Callable):
        return query_search_func(question)
    else:
        raise NotImplementedError

@tool(parse_docstring=True)
def rag_fusion_query_tool(question:str, retriever_id: Annotated[str, InjectedToolArg]):
    '''生成关于用户问题的**多个方面**的查询，以提供对原始问题的**不同角度**的检索结果，把这些结果和原始查询一起进行倒数排序融合，最终返回查询到的Document。

    Args:
        question (str): 用户提出的问题
        retriever_id (Annotated[str, InjectedToolArg]): 查询工具
    '''
    # 从全局字典中获取 retriever
    retriever = retriever_store.get(retriever_id)
    if retriever is None:
        raise ValueError(f"Invalid retriever_id: {retriever_id}")
    query_search_func = create_abstract_query_chain(retriever)
    if isinstance(query_search_func, Runnable):
        return create_abstract_query_chain(retriever).invoke({"question":question})
    elif isinstance(query_search_func, Callable):
        return query_search_func(question)
    else:
        raise NotImplementedError

@tool(parse_docstring=True)
def decomposition_query_tool(question:str, retriever_id: Annotated[str, InjectedToolArg]):
    '''对用户提出的问题进行分解，生成多个子问题，然后分别查询，最后将查询结果进行融合。

    Args:
        question (str): 用户提出的问题
        retriever_id (Annotated[str, InjectedToolArg]): 查询工具
    '''
    # 从全局字典中获取 retriever
    retriever = retriever_store.get(retriever_id)
    if retriever is None:
        raise ValueError(f"Invalid retriever_id: {retriever_id}")
    query_search_func = create_decomposition_search(retriever)
    if isinstance(query_search_func, Runnable):
        return create_decomposition_search(retriever).invoke({"question":question})
    elif isinstance(query_search_func, Callable):
        return query_search_func(question)
    else:
        raise NotImplementedError

@tool(parse_docstring=True)
def create_abstract_query_tool(question:str, retriever: Annotated[VectorStoreRetriever, InjectedToolArg]):
    '''生成关于用户问题的**多个方面**的查询，以提供对原始问题的**不同角度**的检索结果，把这些结果和原始查询一起进行倒数排序融合，最终返回查询到的Document。

    Args:
        question (str): 用户提出的问题
        retriever (Annotated[VectorStoreRetriever, InjectedToolArg]): 查询工具
    '''
    query_search_func = create_abstract_query_chain(retriever)
    if isinstance(query_search_func, Runnable):
        return create_abstract_query_chain(retriever).invoke({"question":question})
    elif isinstance(query_search_func, Callable):
        return query_search_func(question)
    else:
        raise NotImplementedError
    

# ===== 状态定义 =====
class State(TypedDict):
    messages: Annotated[Sequence[BaseMessage], add_messages]
    needs_rag: bool  # 标记是否需要RAG
    query: str      # 存储检索内容
    cur_node: str     # 记录当前节点名称
    title_list: list[str] # 和当前问题相关的论文标题列表
    search_type: str # 检索类型
    search_kwargs: dict # 检索参数
# ===== llm定义 =====
judge_llm = ChatZhipuAI(model="glm-4-plus")
answer_llm = ChatZhipuAI(model="glm-4-plus")
tooluse_llm = ChatZhipuAI(model="glm-4-plus")
tools = [multi_query_tool, rag_fusion_query_tool, decomposition_query_tool]
tooluse_llm = tooluse_llm.bind_tools(tools)

def inject_retriever(ai_msg, vectordb, search_type:str, search_kwargs:dict):
    tool_calls = []
    for tool_call in ai_msg.tool_calls:
        tool_call_copy = deepcopy(tool_call)
        # 生成唯一的 retriever_id
        retriever_id = str(uuid4())
        # 创建 retriever 并存储
        retriever = vectordb.as_retriever(search_type=search_type, search_kwargs=search_kwargs)
        retriever_store[retriever_id] = retriever
        # 将 retriever_id 注入参数
        tool_call_copy["args"]["retriever_id"] = retriever_id
        tool_calls.append(tool_call_copy)
    ai_msg.tool_calls = tool_calls
    return ai_msg

# ===== 节点函数 =====
def judge_node(state: State):
    """判断是否需要RAG"""
    # 构造判断提示
    prompt = judge_prompt.format_messages(messages=state["messages"])
    response = judge_llm.invoke(prompt).content
    #state["needs_rag"] = True if "Y" in response.upper() else False
    need_rag = True if "Y" in response.upper() else False
    return {"needs_rag": need_rag, "cur_node":"judge_node"}

def locate_paper(state:State):
    abstract_retreiver = abstract_vectordb.as_retriever(search_type = state.get("search_type", "similarity_score_threshold"), search_kwargs = state.get("search_kwargs", {'k':5, 'score_threshold':0.3}))
    abstract_docs = create_abstract_query_tool.invoke({"question":"deepseek论文中都有哪些创新点?", "retriever":abstract_retreiver})
    paper_list = [doc[0].metadata['paper_title'] for doc in abstract_docs]
    print("检索到相关论文：", paper_list)
    return {"title_list":paper_list, "cur_node":"locate_paper"}

def select_retrieve_node(state: State):
    """模拟检索上下文（实际应替换为真实检索逻辑）"""
    print("\n=== 深入检索pdf... ===")
    prompt = choose_tool_prompt.format_messages(question=state["messages"][-1].content)
    response = tooluse_llm.invoke(prompt)
    return {'messages':[response], "query":state["messages"][-1].content, "cur_node":"select_retrieve_node"}

def inject_tools(state: State):
    """注入工具"""
    title_filter = gen_title_filter(state["title_list"])
    tool_msg = inject_retriever(ai_msg=state["messages"][-1], vectordb=content_vectordb, search_type=state.get("search_type", "similarity"), search_kwargs=state.get("search_kwargs", {'k':5, 'fetch_k':50, 'filter':title_filter}))
    return {'messages':[tool_msg], "cur_node":"inject_tools"}

def answer_with_rag(state: State):
    """使用RAG生成回答"""
    prompt = rag_prompt.format_messages(
        context=state["messages"][-1].content,
        messages=fill_messages(content=state["query"], role="user")
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

retrieve_tool_node = ToolNode(tools=tools)

# ===== 构建工作流 =====
workflow = StateGraph(State)

# 添加节点
workflow.add_node("check_exit", judge_node)
workflow.add_node("locate_paper", locate_paper)
workflow.add_node("select_retrieve", select_retrieve_node)
workflow.add_node("tools", retrieve_tool_node)
workflow.add_node("answer_rag", answer_with_rag)
workflow.add_node("answer_direct", answer_directly)
workflow.add_node("output", output_node)
workflow.add_node("inject_tools", inject_tools)
# 设置入口节点
workflow.set_entry_point("check_exit")

# 设置边
workflow.add_conditional_edges(
    "check_exit",
    lambda state: END if "再见！" in str(state["messages"][-1]) else ("locate_paper" if state["needs_rag"] else "answer_direct")
)
workflow.add_edge("locate_paper", "select_retrieve")
workflow.add_edge("select_retrieve", "inject_tools")
workflow.add_conditional_edges("inject_tools", tools_condition)
workflow.add_edge("tools", "answer_rag")
workflow.add_edge("answer_rag", "output")
workflow.add_edge("answer_direct", "output")
workflow.add_edge("output", END)

# 持久化记忆
memory = MemorySaver()

# ===== 对话循环 =====
def chat_loop(app):
    print("论文助手：您好！我是学术论文阅读助手，请输入您的问题（输入'退出'或'exit'结束）")
    
    thread_id = "user_123"  # 用户唯一标识，用于保存和恢复对话内容
    while True:
        user_input = input("\n用户：")
        
        # 初始化或加载对话状态
        config = {"configurable": {"thread_id": thread_id}}
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
        image.save('graph.png')  # 替换为你想保存的路径

    except Exception as e:
        # 处理可能出现的错误（例如缺少依赖）
        print(f"An error occurred: {e}")

if __name__ == "__main__":
    md5_value = calculate_md5_from_filepaths(pdf_dir)
    filter_docs, abstract_docs = generate_docs(loader, 30)
    filter_split_docs = split_docs(filter_docs, chunk_size=520, overlap_size=32, single_max_chunk=64)
    abstract_split_docs = split_docs(abstract_docs, chunk_size=520, overlap_size=32, single_max_chunk=64)
    content_vectordb = get_faiss_index(filter_split_docs, md5_value=md5_value, index_path='/Users/liyihang/code/langchain_study/rag_agent_simple/faiss_db', embedding_model=embedding_model)
    abstract_vectordb = get_faiss_index(abstract_split_docs,md5_value=md5_value, index_path='/Users/liyihang/code/langchain_study/rag_agent_simple/faiss_db', embedding_model=embedding_model, index_name='abstract_faiss_index')
    app = workflow.compile(checkpointer=memory)
    gen_graph_img(app)
    thread_id = "user123"
    config = {"configurable": {"thread_id": thread_id}}
    while True:
        user_input = input("\n用户：")
        
        # 初始化或加载对话状态
        config = {"configurable": {"thread_id": thread_id}}
        # 调用工作流
        for event in app.stream(
            {"messages": [HumanMessage(user_input)]},
            config=config,
            stream_mode="values"
        ):
            if "cur_node" in event.keys() and (event["cur_node"] == "end"):
                event["messages"][-1].pretty_print()