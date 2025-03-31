from langchain.prompts import ChatPromptTemplate
import logging
# RAG-Fusion: Related
template = """针对下面的问题生成4个不同方面的查询，以提供对原始问题的不同角度的检索结果。 问题: {question} \n
直接分条返回用户初始查询和生成的其他查询: """
prompt_rag_fusion = ChatPromptTemplate.from_template(template)
from langchain_core.output_parsers import StrOutputParser
from langchain_community.chat_models import ChatZhipuAI
from langchain_core.runnables import RunnableLambda
from custom_tools.common import rerank_documents
from custom_tools.common import docs2str, LineListOutputParser, reciprocal_rank_fusion
gen_model = ChatZhipuAI(model="glm-4-plus", temperature=0.1)
# 添加打印生成的查询的中间步骤
print_queries = RunnableLambda(lambda x: (print("\n=== 生成的查询 ===", *x, sep="\n"), x)[1])
generate_queries = (
    prompt_rag_fusion 
    | gen_model
    | LineListOutputParser()
    | print_queries
)

def print_scores(docs_list):
    print("\n=== 检索分数 ===")
    for i, docs in enumerate(docs_list, 1):
        if len(docs) == 0:
            logging.info(f"\n第 {i} 个查询没有检索到结果。")
            continue
        logging.info(f"\n第 {i} 个查询的检索结果：")
        for doc in docs:
            logging.info(f"检索结果：{doc.metadata['paper_title']}")
    return docs_list

from langchain_core.runnables import Runnable

def get_docs(rank_fusion_results:list)->list:
    return [doc for doc, score in rank_fusion_results]

def create_query_fusion_chain(retriever)->Runnable:
    return generate_queries | retriever.map() | reciprocal_rank_fusion | get_docs | rerank_documents

def create_abstract_query_chain(retriever)->Runnable:
    retriever_with_scores = (
        retriever.map() 
        | RunnableLambda(print_scores)  # 添加打印步骤
    )
    return generate_queries | retriever_with_scores | reciprocal_rank_fusion
#TODO 把可以import的变量都放在__init__的__all__里
rag_fusion_name = "rag_fusion_search"
rag_fusion_description = "生成关于用户问题的多个方面的查询，以提供对原始问题的不同角度的检索结果，把这些结果和原始查询一起进行倒数排序融合，最终返回查询到的文本。"


