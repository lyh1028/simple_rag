from langchain_community.chat_models import ChatZhipuAI
from langchain_core.runnables import RunnableLambda, Runnable
from custom_tools.common import docs2str, LineListOutputParser
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.vectorstores import VectorStoreRetriever
from langchain.load import dumps, loads
# 1. 定义与 retriever 无关的通用组件
# 多句查询 + step_back trick
multi_query_prompt_template = ChatPromptTemplate.from_messages([
    ("system", '''你是一个AI论文检索查询助手，需要帮助改写用户查询以生成更好的向量数据库的查询语句。
    Workflow: 阅读用户的问题，生成5个措辞不同但语义一致的问题，如果用户的问题是细节问题，可以从在5个问题改写中提供一些从更宏观的角度提问的问题，以便从向量数据库中检索相关文档，直接分条返回生成的问题'''),
    ("user", "用户的问题是：{question}")
])
output_parser = LineListOutputParser()
gen_model = ChatZhipuAI(model="glm-4-plus", temperature=0.1)
multi_query_chain = multi_query_prompt_template | gen_model | output_parser

def get_unique_union(documents: list[list]):
    flattened_docs = [dumps(doc) for sublist in documents for doc in sublist]
    unique_docs = list(set(flattened_docs))
    return [loads(doc) for doc in unique_docs]

# 2. 通过函数动态绑定 retriever
def create_multi_query_chain(retriever:VectorStoreRetriever)->Runnable:
    return multi_query_chain | retriever.map() | RunnableLambda(get_unique_union) | RunnableLambda(docs2str)

#TODO 把可以import的变量都放在__init__的__all__里
multi_query_name = "multi_query_search"
multi_query_description = "为了缓解余弦相似度查询的局限性，生成多个语义类似但措辞不同的查询，以获取更全面的上下文"
