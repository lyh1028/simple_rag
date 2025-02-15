
from langchain_community.embeddings import ZhipuAIEmbeddings
from typing import List
from langchain_community.document_loaders.parsers import GrobidParser
from langchain_community.document_loaders.generic import GenericLoader
from langchain_community.document_loaders import PyPDFLoader
from langchain_core.documents import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_core.output_parsers import BaseOutputParser
from langchain.retrievers.multi_query import MultiQueryRetriever
from langchain_community.vectorstores import FAISS
from langchain.prompts import PromptTemplate
import logging
import os
from env_var import *
from langchain_community.chat_models import ChatZhipuAI
import ipdb
'''
为了import上面的包，你需要有：
langchain                 0.3.14                   pypi_0    pypi
langchain-community       0.3.14                   pypi_0    pypi
langchain-core            0.3.29                   pypi_0    pypi
langchain-openai          0.2.14                   pypi_0    pypi

langchain-text-splitters  0.3.5                    pypi_0    pypi
langchainhub              0.1.21                   pypi_0    pypi
langgraph                 0.2.61                   pypi_0    pypi
langgraph-checkpoint      2.0.9                    pypi_0    pypi
langgraph-sdk             0.1.51                   pypi_0    pypi
langsmith                 0.2.10                   pypi_0    pypi
'''
embedding_model = ZhipuAIEmbeddings(
    model="embedding-3",
    api_key=os.environ['ZHIPUAI_API_KEY'],
    dimensions=2048
)

class myPDFReader:
    def __init__(self):
        self.loader_type = None
    
    def pypdf_load(self, pdf_path):
        loader = PyPDFLoader(file_path=pdf_path)
        docs = []
        for doc in loader.lazy_load():
            docs.append(doc)
        return docs
    
    def grobid_load(self, pdf_path,**kwargs):
        '''
        docker pull lfoppiano/grobid:0.8.1  然后启动docker
        docker run --rm --init --ulimit core=0  -p 8070:8070 lfoppiano/grobid:0.8.1
        出现以下信息：
        [Wapiti] Loading model: "/opt/grobid/grobid-home/models/funding-acknowledgement/model.wapiti"
        Model path: /opt/grobid/grobid-home/models/funding-acknowledgement/model.wapiti
        说明服务已经在http://localhost:8070 启动，之后不要退出终端
        '''
        segment_sentences = kwargs.get('segment_sentences', False)
        loader = GenericLoader.from_filesystem(
            pdf_path,
            glob="*",
            suffixes=[".pdf"],
            parser= GrobidParser(segment_sentences=segment_sentences)
        )
        docs = loader.load()
        return docs
    
    def load_pdf(self, type, **kwargs)->List[Document]:
        if type == 'pypdf':
            return self.pypdf_load(**kwargs)
        elif type == 'grobid':
            return self.grobid_load(**kwargs)
        else:
            raise ValueError("Invalid loader type")

##spliter
from langchain_text_splitters import RecursiveCharacterTextSplitter
class mySplitter:
    def __init__(self, chunk_size, overlap_size, use_token_split=True, encoding_name="cl100k_base"):
        self.chunk_size = chunk_size
        self.overlap_size = overlap_size
        self.use_token_split = use_token_split
        self.encoding_name = encoding_name
        if use_token_split:
            self.text_splitter = RecursiveCharacterTextSplitter.from_tiktoken_encoder(encoding_name=encoding_name, chunk_size=chunk_size, chunk_overlap=overlap_size)
        else:
            self.text_splitter = RecursiveCharacterTextSplitter(chunk_size=chunk_size, chunk_overlap=overlap_size)

    def split_docs(self, docs):
        all_splits = self.text_splitter.split_documents(docs)
        return all_splits

def create_faiss_index(pdf_path):
    FAISS_INDEX_PATH = f"{faiss_index_dir}/{pdf_path.split('/')[-1]}_index"
    #print(f"开始创建向量数据库：{FAISS_INDEX_PATH}")
    if os.path.exists(FAISS_INDEX_PATH):
        # 加载已存在的向量数据库
        vectordb = FAISS.load_local(
            FAISS_INDEX_PATH,
            embedding_model,
            allow_dangerous_deserialization=True  # 明确允许反序列化
        )
        print("已加载本地向量数据库")
    else:
        # loader
        pdf_reader = myPDFReader()

        docs = pdf_reader.load_pdf('grobid', pdf_path=pdf_path)

        # Spliter
        pdf_splitter = mySplitter(chunk_size=512, overlap_size=32, use_token_split=True, encoding_name="cl100k_base")
        split_docs = pdf_splitter.split_docs(docs)
        def split_array(arr:list[Document], single_array_size:int)->list[list[Document]]:
            # 使用列表切片按 single_array_size 拆分list
            # 因为zhipuAI embedding在调用add_documents时最大允许的文档数是64，所以要分批add
            return [arr[i:i + single_array_size] for i in range(0, len(arr), single_array_size)]
        splits_docs = split_array(split_docs, 64)

        #index
        for i, docs in enumerate(splits_docs):
            if i == 0:
                vectordb = FAISS.from_documents(documents=docs, embedding=embedding_model)
            else:
                vectordb.add_documents(documents=docs)
        vectordb.save_local(FAISS_INDEX_PATH)
    return vectordb
# Output parser will split the LLM result into a list of queries
class LineListOutputParser(BaseOutputParser[List[str]]):
    """Output parser for a list of lines."""

    def parse(self, text: str) -> List[str]: #自定义parser只需要重载parse函数
        lines = text.strip().split("\n")
        return list(filter(None, lines))  # Remove empty lines
    
output_parser = LineListOutputParser()

QUERY_PROMPT = PromptTemplate(
    input_variables=["question"],
    template="""你是一名 AI 语言模型助手。你的任务是：阅读用户的问题，生成5个不同版本的问题，以便从向量数据库中检索相关文档。
    你的目标是，生成用户问题的多个视角，帮助用户克服基于距离的相似性搜索的一些限制。备选问题以换行符隔开。
    原始问题: {question}""",
)
gen_model = ChatZhipuAI(
    model="glm-4-plus",
    temperature=0.5,
)
generate_questions_chain = QUERY_PROMPT | gen_model | output_parser
vectordb = create_faiss_index(pdf_path)

logging.basicConfig()
logging.getLogger("langchain.retrievers.multi_query").setLevel(logging.INFO) #输出multi_query的中间信息
retriever = MultiQueryRetriever(
    retriever=vectordb.as_retriever(search_type="mmr",search_kwargs={"k": 5, "lambda_mult":0.8}), 
    llm_chain=generate_questions_chain, parser_key="lines" #输出以'\n'分隔，是列表；  其他参数: json, comma
)

if __name__ == '__main__':
    question = "请介绍这一篇技术报告的主要内容。"
    docs = retriever.invoke(question)
    from langchain_community.document_transformers import LongContextReorder
    reordering = LongContextReorder()
    docs = reordering.transform_documents(docs)
    context = ""
    for doc in docs:
        context += doc.page_content
    answer_llm = ChatZhipuAI(
        model=answer_llm_name,
        temperature=0.1,
    )
    answer = answer_llm.invoke(f"基于以下信息回答问题：{question}, 信息如下：{context}")
    print(answer.content)