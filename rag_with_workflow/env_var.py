import os
os.environ["LANGCHAIN_TRACING_V2"] = "true"
os.environ['LANGCHAIN_ENDPOINT'] = "https://api.smith.langchain.com"
os.environ['LANGCHAIN_API_KEY'] = 'xxx'
os.environ['OPENAI_API_KEY'] = 'xxx'
os.environ['ZHIPUAI_API_KEY'] = 'xxx'
pdf_path = '/Users/liyihang/Documents/deepseek-r1-report.pdf'
multi_query_k = 5
answer_llm_name = "glm-4-plus"
img_save_path = 'xxx/graph.png'
faiss_index_dir = 'xxx/faiss_db'