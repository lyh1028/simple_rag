import os
os.environ['LANGCHAIN_ENDPOINT'] = "https://api.smith.langchain.com"
os.environ['OPENAI_API_KEY'] = 'sk-bd29a790b61a4cfe856c1359479a07b8'
os.environ["LANGSMITH_TRACING"] = "true"
os.environ["LANGSMITH_API_KEY"] = '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'
os.environ['ZHIPUAI_API_KEY'] = 'bfcd25e3a3f4450d9883cfe21ddec0b2.NcQIElTypgZDUT8A'
pdf_path = '/Users/liyihang/Documents/deepseek-r1-report.pdf'
multi_query_k = 5
answer_llm_name = "glm-4-plus"
img_save_path = '/Users/liyihang/code/langchain_study/rag_agent_simple/graph.png'
faiss_index_dir = '/Users/liyihang/code/langchain_study/rag_agent_simple/faiss_db'