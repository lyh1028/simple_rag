from langchain.prompts import ChatPromptTemplate
from functools import partial
from operator import itemgetter
# Decomposition
#TODO 用decomposition处理多跳问题
llm_template = """给定用户的问题，你需要将这个问题分解为3个子问题，三个子问题是层层递进的，通过逐步回答子问题，最终获得原问题的答案。
用户的问题是： {question} \n
直接分条输出三个子问题：:"""
question_decomposition = ChatPromptTemplate.from_template(llm_template)

from custom_tools.common import LineListOutputParser
from langchain_core.output_parsers import StrOutputParser
from langchain_community.chat_models import ChatZhipuAI
decompose_search_name = "decompose_search"
decompose_search_description = "把一个大问题分解成多个子问题，逐步回答子问题后，可以获得原查询问题的答案。"
def create_decomposition_search(retriever):
    # LLM
    def recursvie_search(retriever, question):
        llm = ChatZhipuAI(model='glm-4-plus', temperature=0)

        # Chain
        generate_queries_decomposition = ( question_decomposition | llm | LineListOutputParser() )
        questions = generate_queries_decomposition.invoke({"question": question})
        template = """作为一名科研助手，你需要回答以下问题:
        \n --- \n {question} \n --- \n

        与这个问题相关的问题-答案对如下：
        \n --- \n {q_a_pairs} \n --- \n
        问题相关的补充信息如下：
        \n --- \n {context} \n --- \n

        使用上述信息回答： \n {question}
        """
        prompt_template = ChatPromptTemplate.from_template(template)
        def format_qa_pair(question, answer):
            """Format Q and A pair"""
            formatted_string = ""
            formatted_string += f"Question: {question}\nAnswer: {answer}\n\n"
            return formatted_string.strip()
        
        q_a_pairs = "temporary None"
        for q in questions:
            rag_chain = (
            {"context": itemgetter("question") | retriever, 
            "question": itemgetter("question"),
            "q_a_pairs": itemgetter("q_a_pairs")} 
            | prompt_template
            | llm
            | StrOutputParser())

            answer = rag_chain.invoke({"question":q,"q_a_pairs":q_a_pairs})
            q_a_pair = format_qa_pair(q,answer)
            q_a_pairs = q_a_pairs + "\n---\n"+  q_a_pair
        
        return answer
    return partial(recursvie_search, retriever)





    
    