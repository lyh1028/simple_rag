# simple_rag
A toy rag project using langchain.

## Rag agent simple

After giving your env_var, run "python run.py"

You should also modify GrobidParser code, adding logic for "abstract" of the document(process_xml function).

```python
abstract = soup.find_all("abstract")
....
...
if abstract:
    paragraph_dict = {
                    "text": abstract[0].text,
                    "para": str(0),
                    "bboxes": [],
                    "section_title": 'Abstract',
                    "section_number": 0,
                    "pages": (1, 1),
                    "pub_time": pub_time_values
                }
    chunks.append(paragraph_dict)
```
A simple agent, reading a directory of article in pdf and building faiss index.

First retrieve summary to decide which articles are needed for user's question, then automatically choose different rag methods according to the question (use @tool).[Step-back,  Multi-query, rag-fusion, ...] You can also implement your method.
