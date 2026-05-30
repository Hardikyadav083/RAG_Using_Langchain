from langchain_community.document_loaders import TextLoader,UnstructuredFileLoader, PyPDFLoader,DirectoryLoader,WebBaseLoader
from langchain_huggingface import HuggingFaceEndpoint,ChatHuggingFace
from langchain_core.prompts import PromptTemplate
from bs4 import BeautifulSoup
from langchain_core.output_parsers import StrOutputParser
from dotenv import load_dotenv
import os

load_dotenv()


llm=HuggingFaceEndpoint(
    repo_id="zai-org/GLM-4.5",
    task="text-generation"
)

model=ChatHuggingFace(llm=llm)
prompt=PromptTemplate(
    template="give the summary for this text {text}",
    input_variables=["text"]
)

# loader=TextLoader("C:/HARDIK YADAV/my_doc.txt",encoding="utf-8")

# data=loader.load()

# loader=PyPDFLoader("C:/HARDIK YADAV/Hardik_R.pdf")

# docs=loader.load()
# chain=prompt | model | StrOutputParser()

# result=chain.invoke({"text":docs[0].page_content})

# # print(result)
# print(len(docs))

# loader=DirectoryLoader(
#     path="books",
#     glob="*.pdf",
#     loader_cls=PyPDFLoader,
#     show_progress=True
# )

# docs=loader.load()

# for i in range(len(docs)):
#     print(docs[i].metadata)

url="https://docs.langchain.com/oss/python/integrations/document_loaders/pdfplumber"

loader=WebBaseLoader(url)

docs=loader.load()
print(len(docs))
print(docs[0].metadata)
