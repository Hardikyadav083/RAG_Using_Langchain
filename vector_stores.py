from langchain_huggingface import HuggingFaceEndpoint,ChatHuggingFace, HuggingFaceEmbeddings
from langchain_core.prompts import PromptTemplate
from langchain_core.documents import Document
from langchain_core.output_parsers import StrOutputParser, JsonOutputParser
from langchain_classic.output_parsers.structured import (
    StructuredOutputParser, ResponseSchema
)
from langchain_community.document_loaders import TextLoader,UnstructuredFileLoader, PyPDFLoader,DirectoryLoader,WebBaseLoader
from langchain_community.vectorstores import Chroma
from dotenv import load_dotenv
import os

load_dotenv()

embedding=HuggingFaceEmbeddings(
    model_name="sentence-transformers/all-MiniLM-L6-v2",
)

doc1 = Document(
        page_content="Virat Kohli is one of the most successful and consistent batsmen in IPL history. Known for his aggressive batting style and fitness, he has led the Royal Challengers Bangalore in multiple seasons.",
        metadata={"team": "Royal Challengers Bangalore"}
    )
doc2 = Document(
        page_content="Rohit Sharma is the most successful captain in IPL history, leading Mumbai Indians to five titles. He's known for his calm demeanor and ability to play big innings under pressure.",
        metadata={"team": "Mumbai Indians"}
    )
doc3 = Document(
        page_content="MS Dhoni, famously known as Captain Cool, has led Chennai Super Kings to multiple IPL titles. His finishing skills, wicketkeeping, and leadership are legendary.",
        metadata={"team": "Chennai Super Kings"}
    )
doc4 = Document(
        page_content="Jasprit Bumrah is considered one of the best fast bowlers in T20 cricket. Playing for Mumbai Indians, he is known for his yorkers and death-over expertise.",
        metadata={"team": "Mumbai Indians"}
    )
doc5 = Document(
        page_content="Ravindra Jadeja is a dynamic all-rounder who contributes with both bat and ball. Representing Chennai Super Kings, his quick fielding and match-winning performances make him a key player.",
        metadata={"team": "Chennai Super Kings"}
    )

doc=[doc1,doc2,doc3,doc4,doc5]

vector_store=Chroma(
    embedding_function=embedding,
    collection_name="smaple",
    persist_directory="my_chroma_db"

)

vector_store.add_documents(doc)

vector_store.get(include=["embeddings","documents","metadatas"])

result=vector_store.similarity_search(
    query="who amoung is the bolwer ? ",
    k=1
)

print(result[0].page_content)
