from langchain_huggingface import HuggingFaceEndpoint, ChatHuggingFace
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_core.documents import Document
from langchain_community.document_loaders import TextLoader, UnstructuredFileLoader, PyPDFLoader, DirectoryLoader, WebBaseLoader
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_community.retrievers import WikipediaRetriever 
from langchain_community.vectorstores import Chroma, FAISS
from langchain_classic.retrievers.multi_query import MultiQueryRetriever
from langchain_classic.retrievers import ContextualCompressionRetriever
from langchain_classic.retrievers.document_compressors import LLMChainExtractor
from dotenv import load_dotenv
import os

load_dotenv()

llm = HuggingFaceEndpoint(
    repo_id="zai-org/GLM-4.5", # Added the missing comma here
    task="text-generation"
)

model = ChatHuggingFace(llm=llm)

embedding = HuggingFaceEmbeddings(
    model_name="intfloat/e5-small-v2",
)

# Your script continues from here...

# retriever=WikipediaRetriever(top_k_results=2,lan="en",load_all_available_meta=True)


# query="the geopolitical history of india and pakistan from the perspective of a chinese."
# docs=retriever.invoke(query)

# for i , docs in enumerate(docs):
#     print(f"\n--- result {i+1}----")
#     print(f"content: \n {docs.page_content}.......")


# docs = [
#     Document(page_content="LangChain makes it easy to work with LLMs."),
#     Document(page_content="LangChain is used to build LLM based applications."),
#     Document(page_content="Chroma is used to store and search document embeddings."),
#     Document(page_content="Embeddings are vector representations of text."),
#     Document(page_content="MMR helps you get diverse results when doing similarity search."),
#     Document(page_content="LangChain supports Chroma, FAISS, Pinecone, and more."),
# ]

# vector_store=FAISS.from_documents(
#     documents=docs,
#     embedding=embedding
# )


# vector_store.add_documents(docs)

# retriever=vector_store.as_retriever(search_type="mmr",search_kwargs={"k":3,"lambda_mult" : 0.5})


# query="what is langchain ?"
# results=retriever.invoke(query)

# for i, doc in enumerate(results):
#     print(f"\n--- Result {i+1} ---")
#     print(doc.page_content)


# Relevant health & wellness documents
all_docs = [
    Document(page_content="Regular walking boosts heart health and can reduce symptoms of depression.", metadata={"source": "H1"}),
    Document(page_content="Consuming leafy greens and fruits helps detox the body and improve longevity.", metadata={"source": "H2"}),
    Document(page_content="Deep sleep is crucial for cellular repair and emotional regulation.", metadata={"source": "H3"}),
    Document(page_content="Mindfulness and controlled breathing lower cortisol and improve mental clarity.", metadata={"source": "H4"}),
    Document(page_content="Drinking sufficient water throughout the day helps maintain metabolism and energy.", metadata={"source": "H5"}),
    Document(page_content="The solar energy system in modern homes helps balance electricity demand.", metadata={"source": "I1"}),
    Document(page_content="Python balances readability with power, making it a popular system design language.", metadata={"source": "I2"}),
    Document(page_content="Photosynthesis enables plants to produce energy by converting sunlight.", metadata={"source": "I3"}),
    Document(page_content="The 2022 FIFA World Cup was held in Qatar and drew global energy and excitement.", metadata={"source": "I4"}),
    Document(page_content="Black holes bend spacetime and store immense gravitational energy.", metadata={"source": "I5"}),
]

vector_store=FAISS.from_documents(
    documents=all_docs,
    embedding=embedding
)

similarity_retriever=vector_store.as_retriever(search_kwargs={"k":4})

# multiquery_retriever=MultiQueryRetriever.from_llm(
#     retriever=vector_store.as_retriever(search_kwargs={"k":2}),
#     llm=model,
# )

# query = "How to improve energy levels and maintain balance?"

# result1=similarity_retriever.invoke(query)
# result2=multiquery_retriever.invoke(query)

# for i, doc in enumerate(result1):
#     print(f"\n--- Result {i+1} ---")
#     print(doc.page_content)

# print("*"*150)

# for i, doc in enumerate(result2):
#     print(f"\n--- Result {i+1} ---")
#     print(doc.page_content)


Doc_compressor=LLMChainExtractor.from_llm(model)
compression_retriever=ContextualCompressionRetriever(
    base_compressor=Doc_compressor,
    base_retriever=similarity_retriever
)

query="How to improve energy levels and maintain balance?"
compressed_doc=compression_retriever.invoke(query)

for i , doc in enumerate(compressed_doc):
    print(f"\n--- Result {i+1} ---")
    print(doc.page_content)
