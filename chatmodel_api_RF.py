from langchain_huggingface import HuggingFaceEndpoint,ChatHuggingFace
from dotenv import load_dotenv
import os

load_dotenv()

llm=HuggingFaceEndpoint(
    repo_id="zai-org/GLM-4.5",
    task="text-generation"
    
)

model=ChatHuggingFace(llm=llm)

result=model.invoke("What is the capital of India")

print(result.content)