from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_openai import ChatOpenAI
from langchain_huggingface import ChatHuggingFace, HuggingFaceEndpoint
from langchain_core.prompts import PromptTemplate,ChatPromptTemplate
from langchain_core.messages import HumanMessage,AIMessage,SystemMessage
from dotenv import load_dotenv
load_dotenv()
import os



llm=HuggingFaceEndpoint(
    repo_id="zai-org/GLM-4.5",
    task="text-generation"
)

model=ChatHuggingFace(llm=llm)

# chat_history=[SystemMessage(content="You are a helpful assistant")]
# while True:
#     user_input=input('User :'  )
#     chat_history.append(HumanMessage(content=user_input))
#     if user_input=="exit":
#         break
#     result=model.invoke(chat_history)
#     chat_history.append(AIMessage(content=result.content))
#     print("AI : " , result.content)

template1=PromptTemplate(
    template=" Clean this product description {text}",
    input_variables=["text"]
)

template2=PromptTemplate(
    template="Extract keyword from clean description {clean_text}",
    input_variables=["clean_text"]

)

template3=PromptTemplate(
    template="create ad diagolue using those keywords {keywords}",
    input_variables=["keywords"]

)

clean_text=model.invoke(template1.invoke({"text":'''This amazing dog water bottle is very very useful for pets and animals especially dogs and cats it is portable and easy to carry anywhere like travel walking park etc it has leak proof design but sometimes little water can come out if not closed properly it is made of plastic but good quality and safe for pets also comes with attached bowl so dog can drink easily no need extra bowl and it is very good product for summer and outdoor use must buy for all pet owners highly recommended best product cheap price good quality'''})).content
keywords=model.invoke(template2.invoke({"clean":clean_text})).content
ad=model.invoke(template3.invoke({"keywords":keywords})).content
print(ad.content)