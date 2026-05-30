from langchain_huggingface import HuggingFaceEndpoint, ChatHuggingFace
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser, JsonOutputParser
from langchain_classic.output_parsers.structured import (
    StructuredOutputParser, ResponseSchema
)
from dotenv import load_dotenv
import os

load_dotenv()

llm=HuggingFaceEndpoint(
    repo_id='zai-org/GLM-4.5',
    task="text-generation"

)
model=ChatHuggingFace(llm=llm)

# template1=PromptTemplate(
#     template="explain in detail about this topic {topic}",
#     input_variables=["topic"]

# )

# template2=PromptTemplate(
#     template="write a 5 line summary about this folling text{text}",
#     input_variables=["text"]
# )

# parsers= StrOutputParser()

# chain= template1 | model | parsers | template2 | model | parsers

# result = chain.invoke({"topic" : "AI"})
# print(result)

# parser=JsonOutputParser()

# template=PromptTemplate(
#     template="give the imagination name age and loction for person \n {format_instructions}",
#     input_variables=[],
#     partial_variables={'format_instructions':parser.get_format_instructions()}
# )

# chain=template | model | parser

# result=chain.invoke({})

# print(result)

Schema=[
    ResponseSchema(name='Fact_1',description='Fact 1 about the topic'),
    ResponseSchema(name='Fact_2',description='Fact 2 about the topic'),
    ResponseSchema(name='Fact_3',description='Fact 3 about the topic'),
]

parser=StructuredOutputParser.from_response_schemas(Schema)

template=PromptTemplate(
    template="give the 3 facts about topic {topic} \n {format_instructions}",
    imput_variable=["topic"],
    partial_variables={'format_instructions':parser.get_format_instructions()}

)

chain=template | model | parser
result=chain.invoke({"topic":"black Hole"})
print(result)
