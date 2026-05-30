from langchain_huggingface import HuggingFaceEndpoint,ChatHuggingFace
from dotenv import load_dotenv
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnableParallel, RunnableBranch, RunnableLambda,RunnableSequence
from langchain_core.output_parsers import PydanticOutputParser
from pydantic import BaseModel, Field
from typing import Literal
import os 

load_dotenv()

llm1=HuggingFaceEndpoint(
    repo_id="deepseek-ai/DeepSeek-V4-Pro",
    task="text-generation"

)
model1=ChatHuggingFace(llm=llm1)

llm2=HuggingFaceEndpoint(
    repo_id ="meta-llama/Llama-3.1-8B-Instruct",
    task="text-generation"
)
model2=ChatHuggingFace(llm=llm2)





# prompt1=PromptTemplate(
#     template="explain the detail Report about tpoic {topic}",
#     input_variables=["topic"]
# )

# prompt2=PromptTemplate(
#     template="g9ive the summary for for info {info} ",
#     input_variables=["info"]

# )

# parser=StrOutputParser()

# chain= prompt1 | model | parser | prompt2 | model | parser

# result=chain.invoke({"topic":"AI"})

# print(result)

# prompt1=PromptTemplate(
#     template="explain the detail report about the text {text}",
#     input_variables=["text"]
# )

# prompt2=PromptTemplate(
#     template="create a 5 question quiz fromthe gevin text {text}",
#     input_variables=["text"]
# )

# prompt3= PromptTemplate(
#     template="merge the both report and quiz into single dcoument \n notes -> {notes} and quiz -> {quiz}",
#     input_variables=["notes","quiz"]
# )

# parser=StrOutputParser()

# parallel_chain=RunnableParallel({
#     "notes": prompt1 | model1 | parser,
#     "quiz": prompt2 | model2 | parser
# })

# merge_chain= prompt3 | model1 | parser

# chain=parallel_chain | merge_chain

# text=''' Support vector machines (SVMs) are a set of supervised learning methods used for classification, regression and outliers detection.

# The advantages of support vector machines are:

# Effective in high dimensional spaces.

# Still effective in cases where number of dimensions is greater than the number of samples.

# Uses a subset of training points in the decision function (called support vectors), so it is also memory efficient.

# Versatile: different Kernel functions can be specified for the decision function. Common kernels are provided, but it is also possible to specify custom kernels.

# The disadvantages of support vector machines include:

# If the number of features is much greater than the number of samples, avoid over-fitting in choosing Kernel functions and regularization term is crucial.

# SVMs do not directly provide probability estimates, these are calculated using an expensive five-fold cross-validation (see Scores and probabilities, below).

# The support vector machines in scikit-learn support both dense (numpy.ndarray and convertible to that by numpy.asarray) and sparse (any scipy.sparse) sample vectors as input. However, to use an SVM to make predictions for sparse data, it must have been fit on such data. For optimal performance, use C-ordered numpy.ndarray (dense) or scipy.sparse.csr_matrix (sparse) with dtype=float64.'''

# result=chain.invoke({"text":text})

# print(result)



class SentimentAnalysis(BaseModel):
    sentiment : Literal['positive', 'negative']= Field(description="give the sentiment of the feedback")

parser2=PydanticOutputParser(pydantic_object=SentimentAnalysis)

prompt1=PromptTemplate(
    template='give the sentiment of review eiter a postive or negative using given review {review} \n {format_instructions}',
    input_variables=["review"],
    partial_variables={'format_instructions':parser2.get_format_instructions()}

)

parser=StrOutputParser()

prompt2=PromptTemplate(
    template=" give the approprite response for the postive review {review}",
    input_variables=["review"]
)

prompt3=PromptTemplate(
    template="give the approprite response for the negative review {review}",
    input_variables=["review"]
)

chain_simple=RunnableSequence(prompt1 , model1 , parser2)

branch_chain=RunnableBranch(
    (lambda X : X.sentiment=="positive" , prompt2 | model1 | parser ),
    (lambda X : X.sentiment=="negative" , prompt3 | model1 | parser ),
    RunnableLambda(lambda X : "could not find out the sentiment")
)

chain= chain_simple | branch_chain

result=chain.invoke({"review":"I recently upgraded to the Samsung Galaxy S24 Ultra, and I must say, it’s an absolute powerhouse! The Snapdragon 8 Gen 3 processor makes everything lightning fast—whether I’m gaming, multitasking, or editing photos. The 5000mAh battery easily lasts a full day even with heavy use, and the 45W fast charging is a lifesaver"})


print(result)


