# from langchain_huggingface import HuggingFaceEndpoint,ChatHuggingFace
# from dotenv import load_dotenv
# from typing import TypedDict
# import os

# load_dotenv()

# llm=HuggingFaceEndpoint(
#     repo_id="zai-org/GLM-4.5",
#     task="text-generation"
# )

# model=ChatHuggingFace(llm=llm)

# class User(TypedDict):
#     name : str
#     email : str

# class product(TypedDict):
#     name:str
#     price : float
#     quantity : int

# class Order(TypedDict):
#     order_id: int
#     user : User
#     items: list[product]
#     total_price : float

# structure_output=model.with_structured_output(Order)

# result=structure_output.invoke(
#     "Create an order for user Hardik with email hardik@example.com. "
#     "He bought 1 laptop worth 50000 and 2 mouse worth 500 each."
# )

# class UserProfiledRequired(TypedDict):
#     name: str
#     age: int
#     email: str

# class UserProfileOptional(TypedDict):
#     phone : str
#     address :str

# class UserProfile(UserProfiledRequired, UserProfileOptional):
#     pass


# structure_output=model.with_structured_output(UserProfile)

# # result=structure_output.invoke( "Create a user profile for Hardik with email hardik@example.com and age is 22. "
# #     "Phone is 9876543210. Address is Mumbai.")


# # print(result)

# def process_user(profile : UserProfile):
#     name=profile["name"]
#     age=profile["age"]
#     email=profile["email"]
    
#     phone=profile.get("phone","notprovided")
#     address= profile.get("address","notprovided")

#     print('f Name : {name}')
#     print('f Age : {age}')
#     print('f Email : {email}')
#     print('f Phone : {phone}')
#     print('f Address : {address}')

# result=structure_output.invoke( "Create a user profile for Rahul with email rahul@example.com and age 21")

# print(result)


# Pydantic



# from pydantic import BaseModel, Field
# from typing import Optional
# class User(BaseModel):
#     name : str= Field(min_length=6)
#     age : int =Field(ge=18,le=60)
#     email : str

# new_user={"name": "hardik","age": 20,"email": "hardik@gmail.com"}

# user=User(**new_user)

# user_json=user.model_dump_json()
# print(user_json)




# from pydantic import BaseModel,field_validator,Field
# from typing import Optional

# class User(BaseModel):
#     password : str

#     @field_validator("password")
#     def check_password(cls,value):
#         if len(value)>8:
#             raise ValueError ("password must be 8 character long")
#         if not any(char.isdigit() for char in value):
#             raise ValueError("password must contain at least one digit")
#         return value
    

# user=User(password="abc23")
# print(user)





from pydantic import BaseModel,field_validator,Field
from typing import Optional,Literal,TypedDict
from langchain_huggingface import HuggingFaceEndpoint,ChatHuggingFace
from dotenv import load_dotenv
from typing import TypedDict
import os

load_dotenv()

llm=HuggingFaceEndpoint(
    repo_id="mistralai/Mistral-7B-Instruct-v0.2",
    task="text-generation"
)

model=ChatHuggingFace(llm=llm)

class Review(BaseModel):
    key_themes : list[str]=Field(description="Write down all the key themes discussed in the review in a list")
    summary : str=Field(description="A brief summary of the review")
    sentiment : Literal["pos","neg"]=Field(description="Return sentiment of the review either negative, positive or neutral")
    pros: Optional[list[str]] = Field(default=None, description="Write down all the pros inside a list")
    cons: Optional[list[str]] = Field(default=None, description="Write down all the cons inside a list")
    name: Optional[str] = Field(default=None, description="Write the name of the reviewer")

structure_output=model.with_structured_output(Review)
result=structure_output.invoke("""I recently upgraded to the Samsung Galaxy S24 Ultra, and I must say, it’s an absolute powerhouse! The Snapdragon 8 Gen 3 processor makes everything lightning fast—whether I’m gaming, multitasking, or editing photos. The 5000mAh battery easily lasts a full day even with heavy use, and the 45W fast charging is a lifesaver.

The S-Pen integration is a great touch for note-taking and quick sketches, though I don't use it often. What really blew me away is the 200MP camera—the night mode is stunning, capturing crisp, vibrant images even in low light. Zooming up to 100x actually works well for distant objects, but anything beyond 30x loses quality.

However, the weight and size make it a bit uncomfortable for one-handed use. Also, Samsung’s One UI still comes with bloatware—why do I need five different Samsung apps for things Google already provides? The $1,300 price tag is also a hard pill to swallow.

Pros:
Insanely powerful processor (great for gaming and productivity)
Stunning 200MP camera with incredible zoom capabilities
Long battery life with fast charging
S-Pen support is unique and useful

Review by Hardik Yadav
""")

print(result)



