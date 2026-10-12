#so now we are creating a graph
#now first thing you create is a state
import os

#1)  typed DICT (common)

from typing import TypedDict

class State(TypedDict):
    topic:str
    summary:str
    score:str  



#2) using pydantic (it is good at data validation and type checking use in fastapi)

from pydantic import BaseModel, field_validator

class State(BaseModel):
    topic:str
    summary:str =""
    score:str

    @field_validator
    def score_positive(cls,v):
        if v<0:
            raise ValueError("score must be positive")


#3) python dataclasses (it is stansard python data class but it used very rarely)

from dataclasess import dataclass,field

@dataclass
class State:
    topic:str =""
    summary:str =""
    meassage:list = field(default_factory=list)

#4)using langgraph
from langgraph.graph import MessagesState

class State(MessagesState):
    #messages field is already included with add_messages reducer
    #just add your extra field
    user_name:str
    language:str
