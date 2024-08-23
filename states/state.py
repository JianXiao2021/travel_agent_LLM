from typing import TypedDict, Annotated
from langgraph.graph.message import add_messages

# Define the state object for the agent graph
class PublicState(TypedDict):
    messages: Annotated[list, add_messages]
