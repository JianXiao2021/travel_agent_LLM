from typing import TypedDict, Annotated
from langgraph.graph.message import add_messages
from operator import add

# Define the state object for the agent graph
class PublicState(TypedDict):
    chat_history: Annotated[list, add_messages] # Only include the messages of AI and human interaction
    messages: list # Include the messages of AI, human interaction, tool calls and tool messages
    saved_information: Annotated[list[str], add]
