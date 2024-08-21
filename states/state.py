from typing import TypedDict, Annotated
from langgraph.graph.message import add_messages
from operator import add

# Define the state object for the agent graph
class PublicState(TypedDict):
    chat_history: Annotated[list, add_messages] # Only include the messages of AI and human interaction
    current_task_history: list # Include the messages of AI, human interaction, tool calls and tool messages
    messages: list # It should only have either an AI message or a tool message returned from a tool.
                   # It has to be named 'messages' because LangGraph's ToolNode class hardcodes this name.
    saved_information: Annotated[list[str], add]
    tool_called: bool
