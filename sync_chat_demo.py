#-*- coding: utf-8

from graph.graph import init_app
from utils.helper import save_chat_history, get_thread_id
from langchain_core.messages import HumanMessage, AIMessage, ToolMessage
import warnings
from langchain_core._api.beta_decorator import LangChainBetaWarning
from dotenv import load_dotenv, find_dotenv
import json
import gradio as gr

# Ignore warnings of specific category
warnings.filterwarnings("ignore", category=LangChainBetaWarning)

# Global variable to store chat content
global_chat_output = ""

_ = load_dotenv(find_dotenv())
app = init_app(model_name="gemini-1.5-pro-latest", is_async=False)
thread_id = get_thread_id()
config = {"configurable": {"thread_id": thread_id}}

def chat(user_message, app, config):
    global global_chat_output
    global_chat_output = ""  # Reset for each new conversation

    formatted_user_message = HumanMessage(content=user_message)
    for event in app.stream({"messages": formatted_user_message}, config=config,):
        for value in event.values():
            message = value["messages"][-1]
            content = message.content

            if isinstance(message, AIMessage):
                if message.tool_calls:
                    if content:
                        print(content)
                    print("--")
                    for tool_call in message.tool_calls:
                        print(f"Starting tool: {tool_call['name']} with inputs: {tool_call['args']}")
                else:
                    # the message to the user
                    print(content)
                    # Add new content to global variable
                    global_chat_output += content

            if isinstance(message, ToolMessage):
                tool_messages = []
                for msg in reversed(value["messages"]):
                    if isinstance(msg, ToolMessage):
                        tool_messages.append(msg)
                    else:
                        break

                for tool_message in reversed(tool_messages):
                    print("\nTool output:")
                    content = tool_message.content
                    try:
                        decoded_content = json.loads(content)
                        print(decoded_content)
                    except json.JSONDecodeError:
                        print(content)
                    print("--")



# Generator function for Gradio that streams content
def gradio_stream(user_message, history):

    global global_chat_output
    chat(user_message, app, config)
    return global_chat_output

# Define a Gradio interface
gr.ChatInterface(gradio_stream).launch()


