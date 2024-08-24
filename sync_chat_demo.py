import gradio as gr
from graph.graph import init_app
from utils.helper import save_chat_history, get_thread_id
from langchain_core.messages import HumanMessage, AIMessage, ToolMessage
import warnings
from langchain_core._api.beta_decorator import LangChainBetaWarning
from dotenv import load_dotenv, find_dotenv
import json

# Ignore warnings of specific category
warnings.filterwarnings("ignore", category=LangChainBetaWarning)

_ = load_dotenv(find_dotenv())
# app = init_app(model_name="gemini-1.5-pro-latest", is_async=False)
# app = init_app(model_name="gpt-3.5-turbo", is_async=False)
app = init_app(model_name="gpt-4o", is_async=False)
thread_id = get_thread_id()
config = {"configurable": {"thread_id": thread_id}}

def chat_interface(user_message, history, debug_info):
    
    formatted_user_message = HumanMessage(content=user_message)
    user_input = user_message
    history.append((user_input, ""))  # Append user input to the chat history
    
    debug_messages = []
    
    # Yield the initial state before processing
    yield history, debug_info, ""

    for event in app.stream({"messages": formatted_user_message}, config=config):
        for value in event.values():
            message = value["messages"][-1]
            content = message.content

            if isinstance(message, AIMessage):
                if message.tool_calls:
                    if content:
                        debug_messages.append(f"AI is thinking:\n {content}")
                        yield history, debug_info + "\n".join(debug_messages) + "\n", ""  # Stream AI thoughts
                    debug_messages.append("--")
                    for tool_call in message.tool_calls:
                        debug_messages.append(f"Starting tool: {tool_call['name']} with inputs: {tool_call['args']}")
                        yield history, debug_info + "\n".join(debug_messages) + "\n", ""  # Stream tool call info
                else:
                    # the message to the user
                    history[-1] = (user_input, content)
                    yield history, debug_info + "\n".join(debug_messages) + "\n", ""  # Stream final AI response

            if isinstance(message, ToolMessage):
                tool_messages = []
                for msg in reversed(value["messages"]):
                    if isinstance(msg, ToolMessage):
                        tool_messages.append(msg)
                    else:
                        break

                for tool_message in reversed(tool_messages):
                    debug_messages.append("\nTool output:")
                    content = tool_message.content
                    try:
                        decoded_content = json.loads(content)
                        debug_messages.append(json.dumps(decoded_content, indent=2, ensure_ascii=False))  # Fix Chinese encoding issue
                    except json.JSONDecodeError:
                        debug_messages.append(content)
                    debug_messages.append("--")
                    yield history, debug_info + "\n".join(debug_messages) + "\n", ""  # Stream tool output

    # Final update after processing
    yield history, debug_info + "\n".join(debug_messages) + "\n", ""

# Gradio interface layout
with gr.Blocks(css=".textbox {max-height: 100px; overflow-y: auto;}") as demo:  # Limit the input box height
    with gr.Row():
        with gr.Column(scale=1):
            debug_box = gr.Textbox(label="Debug Info", interactive=False, lines=30, max_lines=30)
        with gr.Column(scale=2):
            with gr.Row():
                chat_history = gr.Chatbot(label="Chat History", height=553)  # Adjust height
            with gr.Row():
                user_input = gr.Textbox(label="Your Message", placeholder="Type your message here...", lines=1, elem_classes="textbox")

    # When a message is sent, update both the chat history and debug info, and clear input
    user_input.submit(chat_interface, inputs=[user_input, chat_history, debug_box], outputs=[chat_history, debug_box, user_input])

demo.launch()
