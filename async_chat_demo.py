import gradio as gr
from graph.graph import init_app
from utils.helper import save_chat_history, get_thread_id
from langchain_core.messages import HumanMessage
import asyncio
import warnings
from langchain_core._api.beta_decorator import LangChainBetaWarning
from dotenv import load_dotenv, find_dotenv

# Ignore warnings of specific category
warnings.filterwarnings("ignore", category=LangChainBetaWarning)

_ = load_dotenv(find_dotenv())
app = init_app(model_name="gpt-3.5-turbo")
thread_id = get_thread_id()
config = {"configurable": {"thread_id": thread_id}}

# Streaming and main message processing generator
async def process_message(user_message, chatbot_history, debug_history):

    # Append user message to chatbot history
    chatbot_history.append((user_message, None))  # User message without label

    formatted_user_message = HumanMessage(content=user_message)

    # Streaming the output incrementally
    async for event in app.astream_events({"messages": formatted_user_message}, config=config, version="v1"):
        kind = event["event"]
        if kind == "on_chat_model_stream":
            content = event["data"]["chunk"].content
            if content:
                # Update the last message for the AI incrementally
                if chatbot_history and chatbot_history[-1][1] is not None:
                    chatbot_history[-1] = (chatbot_history[-1][0], chatbot_history[-1][1] + content)
                else:
                    chatbot_history[-1] = (chatbot_history[-1][0], content)  # Add the first chunk
                yield chatbot_history, debug_history  # Ensure streaming updates
        elif kind == "on_tool_start":
            debug_history += f"Starting tool: {event['name']} with inputs: {event['data'].get('input')}\n"
            yield chatbot_history, debug_history  # Stream tool start info
        elif kind == "on_tool_end":
            debug_history += f"Done tool: {event['name']}\nTool output: {event['data'].get('output')}\n--\n"
            yield chatbot_history, debug_history  # Stream tool end info

# Gradio app interface
def start_gradio():
    with gr.Blocks() as demo:
        with gr.Row():
            with gr.Column(scale=3):
                chatbot = gr.Chatbot(label="User-AI Chat", show_label=False)
            with gr.Column(scale=1):
                debug_info = gr.Textbox(label="Debug Info", lines=10, interactive=False)

        user_input = gr.Textbox(label="Your message", placeholder="Type your message here", lines=1)

        # Submit function for message input
        user_input.submit(
            process_message, 
            [user_input, chatbot, debug_info], 
            [chatbot, debug_info],
            queue=True  # Enable queueing for streaming with async functions
        )

    demo.launch()

if __name__ == "__main__":
    start_gradio()