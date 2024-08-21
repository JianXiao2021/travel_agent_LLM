from graph.graph import init_app
from utils.helper import save_chat_history, get_thread_id
from langchain_core.messages import HumanMessage
import asyncio
import warnings
from langchain_core._api.beta_decorator import LangChainBetaWarning
from dotenv import load_dotenv, find_dotenv

# Ignore warnings of specific category
warnings.filterwarnings("ignore", category=LangChainBetaWarning)

# Streaming output (Reference: https://langchain-ai.github.io/langgraph/how-tos/streaming-tokens/#streaming-llm-tokens)
async def chat(user_message, app, config, verbose):
    formatted_user_message = HumanMessage(content=user_message)
    async for event in app.astream_events({"chat_history": formatted_user_message}, config=config, version="v1"):
        kind = event["event"]
        if kind == "on_chat_model_stream":
            content = event["data"]["chunk"].content
            if content and verbose:
                # Empty content in the context of OpenAI or Anthropic usually means
                # that the model is asking for a tool to be invoked.
                # So we only print non-empty content
                print(content, end="")
        elif kind == "on_tool_start":
            print("--")
            print(
                f"Starting tool: {event['name']} with inputs: {event['data'].get('input')}"
            )
        elif kind == "on_tool_end":
            print(f"Done tool: {event['name']}")
            print(f"Tool output was: {event['data'].get('output')}")
            print("--")

# Define an async main function
async def main():
    _ = load_dotenv(find_dotenv())
    app = init_app(model_name="gpt-3.5-turbo",)
    thread_id = get_thread_id()
    config = {"configurable": {"thread_id": thread_id}}
    verbose = True

    while True:
        user_message = input("\n用户:")
        if user_message.strip() == "":
            save_chat_history(app, thread_id)
            print("Chat history saved.\n")
            break
        print("AI:")
        await chat(user_message, app, config, verbose)

# Use asyncio.run() to run the main function
if __name__ == "__main__":
    asyncio.run(main())

