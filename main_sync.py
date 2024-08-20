from graph.graph import init_app
from utils.helper import save_chat_history, get_thread_id
from langchain_core.messages import HumanMessage
import warnings
from langchain_core._api.beta_decorator import LangChainBetaWarning
from dotenv import load_dotenv, find_dotenv

# Ignore warnings of specific category
warnings.filterwarnings("ignore", category=LangChainBetaWarning)

def chat(user_message, app, config):
    formatted_user_message = HumanMessage(content=user_message)
    # for chunk in app.stream({"chat_history": formatted_user_message}, config=config, stream_mode="values"):
    #     chunk["messages"][-1].pretty_print()
    for event in app.stream({"chat_history": formatted_user_message}, config=config, ):
        for value in event.values():
            print("Assistant:", value["messages"][-1].content)


if __name__ == "__main__":
    _ = load_dotenv(find_dotenv())
    app = init_app(model_name="gpt-3.5-turbo")
    thread_id = get_thread_id()
    config = {"configurable": {"thread_id": thread_id}}

    while True:
        user_message = input("\n用户:")
        if user_message.strip() == "":
            save_chat_history(app, thread_id)
            print("Chat history saved.\n")
            break
        print("AI:")
        chat(user_message, app, config)

