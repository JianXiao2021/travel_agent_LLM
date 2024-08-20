from states.state import PublicState
from langchain_core.tools import StructuredTool
from models.factory import LLMFactory
from langchain_core.messages import AIMessage, SystemMessage, ToolMessage
from utils.helper import get_current_local_datetime

# def init_message_state(state: PublicState):
#     previous_chat_history = state.get("chat_history", [])
#     last_chat = state.get("current_task_history", [])
#     if last_chat:
#         last_chat = last_chat[-2:]  # One is the AI response, the other is the user message
#     return {
#         "chat_history": last_chat,  # will be appeneded to the chat history
#         "current_task_history": previous_chat_history + last_chat  # will replace the current task history
#     }

# When streaming the graph the user message will get added to the 'chat_history' but not the 'messages' (see main.py)
# So we need to manually add the user message to the 'messages' list as the 'messages' list will be used in the LLM invocation
def init_current_task_history(state: PublicState):
    current_task_history = state.get("messages")
    if current_task_history:
         current_task_history.append(state["chat_history"][-1])  # append the user message
    else:
        current_task_history = state["chat_history"]  # The beginning of the conversation
    return {"messages": current_task_history}

class Agent:
    def __init__(self, model_name: str, temperature: float, prompt_template: str, tools: list[StructuredTool]):
        # Init the LLM only when creating the agent in graph initialization
        # so we don't need to create it every time the agent is invoked
        self.llm = LLMFactory.get_llm(model=model_name, temperature=temperature)
        if tools:
            self.llm = self.llm.bind_tools(tools)
        self.prompt_template = prompt_template


class AsyncAgent(Agent):

    def need_clear_current_task_history(self, last_message):
        return (
            last_message
            and isinstance(last_message, ToolMessage)
            and last_message.name == "save_info_and_clear_history"
            and hasattr(last_message, 'artifact')
        )

    async def __call__(self, state: PublicState):
        saved_information = state.get("saved_information", [])
        agent_prompt = self.prompt_template.format(current_time=get_current_local_datetime(), saved_information=saved_information)
        messages = [SystemMessage(agent_prompt)] + state['messages']
        response = await self.llm.ainvoke(messages)

        state_update_dict = dict()

        last_message = state['messages'][-1]
        if self.need_clear_current_task_history(last_message):
            # If the LLM called the save_info_and_clear_history tool, we:
            # 1. Save the information to the state. The information is in the 'artifact' field of the last message.
            # 2. Replace the current task history with the chat_history so now it only contains the messages of AI and
            # human interaction but not the tool calls and tool messages.
            state_update_dict['saved_information'] = last_message.artifact
            state_update_dict['messages'] = state['chat_history'] + [response]
        else:
            state_update_dict['messages'] = state['messages'] + [response]
        
        if not response.tool_calls:
            # This is the message to the user, so it need to be appended to the chat history
            state_update_dict['chat_history'] = response  # The reducer `add_messages` will append the response to the chat history

        if response.tool_calls:
            print("calling tools")

        return state_update_dict


# class SyncAgent(Agent):

#     def need_clear_current_task_history(self, last_message):
#         return (
#             last_message
#             and isinstance(last_message, ToolMessage)
#             and last_message.name == "save_info_and_clear_history"
#             and hasattr(last_message, 'artifact')
#         )

#     def __call__(self, state: PublicState):
#         saved_information = state.get("saved_information", [])
#         agent_prompt = self.prompt_template.format(current_time=get_current_local_datetime(), saved_information=saved_information)
#         messages = [SystemMessage(agent_prompt)] + state['messages']
#         response = self.llm.invoke(messages)

#         state_update_dict = dict()

#         last_message = state['messages'][-1]
#         if self.need_clear_current_task_history(last_message):
#             # If the LLM called the save_info_and_clear_history tool, we:
#             # 1. Save the information to the state. The information is in the 'artifact' field of the last message.
#             # 2. Replace the current task history with the chat_history so now it only contains the messages of AI and
#             # human interaction but not the tool calls and tool messages.
#             state_update_dict['saved_information'] = last_message.artifact
#             state_update_dict['messages'] = state['chat_history'] + [response]
#         else:
#             state_update_dict['messages'] = state['messages'] + [response]
        
#         if not response.tool_calls:
#             # This is the message to the user, so it need to be appended to the chat history
#             state_update_dict['chat_history'] = response  # The reducer `add_messages` will append the response to the chat history

#         if response.tool_calls:
#             print("calling tools")

#         return state_update_dict

