from states.state import PublicState
from langchain_core.tools import StructuredTool
from models.factory import LLMFactory
from langchain_core.messages import HumanMessage, ToolMessage
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

# # When streaming the graph the user message will get added to the 'chat_history' but not the 'current_task_history'
# # So we need to manually add the user message to the 'current_task_history' list
# # as the 'current_task_history' list will be used in the LLM invocation.
# def init_current_task_history(state: PublicState):
#     current_task_history = state.get("current_task_history", [])
#     if len(current_task_history) > 0:
#          current_task_history.append(state["chat_history"][-1])  # append the user message
#     # If the 'current_task_history' is empty, it means that this is the beginning of the conversation
#     # and the 'chat_history' only contain the first user message. We don't need the first user message
#     # in the 'current_task_history' list, as it will be integrated into the system prompt when later
#     # we use the 'current_task_history' to invoke the LLM.
#     return {"current_task_history": current_task_history}

class Agent:
    def __init__(self, model_name: str, temperature: float, prompt_template: str, tools: list[StructuredTool]):
        # Init the LLM only when creating the agent in graph initialization
        # so we don't need to create it every time the agent is invoked
        self.llm = LLMFactory.get_llm(model=model_name, temperature=temperature)
        if tools:
            self.llm = self.llm.bind_tools(tools)
        self.prompt_template = prompt_template


class AsyncAgent(Agent):

    def clear_task_history_callback(self, tool_message):
        return (
            tool_message
            and isinstance(tool_message, ToolMessage)
            and tool_message.name == "save_info_and_clear_history"
            and hasattr(tool_message, 'artifact')
        )

    async def __call__(self, state: PublicState):
        saved_information = state.get("saved_information") or []
        # Here this first_prompt equals to system prompt + the first user message.
        # Some LLM like gemini does not support system prompt, so we workaround like this
        first_prompt = HumanMessage(self.prompt_template.format(
            current_time=get_current_local_datetime(),
            saved_information=saved_information,
            first_user_message=state['chat_history'][0].content
        ))

        state_update_dict = dict()  # Record which fields of the state need to be updated

        tool_called = state.get("tool_called") or False
        tool_message = None
        current_task_history = state.get("current_task_history") or []
        if tool_called:
            # return from tool call, append the tool message
            tool_message = state['messages'][0]
            current_task_history.append(tool_message)
            tool_called = False
        else:
            # return from user, append the user message
            current_task_history.append(state['chat_history'][-1])

        # The first element of 'current_task_history' is the first user message, which is already in the 'first_prompt', so we don't need it
        prompt = [first_prompt] + current_task_history[1:]
        response = await self.llm.ainvoke(prompt)

        if response.tool_calls:
            # The LLM called a tool, we put the tool calling in the 'messages' field of the state.
            # It will be used later as the input of the ToolNode.
            tool_called = True
        else:
            # The LLM replies to the user, we need to append this message to the chat history
            state_update_dict['chat_history'] = response  # The reducer is `add_messages`

        # We should put the response in the 'messages' whatever the LLM call the tool or not for tool condition checking
        state_update_dict['messages'] = [response]

        if self.clear_task_history_callback(tool_message):
            # If the LLM called the save_info_and_clear_history tool, we:
            # 1. Save the information to the state. The information is in the 'artifact' field of the last message.
            # 2. Replace the current task history with the chat_history so now it only contains the messages of AI and
            # human interaction but not the tool calls and tool messages.
            state_update_dict['saved_information'] = tool_message.artifact
            current_task_history = state['chat_history']

        state_update_dict['current_task_history'] = current_task_history + [response]
        state_update_dict['tool_called'] = tool_called
        return state_update_dict

class SyncAgent(Agent):

    def clear_task_history_callback(self, tool_message):
        return (
            tool_message
            and isinstance(tool_message, ToolMessage)
            and tool_message.name == "save_info_and_clear_history"
            and hasattr(tool_message, 'artifact')
        )

    def __call__(self, state: PublicState):
        saved_information = state.get("saved_information") or []
        # Here this first_prompt equals to system prompt + the first user message.
        # Some LLM like gemini does not support system prompt, so we workaround like this
        first_prompt = HumanMessage(self.prompt_template.format(
            current_time=get_current_local_datetime(),
            saved_information=saved_information,
            first_user_message=state['chat_history'][0].content
        ))

        state_update_dict = dict()  # Record which fields of the state need to be updated

        tool_called = state.get("tool_called") or False
        tool_message = None
        current_task_history = state.get("current_task_history") or []
        if tool_called:
            # return from tool call, append the tool message
            tool_message = state['messages'][0]
            current_task_history.append(tool_message)
            tool_called = False
        else:
            # return from user, append the user message
            current_task_history.append(state['chat_history'][-1])

        # The first element of 'current_task_history' is the first user message, which is already in the 'first_prompt', so we don't need it
        prompt = [first_prompt] + current_task_history[1:]
        response = self.llm.invoke(prompt)

        if response.tool_calls:
            # The LLM called a tool, we put the tool calling in the 'messages' field of the state.
            # It will be used later as the input of the ToolNode.
            tool_called = True
        else:
            # The LLM replies to the user, we need to append this message to the chat history
            state_update_dict['chat_history'] = response  # The reducer is `add_messages`

        # We should put the response in the 'messages' whatever the LLM call the tool or not for tool condition checking
        state_update_dict['messages'] = [response]

        if self.clear_task_history_callback(tool_message):
            # If the LLM called the save_info_and_clear_history tool, we:
            # 1. Save the information to the state. The information is in the 'artifact' field of the last message.
            # 2. Replace the current task history with the chat_history so now it only contains the messages of AI and
            # human interaction but not the tool calls and tool messages.
            state_update_dict['saved_information'] = tool_message.artifact
            current_task_history = state['chat_history']

        state_update_dict['current_task_history'] = current_task_history + [response]
        state_update_dict['tool_called'] = tool_called
        return state_update_dict
