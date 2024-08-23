from states.state import PublicState
from langchain_core.tools import StructuredTool
from models.factory import LLMFactory
from langchain_core.messages import HumanMessage, ToolMessage
from utils.helper import get_current_local_datetime

class Agent:
    def __init__(self, model_name: str, temperature: float, prompt_template: str, tools: list[StructuredTool]):
        # Init the LLM only when creating the agent in graph initialization
        # so we don't need to create it every time the agent is invoked
        self.llm = LLMFactory.get_llm(model=model_name, temperature=temperature)
        if tools:
            self.llm = self.llm.bind_tools(tools)
        self.prompt_template = prompt_template


class AsyncAgent(Agent):

    async def __call__(self, state: PublicState):
        # Here this first_prompt equals to system prompt + the first user message.
        # Some LLM like gemini does not support system prompt, so we workaround like this
        first_prompt = HumanMessage(self.prompt_template.format(
            current_time=get_current_local_datetime(),
            first_user_message=state['messages'][0].content
        ))
        prompt = [first_prompt] + state['messages'][1:]
        response = await self.llm.ainvoke(prompt)
        return {'messages': [response]}

class SyncAgent(Agent):

    def __call__(self, state: PublicState):
        # Here this first_prompt equals to system prompt + the first user message.
        # Some LLM like gemini does not support system prompt, so we workaround like this
        first_prompt = HumanMessage(self.prompt_template.format(
            current_time=get_current_local_datetime(),
            first_user_message=state['messages'][0].content
        ))
        prompt = [first_prompt] + state['messages'][1:]
        response = self.llm.invoke(prompt)
        return {'messages': [response]}