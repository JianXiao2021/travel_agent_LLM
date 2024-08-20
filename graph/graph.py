# Reference: https://langchain-ai.github.io/langgraph/tutorials/introduction/#part-2-enhancing-the-chatbot-with-tools

from langgraph.checkpoint.aiosqlite import AsyncSqliteSaver
from agents.agents import AsyncAgent as MainAgent
# from langgraph.checkpoint.sqlite import SqliteSaver
# from agents.agents import SyncAgent as MainAgent
from langgraph.graph import StateGraph, START
from states.state import PublicState
from agents.agents import init_current_task_history

from langgraph.prebuilt import ToolNode, tools_condition
from prompts.main import agent_prompt_template
from tools import *

memory = AsyncSqliteSaver.from_conn_string(":memory:")
# memory = SqliteSaver.from_conn_string(":memory:")


def create_graph(model_name):
    graph = StateGraph(PublicState)

    graph.add_node("init", init_current_task_history)

    tools = [web_search,
             get_batch_location_coordinates,
             get_attractions_information,
             route_planning,
             save_info_and_clear_history]

    travel_agent = MainAgent(
        model_name=model_name,
        temperature=0.7,
        prompt_template=agent_prompt_template,
        tools=tools)
    # Pass the "__call__" function in the ChatterAgent class to add_node. This function will be called when the node is invoked.
    # The function should be able to use extractor's 'llm' and 'prompt_template' attributes as they have been initialized when created
    # the extractor instance.
    graph.add_node("agent", travel_agent)

    tool_node = ToolNode(tools)
    graph.add_node("tools", tool_node)

    graph.add_edge(START, "init")
    graph.add_edge("init", "agent")
    graph.add_conditional_edges("agent", tools_condition)  # Will either direct to a specific tool in tools or to the END node
    graph.add_edge("tools", "agent")

    return graph

def init_app(model_name):
    graph = create_graph(model_name)
    app = graph.compile(checkpointer=memory)
    return app
