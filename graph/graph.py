# Reference: https://langchain-ai.github.io/langgraph/tutorials/introduction/#part-2-enhancing-the-chatbot-with-tools

from langgraph.checkpoint.aiosqlite import AsyncSqliteSaver
from langgraph.checkpoint.sqlite import SqliteSaver
from langgraph.graph import StateGraph, START
from states.state import PublicState

from langgraph.prebuilt import ToolNode, tools_condition
from prompts.main import agent_prompt_template
from tools import *


def create_graph(model_name, is_async=True):
    graph = StateGraph(PublicState)

    tools = [web_search,
             get_location_coordinate,
             get_attractions_information,
             route_planning,
             search_nearby_poi,
            #  save_info_and_clear_history,
            ]

    if is_async:
        from agents.agents import AsyncAgent as MainAgent
    else:
        from agents.agents import SyncAgent as MainAgent

    travel_agent = MainAgent(
        model_name=model_name,
        temperature=0,
        prompt_template=agent_prompt_template,
        tools=tools)
    # Pass the "__call__" function in the ChatterAgent class to add_node. This function will be called when the node is invoked.
    # The function should be able to use extractor's 'llm' and 'prompt_template' attributes as they have been initialized when created
    # the extractor instance.
    graph.add_node("agent", travel_agent)

    tool_node = ToolNode(tools)
    graph.add_node("tools", tool_node)

    # graph.add_edge(START, "init")
    # graph.add_edge("init", "agent")
    graph.add_edge(START, "agent")
    graph.add_conditional_edges("agent", tools_condition)  # Will either direct to a specific tool in tools or to the END node
    graph.add_edge("tools", "agent")

    return graph

def init_app(model_name, is_async=True):
    graph = create_graph(model_name, is_async)
    if is_async:
        memory = AsyncSqliteSaver.from_conn_string(":memory:")
    else:
        memory = SqliteSaver.from_conn_string(":memory:")
    app = graph.compile(checkpointer=memory)
    return app
