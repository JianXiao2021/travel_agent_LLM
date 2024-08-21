from langchain_core.tools import tool
from typing import Annotated, Tuple

@tool(response_format="content_and_artifact")
def save_info_and_clear_history(
    infomation_to_save: Annotated[str, "需要保存的有用信息。主要是你调用的工具返回给你的信息中有用的部分。尽量详细。"],
) -> Tuple[str, str]:
    """信息保存工具。保存当前已获取的有用信息。"""
    # In fact, as LangGraph tool does not have the ability to update graph state, we only return the tool message to the main agent.
    # The update of the graph state will be handled in the main agent logic.
    content = "信息已保存。"
    # The LLM only get the content, the information to save will be saved in the 'artifact' field of the tool message.
    return content, infomation_to_save

# test the tool
if __name__ == "__main__":
    print(save_info_and_clear_history.args_schema.schema())
    tool_call =     {
        "name": "save_info_and_clear_history",
        "args": {"infomation_to_save": "2024年8月21日"},
        "id": "123",  # required
        "type": "tool_call",  # required
    }
    a=save_info_and_clear_history.invoke(tool_call)
    print(a)