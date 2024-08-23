import os
from langchain_core.tools import tool
from typing import Annotated, List, Dict
import requests
import json
from dotenv import load_dotenv, find_dotenv

@tool
def get_location_coordinate(
    location: Annotated[str, "要获取经纬度的地点名称"],
    city: Annotated[str, "要获取经纬度的地点所在的城市名称，必须是地级市，不能是县级市或村镇。如果能确定的话填写，不能确定可以不填"] = ""
) -> List[Dict[str, str]]:
    """位置获取工具。根据地点名称和城市名称获取该地点的经纬度。由于可能存在同名地点，所以返回的是一个包含所有同名地点详细地址和经纬度的列表"""
    
    _ = load_dotenv(find_dotenv())
    amap_key = os.getenv("AMAP_API_KEY")
    base_url = "https://restapi.amap.com/v3/geocode/geo?"
    url = f"{base_url}key={amap_key}&address={location}&city={city}"
    r = requests.get(url)
    result = r.json()
    if result['status'] == '0':
        error_msg = f"获取以下地点坐标失败：{location}。错误信息: {result['info']}。请检查参数格式是否符合规范（经度在前、纬度在后），地点和城市名称是否正确，或者考虑该名称是否是俗称，能否推断出其正式名称。"
        return error_msg
    location_coordinates = []
    for item in result['geocodes']:
        location_coordinates.append({
            "address": item['formatted_address'],
            "coordinate": item['location'],
            "citycode": item['citycode']
        })
    return location_coordinates

# test the tool
if __name__ == "__main__":
    print(get_location_coordinate.args_schema.schema())
    a=get_location_coordinate.invoke({"location": "迪士尼乐园", "city": "上海"})
    print(a)

# We don't need batch search as the LLM will use parallel tool calling to achieve the same effect
# @tool
# def get_batch_location_coordinates(
#     locations: Annotated[List[Dict[str, str]], "要获取经纬度的地点及其所在城市名称的列表。列表中每个元素是包含'location'和'city'两个键值对的字典，分别表示地点名称及其所在城市名称。城市名称如果不能确定可以不填"],
# ) -> Dict[str, List[Dict[str, str]]]:
#     """位置获取工具。根据多个地点名称和城市名称获取这些地点的经纬度。返回一个字典列表，字典的键是所查询的地点名称，值是包含所有同名地点详细地址、经纬度及其所在城市编码的列表"""
#     results = {}
#     for loc in locations:
#         location = loc.get('location')
#         city = loc.get('city')
#         location_coordinates = get_location_coordinate(location, city)
#         results[location] = location_coordinates
#     return results

# # test the tool
# if __name__ == "__main__":
#     print(get_batch_location_coordinates.args_schema.schema())
#     locations = [
#         {"location": "梦幻家园"},
#         {"location": "迪士尼乐园", "city": "上海"},
#         {"location": "北京路", "city": "广州"}
#     ]
#     a=get_batch_location_coordinates.invoke({"locations": locations})
#     print(a)

# # Unfortunatly, germini-1.5-pro does not support list as function input, so we have to use a string as input
# @tool
# def get_batch_location_coordinates(
#     locations_str: Annotated[str, """包含所有地点及其所在城市的字符串。格式为:'[{"location": "地点1", "city": "城市1"}, {"location": "地点2", "city": ""}, ...]'城市名称如果不能确定可以不填。"""],
# ) -> Dict[str, List[Dict[str, str]]]:
#     """位置获取工具。根据多个地点名称及其所在的城市名称获取这些地点的经纬度。返回一个字典列表，其中每个字典的键是所查询的地点名称，值是包含所有同名地点详细地址、经纬度及其所在城市编码的列表。需要分析列表从中挑选出对应正确地点的经纬度。"""
#     locations = json.loads(locations_str)
#     results = {}
#     for loc in locations:
#         location = loc.get('location')
#         city = loc.get('city')
#         location_coordinates = get_location_coordinate(location, city)
#         results[location] = location_coordinates
#     return results

# # test the tool
# if __name__ == "__main__":
#     print(get_batch_location_coordinates.args_schema.schema())
#     locations_str = """[{"location": "梦幻家园"}, {"location": "圣索非亚大教堂", "city": "哈尔滨"}, {"location": "北京路", "city": "广州"}]"""
#     a=get_batch_location_coordinates.invoke({"locations_str": locations_str})
#     print(a)