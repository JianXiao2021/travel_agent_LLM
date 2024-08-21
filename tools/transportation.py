import os
from langchain_core.tools import tool
from typing import Annotated
import requests
from dotenv import load_dotenv, find_dotenv

@tool
def route_planning(
    origin: Annotated[str, "出发点的经纬度，以“,”分割，经度在前，纬度在后，如117.500244, 40.417801"],
    destination: Annotated[str, "目的地的经纬度，以“,”分割，经度在前，纬度在后，如117.500244, 40.417801"],
    origin_city_code: Annotated[str, "出发点所在的城市编码"],
    dest_city_code: Annotated[str, "目的地所在的城市编码。"]
) -> dict:
    """路线规划工具。规划综合各类公共交通方式（火车、公交、地铁）的交通方案，返回从出发点到目的地的步行距离、出租车费用以及公共交通方案列表。返回结果中，距离的单位都是米，时间的单位都是秒，费用的单位都是元。如果返回的公共交通方案列表为空，说明两个地点之间没有可用的公共交通方式。"""
    _ = load_dotenv(find_dotenv())
    amap_key = os.getenv("AMAP_API_KEY")
    base_url = "https://restapi.amap.com/v3/direction/transit/integrated?"
    if dest_city_code is None:
        url = f"{base_url}origin={origin}&destination={destination}&city={origin_city_code}&key={amap_key}"
    else:
        url = f"{base_url}origin={origin}&destination={destination}&city={origin_city_code}&cityd={dest_city_code}&key={amap_key}"
    r = requests.get(url)
    result = r.json()
    if result['status'] == '0':
        raise ValueError(f"获取从{origin}到{destination}的交通方案失败。请检查出发点、目的地和城市编码是否正确。错误信息: {result['info']}")

    routes = {
        "origin": result['route']['origin'],
        "destination": result['route']['destination'],
        "walking_distance": result['route']['distance'],
        "taxi_cost": result['route']['taxi_cost'] if result['route']['taxi_cost'] != '0' else "not available",
        "public_transport_options_list": []
    }

    for item in result['route']['transits']:
        routes['public_transport_options_list'].append({
            "cost": item['cost'],
            "duration": item['duration'],
            "walking_distance": item['walking_distance']
        })
    return routes

# test the tool
if __name__ == "__main__":
    print(route_planning.args_schema.schema())
    a=route_planning.invoke({
        "origin": "126.634,45.774",
        "destination": "127.954017,45.212528",
        "origin_city_code": "0451",
        "dest_city_code": "0451"})
    print(a)