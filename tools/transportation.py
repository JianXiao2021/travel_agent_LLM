import os
from langchain_core.tools import tool
from typing import Annotated
import requests
from dotenv import load_dotenv, find_dotenv

@tool
def route_planning(
    origin: Annotated[str, "出发点的经纬度，以“,”分割，如117.500244, 40.417801"],
    destination: Annotated[str, "目的地的经纬度，以“,”分割，如117.500244, 40.417801"],
    origin_city_code: Annotated[str, "出发点所在的城市编码"],
    dest_city_code: Annotated[str, "目的地所在的城市编码。如果出发点和目的地在同一个城市可以不填，若不在同一个城市则必填"] = None
) -> dict:
    """路线规划工具。规划综合各类公共交通方式（火车、公交、地铁）的交通方案，返回从出发点到目的地的步行距离、出租车费用以及公共交通方案列表。返回结果中，距离的单位都是米，时间的单位都是秒，费用的单位都是元。"""
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
        raise ValueError(f"Failed to get the route from {origin} to {destination}. Error message: {result['info']}")

    routes = {
        "walking_distance_from_origin_to_destination": result['route']['distance'],
        "taxi_cost": result['route']['taxi_cost'],
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
    a=route_planning.invoke({"origin": "116.481499,39.990475", "destination": "116.434446,39.90816", "city_code": "010"})
    print(a)