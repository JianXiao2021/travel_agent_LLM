import os
from langchain_core.tools import tool
from typing import Annotated, List, Dict
import requests
import json
from dotenv import load_dotenv, find_dotenv

@tool
def get_location_coordinate(
    location: Annotated[str, "要获取地图的地点的坐标，以“,”分割，经度在前，纬度在后，如117.500244,40.417801"],
):
    """地图获取工具，根据输入的地点经纬度返回地图图片"""
    
    _ = load_dotenv(find_dotenv())
    amap_key = os.getenv("AMAP_API_KEY")
    base_url = "https://restapi.amap.com/v3/staticmap?"
    url = f"{base_url}key={amap_key}&location={location}&zoom=10"
    r = requests.get(url)

    image_data = r.content
    file_name = location.replace(".", "_").replace(",", "_") + ".png"
    with open(file_name, "wb") as f:
        f.write(image_data)
    return file_name

# test the tool
if __name__ == "__main__":
    print(get_location_coordinate.args_schema.schema())
    a=get_location_coordinate.invoke({"location": "117.500244,40.417801"})
    print(a)