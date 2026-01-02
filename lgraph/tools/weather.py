from typing import Optional

import requests
import json


from langchain_core.tools import tool
from pydantic import Field, BaseModel

# 和风天气开发服务 https://dev.qweather.com/
API_KEY = "83ba4ee61a7d4aa0b738e8fa132e6b19"  # 替换成你自己的和风天气API密钥
BASE_URL = "https://kd6e4e56cv.re.qweatherapi.com/v7/weather/now?location={}"

class WeatherCode(BaseModel):
    city: str = Field(description="The location code of the city")
@tool(args_schema=WeatherCode)
def get_weather(city: str) -> str:
    """
    查询即时天气函数
    :param city:必要参数，字符串类型，表示要查询的天气的城市的编码
    :return:API查询的天气结果，返回为字符串格式
    """
    url = BASE_URL.format(city)

    # 设置请求头
    headers = {
        "X-QW-Api-Key": API_KEY
    }

    # 发送GET请求
    response = requests.get(url, headers=headers)

    # 检查响应状态
    response.raise_for_status()

    # 解析JSON响应
    json_node = response.json()
    print(json_node)
    # weather_info = f"当前天气（地点编码：{city}）是{json_node['now']['text']}，温度是{json_node['now']['temp']}°C"
    return str(json_node['now'])

class MultiplyParams(BaseModel):
    a: int = Field(description="One of the parameters for multiplication")
    b: int = Field(description="Another one of the parameters for multiplication")

@tool(args_schema=MultiplyParams)
def multiply(a: int, b: int) -> Optional[int]:
    """将2个数相乘"""
    return a * b


class WeatherInfo(BaseModel):
    """Extracted weather info from a specific city"""
    text: str = Field(description="The weather in the city")
    windDir: str = Field(description="The wind direction of the city")
    windSpeed: str = Field(description="The wind speed of the city")

class QueryWeatherSchema(BaseModel):
    """Schema for query weather info by city code"""
    city_code: str = Field(description="The location code of the city")


@tool(args_schema=QueryWeatherSchema)
def query_weather_from_db(city_code: str):
    """Query weather info by city code from db"""
    print("query db：", city_code)
    # return f"{city_code},text:大雨，windDir:西北风"
    return "没有查询到任何信息"

@tool(args_schema=WeatherInfo)
def insert_weather_to_db(text, windDir, windSpeed):
    """Insert weather information into the database."""
    print("insert db：", text, windDir, windSpeed)
    return "已成功保存到数据库"


@tool(args_schema=QueryWeatherSchema)
def delete_weather_from_db(city_code: str):
    """Delete weather information from the database by city code."""
    print("delete db：", city_code)

@tool(args_schema=WeatherInfo)
def update_weather_to_db(text, windDir, windSpeed):
    """Update weather information into the database."""
    print("update db：", text, windDir, windSpeed)
# 使用示例
if __name__ == "__main__":
    try:
        result = get_weather("101010100")
        print(result)
    except Exception as e:
        print(f"获取天气信息失败: {e}")