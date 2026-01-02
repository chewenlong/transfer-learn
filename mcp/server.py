import json
import os

import requests
from typing import Any
from mcp.server.fastmcp import FastMCP

# 初始化 MCP 服务器
mcp = FastMCP("WeatherServer")

# 和风天气 API 配置
API_KEY = os.getenv("HE_FENG_KEY")  # 替换为你自己的和风天气API密钥
BASE_URL = "https://kd6e4e56cv.re.qweatherapi.com/v7/weather/now?location={}"
USER_AGENT = "weather-app/2.0"


def fetch_weather(city_code: str) -> dict[str, Any]:
    """
    从和风天气 API 获取当前天气信息。
    :param city_code: 城市编码（例如：北京=101010100）
    :return: 天气数据字典；若出错返回包含 error 信息的字典
    """
    url = BASE_URL.format(city_code)
    headers = {
        "User-Agent": USER_AGENT,
        "X-QW-Api-Key": API_KEY,
    }

    try:
        response = requests.get(url, headers=headers, timeout=20)
        response.raise_for_status()
        data = response.json()
        if data.get("code") != "200":
            return {"error": f"API返回错误代码: {data.get('code', '未知')}"}
        return data
    except requests.HTTPError as e:
        return {"error": f"HTTP错误: {e.response.status_code}"}
    except Exception as e:
        return {"error": f"请求失败: {str(e)}"}


def format_weather(data: dict[str, Any] | str) -> str:
    """
    将天气数据格式化为易读文本。
    :param data: 天气数据（字典或 JSON 字符串）
    :return: 格式化后的天气信息字符串
    """
    if isinstance(data, str):
        try:
            data = json.loads(data)
        except Exception as e:
            return f"无法解析天气数据: {e}"

    if "error" in data:
        return f"⚠️ {data['error']}"

    now = data.get("now", {})
    city_info = data.get("refer", {}).get("locations", [{}])[0] if data.get("refer") else {}

    city_name = city_info.get("name", "未知城市")
    country = city_info.get("country", "未知国家")
    temp = now.get("temp", "N/A")
    humidity = now.get("humidity", "N/A")
    text = now.get("text", "未知天气")
    wind_dir = now.get("windDir", "未知")
    wind_speed = now.get("windSpeed", "N/A")

    return (
        f"🌍 {city_name}, {country}\n"
        f"🌡 温度: {temp}°C\n"
        f"💧 湿度: {humidity}%\n"
        f"🌬 风向: {wind_dir}\n"
        f"💨 风速: {wind_speed} km/h\n"
        f"🌤 天气: {text}\n"
    )


@mcp.tool()
def query_weather(city_code: str) -> str:
    """
    输入城市编码（如北京=101010100），返回当前天气信息。
    :param city_code: 城市编码
    :return: 格式化后的天气信息
    """
    data = fetch_weather(city_code)
    return format_weather(data)


if __name__ == "__main__":
    # 启动 MCP 服务
    mcp.run(transport="stdio")
