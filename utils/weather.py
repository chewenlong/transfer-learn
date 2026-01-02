import os

import requests
import json

from langchain_core.tools import tool



# 和风天气开发服务 https://dev.qweather.com/
API_KEY = os.getenv("HE_FENG_KEY")  # 替换成你自己的和风天气API密钥
BASE_URL = "https://kd6e4e56cv.re.qweatherapi.com/v7/weather/now?location={}"

@tool
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


# 使用示例
if __name__ == "__main__":
    try:
        result = get_weather("101010100")
        print(result)
    except Exception as e:
        print(f"获取天气信息失败: {e}")