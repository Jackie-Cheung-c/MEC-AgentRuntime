"""Weather lookup utility backed by the Open-Meteo public API."""

from __future__ import annotations

import json
import os
from typing import Any
from urllib.error import HTTPError, URLError
from urllib.parse import urlencode
from urllib.request import Request, urlopen
from langchain.tools import tool


AMAP_GEOCODING_URL = "https://restapi.amap.com/v3/geocode/geo"
AMAP_WEATHER_URL = "https://restapi.amap.com/v3/weather/weatherInfo"
REQUEST_TIMEOUT_SECONDS = 15

def _amap_key() -> str:
    key = os.getenv("GAODE_API_KEY")
    if not key:
        raise RuntimeError("未配置 GAODE_API_KEY，请设置高德地图 Web 服务 Key。")
    return key


def _amap_get_json(url: str, params: dict[str, Any]) -> dict[str, Any]:
    request = Request(
        f"{url}?{urlencode(params)}",
        headers={"User-Agent": "TravelTrace-Agent/1.0"},
    )
    try:
        with urlopen(request, timeout=REQUEST_TIMEOUT_SECONDS) as response:
            data = json.load(response)
    except (HTTPError, URLError, TimeoutError, json.JSONDecodeError) as error:
        raise RuntimeError("高德天气服务暂时不可用，请稍后重试。") from error
    if data.get("status") != "1":
        raise RuntimeError(data.get("info", "高德天气查询失败。"))
    return data


@tool
def get_weather(city: str) -> str:
    """通过高德地图查询目的地实时天气。"""
    city = city.strip()
    if not city:
        return "请输入城市或目的地名称。"
    try:
        key = _amap_key()
        geocodes = _amap_get_json(
            AMAP_GEOCODING_URL, {"key": key, "address": city}
        ).get("geocodes", [])
        if not geocodes:
            return f"未找到“{city}”，请使用更完整的地点名称。"
        adcode = geocodes[0].get("adcode")
        if not adcode:
            return f"未能识别“{city}”的行政区编码，暂时无法查询天气。"
        lives = _amap_get_json(
            AMAP_WEATHER_URL, {"key": key, "city": adcode, "extensions": "base"}
        ).get("lives", [])
    except RuntimeError as error:
        return str(error)
    if not lives:
        return f"暂时无法获取“{city}”的实时天气。"
    weather = lives[0]
    return (
        f"{weather.get('province', '')}{weather.get('city', '')}："
        f"{weather.get('weather', '未知')}，{weather.get('temperature', '未知')}°C，"
        f"湿度 {weather.get('humidity', '未知')}%，"
        f"{weather.get('winddirection', '未知')}风 {weather.get('windpower', '未知')}级。"
    )
