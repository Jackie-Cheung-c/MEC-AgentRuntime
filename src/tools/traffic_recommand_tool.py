"""Route planning for travel destinations using Amap (Gaode) Web Service APIs."""

from __future__ import annotations

import json
import math
from typing import Any
from urllib.error import HTTPError, URLError
from urllib.parse import urlencode
from urllib.request import Request, urlopen

from langchain.tools import tool
import os



AMAP_GEOCODING_URL = "https://restapi.amap.com/v3/geocode/geo"
AMAP_DRIVING_URL = "https://restapi.amap.com/v3/direction/driving"
AMAP_WALKING_URL = "https://restapi.amap.com/v3/direction/walking"
AMAP_BICYCLING_URL = "https://restapi.amap.com/v4/direction/bicycling"
AMAP_TRANSIT_URL = "https://restapi.amap.com/v3/direction/transit/integrated"
REQUEST_TIMEOUT_SECONDS = 12


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
        raise RuntimeError("高德路线服务暂时不可用，请稍后重试。") from error
    if data.get("errcode") == 0:
        return data
    if data.get("status") != "1":
        raise RuntimeError(data.get("info", "高德路线查询失败。"))
    return data


def _minutes(seconds: str | int | None) -> int:
    try:
        return max(1, math.ceil(int(seconds or 0) / 60))
    except (TypeError, ValueError):
        return 0


def _kilometers(meters: str | int | None) -> float:
    try:
        return int(meters or 0) / 1000
    except (TypeError, ValueError):
        return 0.0


def _geocode(place: str, key: str) -> dict[str, Any] | None:
    return _amap_get_json(AMAP_GEOCODING_URL, {"key": key, "address": place}).get("geocodes", [None])[0]


def _route_summary(mode: str, data: dict[str, Any]) -> str | None:
    if mode in {"driving", "walking"}:
        paths = data.get("route", {}).get("paths", [])
        if not paths:
            return None
        path = paths[0]
        return f"{mode}: {_kilometers(path.get('distance')):.1f} km，约 {_minutes(path.get('duration'))} 分钟"

    if mode == "bicycling":
        paths = data.get("data", {}).get("paths", [])
        if not paths:
            return None
        path = paths[0]
        return f"骑行: {_kilometers(path.get('distance')):.1f} km，约 {_minutes(path.get('duration'))} 分钟"

    transits = data.get("route", {}).get("transits", [])
    if not transits:
        return None
    transit = transits[0]
    cost = transit.get("cost")
    cost_text = f"，约 ¥{float(cost):g}" if cost not in (None, "", "0") else ""
    return f"公交/地铁: {_kilometers(transit.get('distance')):.1f} km，约 {_minutes(transit.get('duration'))} 分钟{cost_text}"


@tool
def plan_route(origin: str, destination: str) -> str:
    """从出发地到目的地规划驾车、公交地铁、步行和骑行路线；跨城时给出交通方式建议。"""
    origin = origin.strip()
    destination = destination.strip()
    if not origin or not destination:
        return "请同时提供出发地和目的地，例如“广州南站”到“广州塔”。"

    try:
        key = _amap_key()
        origin_info = _geocode(origin, key)
        destination_info = _geocode(destination, key)
        if not origin_info or not destination_info:
            return "未能识别出发地或目的地。请使用更完整的地点名称，例如“广州南站”或“北京故宫”。"

        origin_location = origin_info["location"]
        destination_location = destination_info["location"]
        common_params = {"key": key, "origin": origin_location, "destination": destination_location}
        results: dict[str, str] = {}

        route_requests = {
            "driving": (AMAP_DRIVING_URL, common_params),
            "walking": (AMAP_WALKING_URL, common_params),
            "bicycling": (AMAP_BICYCLING_URL, common_params),
        }
        for mode, (url, params) in route_requests.items():
            try:
                summary = _route_summary(mode, _amap_get_json(url, params))
                if summary:
                    results[mode] = summary
            except RuntimeError:
                continue

        origin_city = origin_info.get("citycode")
        destination_city = destination_info.get("citycode")
        if origin_city and destination_city:
            try:
                transit_params = {
                    **common_params,
                    "city": origin_city,
                    "cityd": destination_city,
                    "strategy": 0,
                }
                summary = _route_summary("transit", _amap_get_json(AMAP_TRANSIT_URL, transit_params))
                if summary:
                    results["transit"] = summary
            except RuntimeError:
                pass
    except RuntimeError as error:
        return str(error)

    if not results:
        return "暂时未能获取可用路线，请稍后重试或使用更具体的地点名称。"

    origin_name = origin_info.get("formatted_address", origin)
    destination_name = destination_info.get("formatted_address", destination)
    lines = [f"从“{origin_name}”到“{destination_name}”的路线方案："]
    labels = {"transit": "公交/地铁", "driving": "驾车", "bicycling": "骑行", "walking": "步行"}
    for mode in ("transit", "driving", "bicycling", "walking"):
        if mode in results:
            lines.append(f"- {results[mode].replace(mode + ': ', labels.get(mode, mode) + ': ')}")

    if origin_info.get("adcode", "")[:2] != destination_info.get("adcode", "")[:2]:
        lines.append(
            "- 跨城建议：优先比较高铁与飞机的实时班次、票价和到站时间；若两地较近，可比较长途大巴。"
        )
        lines.append("  高德路线接口不提供可靠的实时航班、火车或长途大巴票务数据，请在 12306、航空公司或客运平台确认。")
    return "\n".join(lines)
