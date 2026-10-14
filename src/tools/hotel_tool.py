"""Find accommodations near a travel attraction using OpenStreetMap data."""

from __future__ import annotations

import json
import math
from typing import Any
from urllib.error import HTTPError, URLError
from urllib.parse import urlencode
from urllib.request import Request, urlopen
from langchain.tools import tool

# Amap implementation overrides the legacy OpenStreetMap tool above.
import os

AMAP_GEOCODING_URL = "https://restapi.amap.com/v3/geocode/geo"
AMAP_AROUND_URL = "https://restapi.amap.com/v3/place/around"
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
        raise RuntimeError("高德地图服务暂时不可用，请稍后重试。") from error
    if data.get("status") != "1":
        raise RuntimeError(data.get("info", "高德地图酒店查询失败。"))
    return data


def _amap_cost(poi: dict[str, Any]) -> float | None:
    try:
        value = poi.get("biz_ext", {}).get("cost", "")
        return float(value) if value not in (None, "") else None
    except (TypeError, ValueError):
        return None


@tool
def search_nearby_hotels(attraction: str) -> str:
    """通过高德地图查询景点附近酒店，并按价格从低到高排序。"""
    attraction = attraction.strip()
    if not attraction:
        return "请输入景点名称，例如“广州动物园”。"
    try:
        key = _amap_key()
        geocodes = _amap_get_json(
            AMAP_GEOCODING_URL, {"key": key, "address": attraction}
        ).get("geocodes", [])
        if not geocodes:
            return f"未找到“{attraction}”，请补充城市或使用更完整的景点名称。"
        destination = geocodes[0]
        pois = _amap_get_json(
            AMAP_AROUND_URL,
            {
                "key": key,
                "location": destination["location"],
                "keywords": "酒店",
                "types": "100000",
                "radius": 10000,
                "offset": 25,
                "page": 1,
                "extensions": "all",
            },
        ).get("pois", [])
    except RuntimeError as error:
        return str(error)
    if not pois:
        return f"在“{destination['formatted_address']}”附近未找到酒店。"

    pois.sort(
        key=lambda poi: (
            _amap_cost(poi) is None,
            _amap_cost(poi) if _amap_cost(poi) is not None else float("inf"),
            int(poi.get("distance", 0) or 0),
        )
    )
    lines = [f"“{destination['formatted_address']}”附近酒店推荐（按高德价格从低到高排序）："]
    for index, poi in enumerate(pois[:5], start=1):
        cost = _amap_cost(poi)
        price = f"约 ¥{cost:g}/晚" if cost is not None else "价格待确认"
        distance = poi.get("distance", "")
        distance_text = f"距景点约 {int(distance) / 1000:.1f} 公里" if str(distance).isdigit() else "距离待确认"
        rating = poi.get("biz_ext", {}).get("rating")
        reasons = []
        if rating:
            reasons.append(f"高德评分 {rating}")
        if str(distance).isdigit() and int(distance) <= 3000:
            reasons.append("距离景点较近，出行方便")
        else:
            reasons.append("适合短途车程前往景点")
        if poi.get("tel"):
            reasons.append("提供联系电话，可先确认房态")
        lines.append(
            f"{index}. {poi.get('name', '未命名酒店')}：{price}，{distance_text}。\n"
            f"   推荐理由：{'；'.join(reasons)}。"
        )
    return "\n".join(lines)
