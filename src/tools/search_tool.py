"""Attraction recommendations backed by Amap (Gaode) Web Service APIs."""

from __future__ import annotations

import json
from typing import Any
from urllib.error import HTTPError, URLError
from urllib.parse import urlencode
from urllib.request import Request, urlopen
import os
from langchain.tools import tool


AMAP_TEXT_SEARCH_URL = "https://restapi.amap.com/v3/place/text"
REQUEST_TIMEOUT_SECONDS = 10
SCENIC_SPOT_TYPE = "110000"


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
        raise RuntimeError("高德地图景点服务暂时不可用，请稍后重试。") from error

    if data.get("status") != "1":
        raise RuntimeError(data.get("info", "高德地图景点查询失败。"))
    return data


def _rating(poi: dict[str, Any]) -> float:
    try:
        return float(poi.get("biz_ext", {}).get("rating", 0) or 0)
    except (TypeError, ValueError):
        return 0.0


def _recommendation_reason(poi: dict[str, Any]) -> str:
    reasons: list[str] = []
    rating = _rating(poi)
    if rating:
        reasons.append(f"高德评分 {rating:g}")
    if poi.get("tel"):
        reasons.append("提供联系电话，便于出行前确认开放信息")
    business_area = poi.get("business_area")
    if business_area:
        reasons.append(f"位于{business_area}商圈附近，配套较便利")
    return "；".join(reasons) or "属于高德风景名胜分类，适合纳入当地游览计划"


@tool
def recommend_attractions(destination: str) -> str:
    """根据城市或省份推荐至少五处景点，数据来自高德地图。"""
    destination = destination.strip()
    if not destination:
        return "请输入要查询的城市或省份，例如“广州”或“广东省”。"

    try:
        key = _amap_key()
        pois: list[dict[str, Any]] = []
        for page in (1, 2):
            data = _amap_get_json(
                AMAP_TEXT_SEARCH_URL,
                {
                    "key": key,
                    "keywords": "风景名胜",
                    "types": SCENIC_SPOT_TYPE,
                    "city": destination,
                    "citylimit": "true",
                    "offset": 25,
                    "page": page,
                    "extensions": "all",
                },
            )
            pois.extend(data.get("pois", []))
    except RuntimeError as error:
        return str(error)

    unique_pois: list[dict[str, Any]] = []
    seen_names: set[str] = set()
    for poi in pois:
        name = poi.get("name", "").strip()
        if name and name not in seen_names:
            seen_names.add(name)
            unique_pois.append(poi)

    if not unique_pois:
        return f"未在“{destination}”找到景点数据，请检查城市或省份名称。"

    unique_pois.sort(key=lambda poi: (-_rating(poi), poi.get("name", "")))
    selected = unique_pois[:5]
    lines = [f"“{destination}”景点推荐（高德地图数据，按评分优先排序）："]
    for index, poi in enumerate(selected, start=1):
        rating = _rating(poi)
        rating_text = f"高德评分 {rating:g}" if rating else "暂无高德评分"
        address = poi.get("address") or "地址待确认"
        lines.append(
            f"{index}. {poi['name']}：{rating_text}。\n"
            f"   地址：{address}。推荐理由：{_recommendation_reason(poi)}。"
        )

    if len(selected) < 5:
        lines.append(f"当前仅找到 {len(selected)} 处可用景点数据，建议改用更具体的城市名称。")
    return "\n".join(lines)
