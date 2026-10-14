"""TravelOps 领域模型的共享配置"""
from pydantic import BaseModel, ConfigDict
#--------------------------负责业务逻辑和数据完整性----------------

class DomainModel(BaseModel):
    """Strict base model used by all public TravelOps contracts."""

    model_config = ConfigDict(
        extra="forbid", # 禁止其他字段
        str_strip_whitespace=True,
        validate_assignment=True,
    )
