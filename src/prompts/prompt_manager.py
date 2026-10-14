"""Prompt 管理器 - 统一管理所有提示词模板"""

from pathlib import Path
from typing import Dict, Optional


class PromptManager:
    """提示词管理器，支持缓存和热加载"""

    def __init__(self, prompts_dir: Optional[Path] = None):
        if prompts_dir is None:
            # 默认指向当前文件所在目录
            prompts_dir = Path(__file__).parent
        self.prompts_dir = prompts_dir.resolve()
        self._cache: Dict[str, str] = {}

    def get(self, name: str, refresh: bool = False) -> str:
        """
        获取提示词内容

        Args:
            name: 提示词文件名（不含 .md 后缀）
            refresh: 是否强制刷新缓存

        Returns:
            提示词文本内容

        Raises:
            FileNotFoundError: 当提示词文件不存在时
        """
        if refresh or name not in self._cache:
            file_path = self.prompts_dir / f"{name}.md"
            if not file_path.exists():
                raise FileNotFoundError(
                    f"Prompt file not found: {file_path}\n"
                    f"Available prompts in {self.prompts_dir}: "
                    f"{[f.stem for f in self.prompts_dir.glob('*.md')]}"
                )
            self._cache[name] = file_path.read_text(encoding="utf-8")
        return self._cache[name]

    def reload(self, name: str) -> str:
        """强制重新加载指定的提示词"""
        return self.get(name, refresh=True)

    def reload_all(self) -> None:
        """清除所有缓存，下次获取时重新加载"""
        self._cache.clear()

    def list_available(self) -> list:
        """列出所有可用的提示词文件"""
        return [f.stem for f in self.prompts_dir.glob("*.md")]


# 创建全局单例
prompt_manager = PromptManager()
