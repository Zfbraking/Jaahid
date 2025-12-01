# Lightweight MCP-style adapter to standardize tools.
# Each tool exposes name, description, and a callable "invoke(input_dict) -> dict".

from typing import Callable, Dict, Any, List

class MCPTool:
    def __init__(self, name: str, description: str, func: Callable[[Dict[str, Any]], Dict[str, Any]]):
        self.name = name
        self.description = description
        self._func = func

    def invoke(self, payload: Dict[str, Any]) -> Dict[str, Any]:
        return self._func(payload)

class MCPRegistry:
    def __init__(self):
        self._tools: Dict[str, MCPTool] = {}

    def register(self, tool: MCPTool):
        self._tools[tool.name] = tool

    def get(self, name: str) -> MCPTool:
        return self._tools[name]

    def list(self) -> List[str]:
        return list(self._tools.keys())
