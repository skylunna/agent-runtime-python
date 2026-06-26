import json
import logging 
from typing import Any, Callable, Awaitable
from dataclasses import dataclass


from ..schemas.retrieve import RetrieveRequest
from ..services.retrieval import retrieval_service


logger = logging.getLogger(__name__)


"""工具定义：给LLM看得 schema (OpenAI Tools 格式)"""

TOOL_DEFINITIONS = [
    {
        "type": "function",
        "function": {
            "name": "search_knowledge_base",
            "description": (
                "在医学知识库中检索与用户问题相关的临床指南、教科书或文献片段。"
                "返回 Top-K 相关片段及其来源(文档名、页码)。"
                "当你需要查询医学知识、用药指南、诊断标准、治疗方案时使用此工具。"
                "你可以将用户的口语化问题改写成更精准的检索 query。"
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "query": {
                        "type": "string",
                        "description": "检索 query。建议使用医学术语,可以包含多个关键词,例如:'二甲双胍 肾功能不全 用药调整'"
                    },
                    "top_k": {
                        "type": "integer",
                        "description": "返回的相关片段数量,默认 4,最大 10",
                        "default": 4,
                    }
                },
                "required": ["query"]
            }
        }
    },
    {
        "type": "function",
        "function": {
            "name": "get_patient_context",
            "description": (
                "查询患者的临床信息,包括年龄、性别、合并症、近期检验结果、用药史等。"
                "当用户的问题涉及具体患者(如'我有个 65 岁患者...'或会话中已提到患者 ID)时,"
                "应优先调用此工具获取患者全貌,避免泛泛而谈。"
                "MVP 阶段使用模拟数据,Phase 2 会对接真实 HIS 系统。"
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "patient_id": {
                        "type": "string",
                        "description": "患者 ID,例如 'P001'"
                    }
                },
                "required": ["patient_id"]
            }
        }
    }
]


@dataclass
class ToolResult:
    """统一的工具返回结构"""
    success: bool
    data: Any
    error: str | None = None

    def to_llm_string(self) -> str:
        """转成给 LLM 看的字符串 (JSON格式)"""
        if self.success:
            return json.dumps({"status": "ok", "data": self.data},
                              ensure_ascii=False)
        return json.dumps({"status": "error", "error": self.error},
                          ensure_ascii=False)
    

async def _tool_search_kb(args: dict, ctx: dict) -> ToolResult:
    """检索工具实现"""
    try:
        query = args["query"]
        top_k = min(args.get("top_k", 4), 10)
        kb_id = ctx.get("kb_id")    # 从 Agent 上下文取
        if not kb_id:
            return ToolResult(False, None, "Missing kb_id in context")
        
        result = await retrieval_service.retrieve(
            RetrieveRequest(kb_id=kb_id, query=query, top_k=top_k)
        )

        # 简单返回 (chunk_id 给 LLM 引用用)
        chunks = [
            {
                "chunk_id": c.chunk_id,
                "content": c.content,
                "document": c.document,
                "page": c.page,
                "score": round(c.score, 3),
            }
            for c in result.results
        ]
        return ToolResult(True, {"query": query, "chunks": chunks})
    except Exception as e:
        logger.exception("Tool search_knowledge_base failed")
        return ToolResult(False, None, str(e))
    

# 模拟患者数据 (MVP 用,Phase 2 替换成 HIS API)
_MOCK_PATIENTS = {
    "P001": {
        "patient_id": "P001",
        "age": 65,
        "sex": "male",
        "comorbidities": ["2型糖尿病", "高血压", "慢性肾病3a期"],
        "medications": [
            {"name": "二甲双胍", "dose": "0.5g bid", "duration": "3年"},
            {"name": "氨氯地平", "dose": "5mg qd", "duration": "2年"}
        ],
        "recent_labs": {
            "FBG": "8.2 mmol/L",
            "HbA1c": "7.8%",
            "eGFR": "45 mL/min/1.73m²",
            "Creatinine": "138 μmol/L"
        }
    },
    "P002": {
        "patient_id": "P002",
        "age": 52,
        "sex": "female",
        "comorbidities": ["2型糖尿病"],
        "medications": [],
        "recent_labs": {
            "FBG": "9.5 mmol/L",
            "HbA1c": "8.1%",
            "eGFR": "92 mL/min/1.73m²"
        }
    }
}


async def _tool_get_patient(args: dict, ctx: dict) -> ToolResult:
    """患者信息工具实现 (MVP mock)"""
    patient_id = args["patient_id"]
    patient = _MOCK_PATIENTS.get(patient_id)
    if not patient:
        return ToolResult(False, None, f"Patient not found: {patient_id}")
    return ToolResult(True, patient)


# ==== 工具注册表 ====
ToolFunc = Callable[[dict, dict], Awaitable[ToolResult]]


TOOL_REGISTRY: dict[str, ToolFunc] = {
    "search_knowledge_base": _tool_search_kb,
    "get_patient_context": _tool_get_patient,
}

async def execute_tool(name: str, args_json: str, ctx: dict) -> ToolResult:
    """统一的工具执行入口"""
    func = TOOL_REGISTRY.get(name)
    if not func:
        return ToolResult(False, None, f"Unknown tool: {name}")

    try:
        args = json.loads(args_json) if args_json else {}
    except json.JSONDecodeError as e:
        return ToolResult(False, None, f"Invalid arguments JSON: {e}")

    return await func(args, ctx)