import re
import json
from typing import Any, Tuple

from astrbot.api.event import filter, AstrMessageEvent
from astrbot.api.star import Context, Star
from astrbot.api.provider import LLMResponse
from openai.types.chat.chat_completion import ChatCompletion
try:
    # 谨慎引入，避免在未安装 google-genai 的环境下报错
    from google.genai.types import GenerateContentResponse
except Exception:  # pragma: no cover - 兼容无此依赖的运行环境
    GenerateContentResponse = None  # type: ignore


class R1Filter(Star):
    def __init__(self, context: Context):
        super().__init__(context)

    @filter.on_llm_response()
    async def resp(self, event: AstrMessageEvent, response: LLMResponse):
        cfg = self.context.get_config(umo=event.unified_msg_origin).get(
            "provider_settings", {}
        )
        show_reasoning = cfg.get("display_reasoning_text", False)

        # --- Gemini: 过滤/展示 thought:true 片段 ---
        # Gemini 可能在 parts 中注入 {"thought": true, "text": "..."}
        # 官方 SDK 默认不会返回此字段。
        if GenerateContentResponse is not None and isinstance(
            response.raw_completion, GenerateContentResponse
        ):
            thought_text, answer_text = self._extract_gemini_texts(
                response.raw_completion
            )

            if thought_text or answer_text:
                # 有明确的思考/正文分离信号，则按配置处理
                if show_reasoning:
                    merged = (
                        (f"🤔思考：{thought_text}\n\n" if thought_text else "")
                        + (answer_text or "")
                    ).strip()
                    if merged:
                        response.completion_text = merged
                        return
                else:
                    # 默认隐藏思考内容，仅保留正文
                    if answer_text:
                        response.completion_text = answer_text
                        return

        # --- 非 Gemini 或无明确 thought:true 情况 ---
        if show_reasoning:
            # 显示推理内容的处理逻辑
            if (
                response
                and response.raw_completion
                and isinstance(response.raw_completion, ChatCompletion)
                and len(response.raw_completion.choices) > 0
                and response.raw_completion.choices[0].message
            ):
                message = response.raw_completion.choices[0].message
                reasoning_content = ""  # 初始化 reasoning_content

                # 检查 Groq deepseek-r1-distill-llama-70b 模型的 'reasoning' 属性
                if hasattr(message, "reasoning") and message.reasoning:
                    reasoning_content = message.reasoning
                # 检查 DeepSeek deepseek-reasoner 模型的 'reasoning_content'
                elif (
                    hasattr(message, "reasoning_content") and message.reasoning_content
                ):
                    reasoning_content = message.reasoning_content

                if reasoning_content:
                    response.completion_text = (
                        f"🤔思考：{reasoning_content}\n\n{message.content}"
                    )
                else:
                    response.completion_text = message.content
        else:
            # 过滤推理标签的处理逻辑
            completion_text = response.completion_text

            # 检查并移除 <think> 标签
            if r"<think>" in completion_text or r"</think>" in completion_text:
                # 移除配对的标签及其内容
                completion_text = re.sub(
                    r"<think>.*?</think>", "", completion_text, flags=re.DOTALL
                ).strip()

                # 移除可能残留的单个标签
                completion_text = (
                    completion_text.replace(r"<think>", "")
                    .replace(r"</think>", "")
                    .strip()
                )

            response.completion_text = completion_text

    # ------------------------
    # helpers
    # ------------------------
    def _extract_gemini_texts(self, resp: Any) -> Tuple[str, str]:
        """
        从 GenerateContentResponse 中提取思考文本与正文文本。

        兼容在 parts 中注入 {"thought": true, "text": "..."} 的情况；
        若无法可靠解析，返回 ("", "").
        """
        try:
            candidates = getattr(resp, "candidates", None)
            if not candidates:
                return "", ""

            cand0 = candidates[0]
            content = getattr(cand0, "content", None)
            parts = getattr(content, "parts", None)
            if not parts:
                # 有些实现将纯文本聚合到 resp.text，此时直接返回为空，由上层字符串过滤兜底
                return "", ""

            thought_buf, answer_buf = [], []
            for p in parts:
                txt = getattr(p, "text", None)
                if not txt:
                    continue
                is_thought = False

                # 策略 1：直接属性
                if hasattr(p, "thought") and getattr(p, "thought") is True:
                    is_thought = True
                else:
                    # 策略 2：to_dict()/dict()/__dict__ 兜底
                    p_dict = None
                    for getter in ("to_dict", "dict"):
                        func = getattr(p, getter, None)
                        if callable(func):
                            try:
                                p_dict = func()
                                break
                            except Exception:
                                p_dict = None
                    if p_dict is None:
                        try:
                            # 某些 dataclass 可直接 __dict__
                            p_dict = getattr(p, "__dict__", None)
                        except Exception:
                            p_dict = None

                    if isinstance(p_dict, dict):
                        # 直接 thought 字段或嵌套 metadata.thought
                        if p_dict.get("thought") is True:
                            is_thought = True
                        elif isinstance(p_dict.get("metadata"), dict) and p_dict["metadata"].get(
                            "thought"
                        ) is True:
                            is_thought = True
                        else:
                            # 某些实现将原始 JSON 串保存在 data 等字段
                            raw_json = None
                            for k in ("data", "raw", "extra", "_raw"):
                                v = p_dict.get(k)
                                if isinstance(v, (str, bytes)):
                                    raw_json = v
                                    break
                            if raw_json:
                                try:
                                    d = json.loads(raw_json)
                                    if isinstance(d, dict) and d.get("thought") is True:
                                        is_thought = True
                                except Exception:
                                    pass

                if is_thought:
                    thought_buf.append(txt)
                else:
                    answer_buf.append(txt)

            return "\n".join(thought_buf).strip(), "\n".join(answer_buf).strip()
        except Exception:
            return "", ""
