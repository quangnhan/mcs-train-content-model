"""Metric chung cho 2 notebook đánh giá marketing (baseline + trained).

Cùng định nghĩa: Faithfulness + Expansion + Marketing Vibe + run_batch.
Notebook chỉ cần tạo `JudgeLLM(client, model_id)` rồi gọi `run_batch(cases, judge_llm)`.
"""

from __future__ import annotations

import json
import re
import time
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Sequence, Tuple, Type

import pandas as pd
from deepeval.metrics import GEval
from deepeval.models import DeepEvalBaseLLM
from deepeval.test_case import LLMTestCase, LLMTestCaseParams


# --------------------------------------------------------------------------- #
# Judge wrapper
# --------------------------------------------------------------------------- #
class JudgeLLM(DeepEvalBaseLLM):
    """Bọc OpenAI/AzureOpenAI client thành DeepEval-compatible model."""

    def __init__(self, client, model_id: str) -> None:
        self._client = client
        self._model_id = model_id

    def get_model_name(self) -> str:
        return self._model_id

    def load_model(self):
        return self._client

    def generate(self, prompt: str, schema: Optional[Type] = None):
        if schema is not None:
            try:
                resp = self._client.beta.chat.completions.parse(
                    model=self._model_id,
                    messages=[{"role": "user", "content": prompt}],
                    response_format=schema,
                )
                return resp.choices[0].message.parsed
            except Exception:
                resp = self._client.chat.completions.create(
                    model=self._model_id,
                    messages=[{"role": "user", "content": prompt}],
                    response_format={"type": "json_object"},
                )
                raw = (resp.choices[0].message.content or "{}").strip()
                return schema.model_validate_json(raw)
        resp = self._client.chat.completions.create(
            model=self._model_id,
            messages=[{"role": "user", "content": prompt}],
        )
        return (resp.choices[0].message.content or "").strip()

    async def a_generate(self, prompt: str, schema: Optional[Type] = None):
        return self.generate(prompt, schema)


# --------------------------------------------------------------------------- #
# Faithfulness — rule-based entity presence
# --------------------------------------------------------------------------- #
class EntityPresenceRule:
    """Rule-based: trích entity từ seed và đo tỉ lệ xuất hiện trong output.

    Dùng độc lập trong notebook:
        rule = EntityPresenceRule()
        rule.extract_prices("giảm 25%, còn 149.000đ")   → ['25%', '149.000đ']
        rule.extract_specs("bánh 420g, ly 500ml")         → ['420g', '500ml']
        score, detail = rule.measure(seed_content, actual_output)
    """

    PRICE_RE = re.compile(
        r"(?:\d{1,3}(?:[.,]\d{3})+|\d+)\s*(?:(?:VNĐ|vnđ|vnd|đồng|đ|USD|triệu|tr\.?|k)\b|[%$])",
        re.IGNORECASE,
    )
    SPEC_RE = re.compile(
        r"\b\d+(?:[.,]\d+)?\s*(?:ml|lít|lit|l|cc|kg|g|kcal|cal)\b",
        re.IGNORECASE,
    )
    COMBO_RE = re.compile(
        r"\b(?:combo|set|deal|gói)\s*\d+|\d+\s*(?:món|ly|tô|bát|đĩa|phần|suất|người)\b",
        re.IGNORECASE,
    )
    QUOTED_RE = re.compile(r"[\"\"]([^\"\"]{2,80})[\"\"]")
    NAME_RE = re.compile(
        r"(?:tên|món|combo|set|menu|sản\s+phẩm|thương\s+hiệu|quán)\s*:\s*([^.\n;]{2,})",
        re.IGNORECASE,
    )

    def extract_prices(self, text: str) -> List[str]:
        return [m.group(0).strip() for m in self.PRICE_RE.finditer(text)]

    def extract_specs(self, text: str) -> List[str]:
        return [m.group(0).strip() for m in self.SPEC_RE.finditer(text)]

    def extract_combos(self, text: str) -> List[str]:
        return [m.group(0).strip() for m in self.COMBO_RE.finditer(text)]

    def extract_quoted(self, text: str) -> List[str]:
        return [m.group(1).strip() for m in self.QUOTED_RE.finditer(text)]

    def extract_names(self, text: str) -> List[str]:
        return [m.group(1).strip() for m in self.NAME_RE.finditer(text)]

    def extract_entities(self, text: str) -> List[str]:
        found = (
            self.extract_prices(text)
            + self.extract_specs(text)
            + self.extract_combos(text)
            + self.extract_quoted(text)
            + self.extract_names(text)
        )
        seen = set()
        out = []
        for e in found:
            k = e.lower()
            if k not in seen and len(e) >= 2:
                seen.add(k)
                out.append(e)
        return out

    def measure(self, seed: str, output: str) -> Tuple[float, Dict[str, Any]]:
        entities = self.extract_entities(seed)
        if not entities:
            return 1.0, {"entities": [], "matched": [], "note": "no entities"}
        out_lower = output.lower()
        matched = [e for e in entities if e.lower() in out_lower]
        score = len(matched) / len(entities)
        return score, {"entities": entities, "matched": matched}



class FaithfulnessGEval:
    """GEval metric kiểm tra output có chứa claim không được hỗ trợ bởi context.

    Dùng độc lập trong notebook để test riêng phần LLM judge:
        geval = FaithfulnessGEval(judge_llm)
        score, reason = geval.measure(input_title, seed_content, actual_output)
    """

    _CRITERIA = (
        "Đánh giá mức độ trung thực của bài viết marketing so với nội dung mồi (retrieval context). "
        "Trừ điểm nếu output chứa bất kỳ claim nào KHÔNG được hỗ trợ bởi context — "
        "kể cả claim mơ hồ, hype vô căn cứ, hoặc thông tin phóng đại. "
        "Chỉ các thông tin, con số, và tính năng có trong context mới được phép xuất hiện. "
        "Cho điểm cao nếu output chỉ dùng thông tin từ context, dù diễn giải bằng ngôn ngữ khác."
    )

    def __init__(self, judge_llm: JudgeLLM) -> None:
        self._metric = GEval(
            name="Faithfulness",
            criteria=self._CRITERIA,
            evaluation_params=[
                LLMTestCaseParams.ACTUAL_OUTPUT,
                LLMTestCaseParams.RETRIEVAL_CONTEXT,
            ],
            model=judge_llm,
            async_mode=False,
            verbose_mode=False,
        )

    def measure(self, input_title: str, seed_content: str, actual_output: str) -> Tuple[float, str]:
        case = LLMTestCase(
            input=input_title,
            actual_output=actual_output,
            retrieval_context=[seed_content],
        )
        self._metric.measure(case)
        score = float(self._metric.score or 0.0)
        reason = getattr(self._metric, "reason", "") or ""
        return score, reason


@dataclass
class FaithfulnessResult:
    rule_entity_score: float
    rule_detail: Dict[str, Any]
    llm_faithfulness_score: float
    llm_reason: str = ""
    combined_score: float = 0.0

    def __post_init__(self):
        self.combined_score = (
            0.5 * self.rule_entity_score + 0.5 * self.llm_faithfulness_score
        )
        
class FaithfulnessEvaluator:
    def __init__(self, judge_llm: JudgeLLM) -> None:
        self._geval = FaithfulnessGEval(judge_llm)

    def evaluate_one(
        self, input_title: str, seed_content: str, actual_output: str
    ) -> FaithfulnessResult:
        rule_score, rule_detail = EntityPresenceRule().measure(seed_content, actual_output)
        llm_score, llm_reason = self._geval.measure(input_title, seed_content, actual_output)
        return FaithfulnessResult(
            rule_entity_score=rule_score,
            rule_detail=rule_detail,
            llm_faithfulness_score=llm_score,
            llm_reason=llm_reason,
        )


# --------------------------------------------------------------------------- #
# Expansion Quality
# --------------------------------------------------------------------------- #
class ContentExpansionRule:
    """Rule-based: đo mức độ mở rộng nội dung từ thông tin thô sang marketing copy.

    Đánh giá 3 tín hiệu:
    - Benefit language: có diễn giải lợi ích cho khách hàng không?
    - Usage context: có ngữ cảnh sử dụng thực tế không?
    - Length ratio: output có dài hơn seed hợp lý không? (target 1.5x–2.5x)

    Dùng độc lập trong notebook:
        rule = ContentExpansionRule()
        score = rule.measure(seed_content, actual_output)
    """

    _BENEFIT_RE = re.compile(
        r"\b(giúp|tiết kiệm|tận hưởng|thưởng thức|cảm giác|trải nghiệm|"
        r"phù hợp|lý tưởng|dành cho|mang lại|tăng|cải thiện|giảm bớt|"
        r"không lo|yên tâm|hợp túi tiền|đáng đồng tiền)\b",
        re.IGNORECASE,
    )
    _CONTEXT_RE = re.compile(
        r"\b(khi|lúc|dịp|buổi|cuối tuần|sáng|chiều|tối|ngày|mùa|"
        r"cùng bạn|cùng gia đình|với bạn bè|sau khi|trước khi|"
        r"đi picnic|đi chơi|tụ tập|sum họp|thư giãn)\b",
        re.IGNORECASE,
    )

    def measure(self, seed: str, output: str) -> float:
        ratio = (len(output.strip()) + 1) / (len(seed.strip()) + 1)
        if ratio >= 2.5:
            length_score = 1.0
        elif ratio >= 1.5:
            length_score = 0.7
        elif ratio >= 1.0:
            length_score = 0.4
        else:
            length_score = 0.1

        benefit_count = len(self._BENEFIT_RE.findall(output))
        benefit_score = min(1.0, benefit_count / 3)

        context_count = len(self._CONTEXT_RE.findall(output))
        context_score = min(1.0, context_count / 2)

        return round(0.4 * length_score + 0.35 * benefit_score + 0.25 * context_score, 4)


# Alias để không vỡ code cũ nếu có notebook import trực tiếp
LengthExpansionRule = ContentExpansionRule


class ExpansionGEval:
    """GEval metric đánh giá khả năng mở rộng từ seed thành bài marketing hoàn chỉnh.

    Dùng độc lập trong notebook để test riêng phần LLM judge:
        geval = ExpansionGEval(judge_llm)
        score, reason = geval.measure(input_title, seed_content, actual_output)
    """

    _CRITERIA = (
        "Đánh giá khả năng mở rộng từ nội dung mồi thành một bài viết marketing hoàn chỉnh. "
        "Bài phải diễn giải rõ lợi ích cho khách hàng (benefits) và đưa ra ngữ cảnh sử dụng thực tế "
        "dựa trên các tính năng/thông tin thô trong phần context (nội dung mồi). "
        "Trừ điểm nếu chỉ lặp lại bullet kỹ thuật mà không giải thích giá trị."
    )

    def __init__(self, judge_llm: JudgeLLM) -> None:
        self._metric = GEval(
            name="Expansion Quality",
            criteria=self._CRITERIA,
            evaluation_params=[
                LLMTestCaseParams.INPUT,
                LLMTestCaseParams.ACTUAL_OUTPUT,
                LLMTestCaseParams.CONTEXT,
            ],
            model=judge_llm,
            async_mode=False,
            verbose_mode=False,
        )

    def measure(self, input_title: str, seed_content: str, actual_output: str) -> Tuple[float, str]:
        case = LLMTestCase(
            input=input_title,
            actual_output=actual_output,
            context=[seed_content],
        )
        self._metric.measure(case)
        score = float(self._metric.score or 0.0)
        reason = getattr(self._metric, "reason", "") or ""
        return score, reason


@dataclass
class ExpansionResult:
    llm_expansion_score: float
    llm_reason: str
    rule_length_score: float
    combined_score: float


class ExpansionQualityEvaluator:
    def __init__(self, judge_llm: JudgeLLM) -> None:
        self._geval = ExpansionGEval(judge_llm)

    def evaluate_one(
        self, input_title: str, seed_content: str, actual_output: str
    ) -> ExpansionResult:
        llm_score, llm_reason = self._geval.measure(input_title, seed_content, actual_output)
        rule_content = ContentExpansionRule().measure(seed_content, actual_output)
        combined = 0.9 * llm_score + 0.1 * rule_content
        return ExpansionResult(
            llm_expansion_score=llm_score,
            llm_reason=llm_reason,
            rule_length_score=rule_content,
            combined_score=combined,
        )


# --------------------------------------------------------------------------- #
# Marketing Vibe
# --------------------------------------------------------------------------- #
@dataclass
class MarketingVibeResult:
    llm_hook_tone_score: float
    llm_reason: str
    rule_structure_score: float
    rule_cta_score: float
    rule_combined: float
    combined_score: float


VIBE_RUBRIC_VI = """\
Bạn là biên tập viên cấp cao về marketing performance và content founder/operator
tại thị trường Việt Nam, chuyên Facebook-native, F&B, growth, và franchise.

Bạn ĐANG chấm một bài viết marketing tiếng Việt theo phong cách dataset chuẩn:
- Giọng trực tiếp, thực tế: quyết liệt, gần gũi, KHÔNG brand-speak hoặc AI template.
- Tư duy kinh doanh F&B: giá cụ thể, deal/combo, tiết kiệm chi phí, hoặc KPI xuất hiện tự nhiên.
- Persuasion grounded: thuyết phục dựa trên specifics (con số, deal, thời hạn), không hype rỗng.
- Facebook-native: đoạn ngắn, nhịp gãy, code-switching Việt–Anh tự nhiên.
- CTA chốt rõ, có token hành động cụ thể ("Comment FRANCHISE", "Inbox COMBO89").

Bạn KHÔNG chấm theo tiêu chuẩn văn học, sang trọng, hoặc sáng tạo nghệ thuật.
Một bài "đẹp như văn" nhưng thiếu tính thực tế và thiếu specifics PHẢI bị điểm thấp hơn
một bài thô-nhưng-thật-và-có-số.

Chấm 5 chiều, mỗi chiều là SỐ NGUYÊN 0..4:

D1 VOICE_AUTHENTICITY — Giọng thật, trực tiếp, KHÔNG brand-speak/AI tone/văn hoa.
                        F&B operator tone hoặc founder tone đều được chấp nhận.
D2 BUSINESS_GROUNDING — Bài có neo vào thực tế kinh doanh F&B:
                        giá cụ thể / deal / % giảm / combo / thời hạn / hoặc KPI.
                        KHÔNG yêu cầu ROAS/CPL — price anchoring là đủ.
D3 PERSUASION_GROUND  — Thuyết phục bằng specifics, KHÔNG hype rỗng.
D4 CTA_QUALITY        — CTA cụ thể, có token hành động, audience-fit, low-friction.
D5 FB_NATIVE_FLOW     — Nhịp Facebook tự nhiên, KHÔNG AI tell, KHÔNG văn essay.

Định nghĩa mốc cho MỖI chiều:
  4 = xuất sắc, đúng phong cách dataset, không lỗi.
  3 = tốt, làm đúng vai trò của chiều đó, lệch nhẹ hoặc thiếu một yếu tố nhỏ.
  2 = trung bình, thiếu rõ ràng hoặc chủ yếu generic.
  1 = yếu, chủ yếu generic/AI-template.
  0 = sai phong cách hoàn toàn (văn học hoá, hype rỗng, press release, hoặc thiếu hẳn).

Quy tắc chấm điểm (bắt buộc):
- Không cho cùng số ở cả 5 chiều trừ khi thực sự đồng nhất.
- Bài F&B marketing CHẤT LƯỢNG TỐT (có giá, deal, CTA rõ, nhịp tự nhiên) xứng đáng 3–4.
- Chỉ cho 2 khi chiều đó thực sự thiếu hoặc generic, không phải vì "không hoàn hảo".
- Nếu có lỗi nặng (hype rỗng, AI opener, CTA template), ÍT NHẤT MỘT chiều ≤ 1.

Trả về JSON đúng schema sau, KHÔNG kèm văn bản nào khác:
{
  "d1_founder_voice":      <int 0-4>,
  "d2_business_realism":   <int 0-4>,
  "d3_persuasion_ground":  <int 0-4>,
  "d4_cta_quality":        <int 0-4>,
  "d5_fb_native_flow":     <int 0-4>,
  "reason": "<≤ 320 ký tự, tiếng Việt, nêu 1 điểm mạnh + 1 điểm yếu cụ thể>"
}
"""

# D2 weight reduced (no longer requires hard KPI acronyms); D3 raised (most measurable in F&B copy)
VIBE_WEIGHTS = {"d1": 0.25, "d2": 0.15, "d3": 0.25, "d4": 0.20, "d5": 0.15}

HYPE_BLOCKLIST = [
    r"đỉnh cao",
    r"tuyệt vời nhất",
    r"số\s*1\s*việt\s*nam",
    r"cam kết\s*100\s*%",
    r"không thể bỏ lỡ",
    r"cơ hội vàng",
    r"đột phá",
    r"tiên phong",
    r"vô đối",
    r"chấn động",
    r"hoàn hảo nhất",
]
AI_OPENERS = [
    r"^\s*trong bối cảnh",
    r"^\s*hãy cùng khám phá",
    r"^\s*bạn có biết rằng",
    r"^\s*trong thế giới ngày nay",
    r"^\s*ngày nay,?\s",
]
_EMOJI_RE = re.compile("[\U0001f300-\U0001faff☀-➿]")


class FbStructureRule:
    """Rule-based: đo cấu trúc Facebook của bài viết marketing.

    Dùng độc lập trong notebook:
        rule = FbStructureRule()
        score, detail = rule.measure(actual_output)
    """

    def measure(self, md: str) -> Tuple[float, Dict[str, Any]]:
        paras = [p.strip() for p in re.split(r"\n\s*\n", md) if p.strip()]
        if not paras:
            return 0.0, {"reason": "empty"}
        avg_len = sum(len(p) for p in paras) / len(paras)
        short_para_ratio = sum(1 for p in paras if len(p) <= 320) / len(paras)
        has_list = bool(re.search(r"(?m)^\s*[-*•]\s+\S|^\s*\d+[.)]\s+\S", md))
        has_cta_line = bool(
            re.search(
                r"(?im)^\s*(comment|inbox|nhắn|đăng ký|đặt lịch|booking|nhận ngay)\b", md
            )
        )
        s = 0.0
        s += 0.35 if short_para_ratio >= 0.6 else 0.35 * short_para_ratio
        s += 0.25 if avg_len <= 280 else max(0.0, 0.25 * (1 - (avg_len - 280) / 400))
        s += 0.20 if has_list else 0.0
        s += 0.20 if has_cta_line else 0.0
        return min(1.0, s), {
            "avg_len": avg_len,
            "short_ratio": short_para_ratio,
            "has_list": has_list,
            "has_cta_line": has_cta_line,
        }


class VibePenaltyRule:
    """Rule-based: phát hiện các lỗi vibe cần trừ điểm (hype, AI opener, over-emoji).

    Dùng độc lập trong notebook:
        rule = VibePenaltyRule()
        penalty, detail = rule.measure(actual_output)
    """

    def measure(self, md: str) -> Tuple[float, Dict[str, Any]]:
        hype_hits = sum(len(re.findall(p, md, re.I)) for p in HYPE_BLOCKLIST)
        ai_open_hit = any(re.search(p, md, re.I | re.M) for p in AI_OPENERS)
        paras = [p for p in re.split(r"\n\s*\n", md) if p.strip()]
        over_emoji = any(len(_EMOJI_RE.findall(p)) / max(1, len(p)) > 1 / 40 for p in paras)
        bullets = re.findall(r"(?m)^\s*[-*•]\s+(\S+)", md)
        bullet_templated = len(bullets) >= 4 and len({b.lower() for b in bullets}) <= 2
        p = 0.0
        if hype_hits >= 2:
            p += 0.10
        if ai_open_hit:
            p += 0.08
        if over_emoji:
            p += 0.05
        if bullet_templated:
            p += 0.05
        return p, {
            "hype_hits": hype_hits,
            "ai_open": ai_open_hit,
            "over_emoji": over_emoji,
            "bullet_templated": bullet_templated,
        }


class MarketingVibeJudge:
    """LLM judge: chấm bài viết marketing theo 5 chiều D1–D5 và trả về weighted score.

    Tương tự FaithfulnessGEval / ExpansionGEval nhưng dùng structured JSON thay vì GEval
    vì cần 5 điểm riêng biệt với trọng số khác nhau — GEval chỉ trả 1 score duy nhất.

    Dùng độc lập trong notebook để test riêng phần LLM judge:
        judge = MarketingVibeJudge(judge_llm)
        score, reason = judge.measure(input_title, seed_content, actual_output)
    """

    @staticmethod
    def _clip04(v: Any) -> int:
        try:
            iv = int(round(float(v)))
        except Exception:
            iv = 0
        return max(0, min(4, iv))

    @staticmethod
    def _parse(raw: str) -> Dict[str, Any]:
        if not raw:
            return {}
        m = re.search(r"\{[\s\S]*\}", raw)
        s = m.group(0) if m else raw
        try:
            return json.loads(s)
        except Exception:
            return {}

    def __init__(self, judge_llm: JudgeLLM) -> None:
        self._judge = judge_llm

    def measure(
        self, input_title: str, seed_content: str, actual_output: str
    ) -> Tuple[float, str]:
        prompt = (
            VIBE_RUBRIC_VI
            + "\n\n[TIÊU ĐỀ / YÊU CẦU]\n"
            + (input_title or "").strip()
            + "\n\n[NỘI DUNG MỒI / CONTEXT]\n"
            + (seed_content or "").strip()
            + "\n\n[BÀI VIẾT CẦN CHẤM]\n"
            + (actual_output or "").strip()
            + "\n\nCHỈ TRẢ JSON, không thêm markdown/code-fence."
        )
        raw = self._judge.generate(prompt)
        obj = self._parse(raw if isinstance(raw, str) else str(raw))

        d1 = self._clip04(obj.get("d1_founder_voice"))
        d2 = self._clip04(obj.get("d2_business_realism"))
        d3 = self._clip04(obj.get("d3_persuasion_ground"))
        d4 = self._clip04(obj.get("d4_cta_quality"))
        d5 = self._clip04(obj.get("d5_fb_native_flow"))

        norm = {"d1": d1 / 4, "d2": d2 / 4, "d3": d3 / 4, "d4": d4 / 4, "d5": d5 / 4}
        score = sum(VIBE_WEIGHTS[k] * norm[k] for k in VIBE_WEIGHTS)

        reason_raw = str(obj.get("reason", "")).strip()
        reason = (reason_raw[:380] if reason_raw else "(no reason)") + f" | dims=({d1},{d2},{d3},{d4},{d5})"
        return score, reason


class MarketingVibeEvaluator:
    def __init__(self, judge_llm: JudgeLLM) -> None:
        self._judge = MarketingVibeJudge(judge_llm)

    def evaluate_one(
        self, input_title: str, seed_content: str, actual_output: str
    ) -> MarketingVibeResult:
        llm_vibe, llm_reason = self._judge.measure(input_title, seed_content, actual_output)
        fb_s, _ = FbStructureRule().measure(actual_output)
        pen, _ = VibePenaltyRule().measure(actual_output)
        combined = max(0.0, min(1.0, 0.85 * llm_vibe + 0.15 * fb_s - pen))

        # Append penalty info that was previously mixed into evaluate_one
        reason = llm_reason + f" pen={pen:.2f}"

        return MarketingVibeResult(
            llm_hook_tone_score=llm_vibe,
            llm_reason=reason,
            rule_structure_score=fb_s,
            rule_cta_score=fb_s,
            rule_combined=fb_s,
            combined_score=combined,
        )


# --------------------------------------------------------------------------- #
# Batch runner
# --------------------------------------------------------------------------- #
def run_batch(
    cases: Sequence[Dict[str, str]],
    judge_llm: JudgeLLM,
    progress_chunk: int = 10,
) -> pd.DataFrame:
    f_ev = FaithfulnessEvaluator(judge_llm)
    e_ev = ExpansionQualityEvaluator(judge_llm)
    m_ev = MarketingVibeEvaluator(judge_llm)
    rows: List[Dict[str, Any]] = []
    n = len(cases)
    pc = max(1, int(progress_chunk))
    chunk_t0: Optional[float] = None
    chunk_start_i = 1

    for i, c in enumerate(cases, start=1):
        if (i - 1) % pc == 0:
            chunk_t0 = time.perf_counter()
            chunk_start_i = i

        title, seed, out = c["input_title"], c["seed_content"], c["actual_output"]
        fr = f_ev.evaluate_one(title, seed, out)
        er = e_ev.evaluate_one(title, seed, out)
        mr = m_ev.evaluate_one(title, seed, out)
        rows.append(
            {
                "case_id": c.get("case_id", ""),
                "faithfulness_rule": fr.rule_entity_score,
                "faithfulness_llm": fr.llm_faithfulness_score,
                "faithfulness_combined": fr.combined_score,
                "expansion_llm": er.llm_expansion_score,
                "expansion_rule_length": er.rule_length_score,
                "expansion_combined": er.combined_score,
                "vibe_llm_hook": mr.llm_hook_tone_score,
                "vibe_rule_md_cta": mr.rule_combined,
                "vibe_combined": mr.combined_score,
                "faithfulness_llm_reason": (fr.llm_reason or "")[:500],
                "expansion_llm_reason": (er.llm_reason or "")[:500],
                "vibe_llm_reason": (mr.llm_reason or "")[:500],
            }
        )

        if chunk_t0 is not None and ((i % pc == 0) or (i == n)):
            elapsed = time.perf_counter() - chunk_t0
            print(
                f"Chunk {chunk_start_i}–{i}/{n} hoàn thành trong {elapsed:.1f}s",
                flush=True,
            )

    return pd.DataFrame(rows)
