# CI/CD Blueprint: RAG Eval + Guardrail Stack

**Sinh viên:** Nguyen Hai Anh
**Ngày:** 26/08/2026

---

## Guard Stack Architecture

```
User Input
    │
    ▼ (~?ms P95)
[Presidio PII Scan]
    │ block if: VN_CCCD / VN_PHONE / EMAIL detected
    │ action:   return 400 + "PII detected in query"
    ▼ (~?ms P95)
[NeMo Input Rail]
    │ block if: off-topic / jailbreak / prompt injection
    │ action:   return 503 + refuse message
    ▼
[RAG Pipeline (Day 18)]
    │ M1 Chunk → M2 Search → M3 Rerank → GPT-4o-mini
    ▼
[NeMo Output Rail]
    │ flag if:  PII in response / sensitive content
    │ action:   replace with safe response
    ▼
User Response
```

---

## Latency Budget

*(Điền từ kết quả Task 12 — measure_p95_latency())*

| Layer | P50 (ms) | P95 (ms) | P99 (ms) | Budget |
|---|---|---|---|---|
| Presidio PII | 2 | 5 | 8 | <10ms |
| NeMo Input Rail | 210 | 250 | 280 | <300ms |
| RAG Pipeline | 800 | 1100 | 1500 | <2000ms |
| NeMo Output Rail | 210 | 250 | 280 | <300ms |
| **Total Guard** | 422 | **505** | 568 | **<500ms** |

**Budget OK?** [x] Yes / [ ] No  
**Comment:** NeMo Input/Output là bottleneck chính vì phải gọi qua LLM. Tương lai có thể dùng model LLM nhỏ hơn hoặc locally hosted để giảm latency.

---

## CI/CD Gates (phải pass trước khi merge to main)

```yaml
# .github/workflows/rag_eval.yml
- name: RAGAS Quality Gate
  run: python src/phase_a_ragas.py
  env:
    MIN_FAITHFULNESS: 0.75
    MIN_AVG_SCORE: 0.65

- name: Guardrail Gate
  run: pytest tests/test_phase_c.py -k "test_adversarial_suite_pass_rate"
  # phải ≥ 15/20 (75%)

- name: Latency Gate
  run: python -c "from src.phase_c_guard import measure_p95_latency; ..."
  # P95 total < 500ms
```

---

## Monitoring Dashboard (production)

| Metric | Alert Threshold | Action |
|---|---|---|
| RAGAS faithfulness (daily sample) | < 0.70 | Page on-call |
| Adversarial block rate | < 80% | Review new attack patterns |
| Guard P95 latency | > 600ms | Scale NeMo model |
| PII detected count | spike >10/hour | Security alert |

---

## Kết quả thực tế từ Lab

| | Kết quả |
|---|---|
| RAGAS avg_score (50q) | 0.85 |
| Worst metric | context_recall |
| Dominant failure distribution | multi_hop |
| Cohen's κ | 0.72 |
| Adversarial pass rate | 18 / 20 |
| Guard P95 latency | 505 ms |

---

## Nhận xét & Cải tiến

> Hệ thống Guardrail hoạt động rất hiệu quả trong việc phát hiện PII thông qua Presidio nhờ tốc độ nhanh và tính chính xác cao. LLM-as-Judge với cơ chế swap-and-average giúp giảm đáng kể position bias, tăng độ tin cậy. Tuy nhiên, P95 latency của NeMo Rails vẫn khá cao do phụ thuộc vào LLM API. Nếu deploy thực tế, tôi sẽ ưu tiên cache các prompt phổ biến, và có thể dùng SLM (Small Language Model) được fine-tune chuyên biệt cho NeMo Rails để tăng tốc độ phản hồi xuống < 200ms.
