# Taiwan Value Chain QA Datasets - Complete Summary

This workspace contains comprehensive QA datasets for evaluating LLMs on Taiwan's industrial value chain knowledge.

## Overview

We have created **TWO types of QA tasks** with different difficulty levels:

### 1. Firm → Chains (Easier)
**Task**: Given a company, list all value chains it belongs to.
- **Questions**: 3,187 companies
- **Average answer size**: 1.4 chains
- **Difficulty**: ⭐⭐ Moderate

### 2. Chain → Firms (Harder)
**Task**: Given a value chain, list all companies in it.
- **Questions**: 47 chains
- **Average answer size**: 107.6 companies
- **Difficulty**: ⭐⭐⭐⭐ Very Hard

---

## Dataset Files

### Firm → Chains QA Datasets

| File | Description | Questions | Avg Answer Size |
|------|-------------|-----------|-----------------|
| `firm_chains_qa_local.jsonl` | Local companies only | 2,309 | 1.5 chains |
| `firm_chains_qa_foreign.jsonl` | Foreign companies only | 878 | 1.2 chains |
| `firm_chains_qa.jsonl` | All companies (legacy) | 3,187 | 1.4 chains |

**Total unique companies**: 3,187
- Local: 2,309 (72.5%)
- Foreign: 878 (27.5%)

**Chain distribution**:
- 71% single-chain companies
- 29% multi-chain companies
- Max chains per company: 23

### Chain → Firms QA Dataset

| File | Description | Questions | Avg Answer Size |
|------|-------------|-----------|-----------------|
| `chain_firms_qa.jsonl` | All value chains (all companies) | 47 | ~111 companies |
| `chain_firms_qa_local.jsonl` | All value chains (local only) | 47 | ~85 companies |

**Total unique chains**: 47
- All chains included regardless of size
- Range: 11-341 companies per chain
- Total company mentions: 5,225 (all) / 4,020 (local)

---

## Generation Scripts

### 1. Generate Firm → Chains QA
```bash
python generate_firm_to_chains_qa.py --password [NEO4J_PASSWORD]
```
- Generates: `firm_chains_qa_local.jsonl`, `firm_chains_qa_foreign.jsonl`
- Options: Filter by chain count, company type
- See: `FIRM_CHAINS_QA_README.md`

### 2. Generate Chain → Firms QA
```bash
python generate_chain_to_firms_qa.py
```
- Generates: `chain_firms_qa.jsonl`, `chain_firms_qa_local.jsonl`
- Options: Filter by company count, skip all/local variant
- See: `CHAIN_FIRMS_QA_README.md`

---

## Evaluation Scripts

### OpenAI API Evaluation

#### Single Model Evaluation
```bash
# Evaluate on Firm→Chains (local companies)
python evaluate_openai_on_firm_chains.py \
  --dataset firm_chains_qa_local.jsonl

# Quick test (5 samples)
python test_openai_evaluation.py
```

#### Model Comparison
```bash
# Compare multiple models
python compare_models.py \
  --dataset firm_chains_qa_local.jsonl \
  --models gpt-4o-mini gpt-4o \
  --max-samples 100
```

**See**: `EVALUATION_README.md` and `OPENAI_EVALUATION_QUICKSTART.md`

---

## Evaluation Metrics

All datasets support evaluation using:

1. **Recall**: `|Predicted ∩ Actual| / |Actual|`
   - More critical for Chain→Firms (large answer sets)

2. **Precision**: `|Predicted ∩ Actual| / |Predicted|`
   - Important to avoid hallucination

3. **F1 Score**: `2 × (P × R) / (P + R)`
   - Balanced metric

4. **mAP**: Mean Average Precision
   - For ranked predictions

5. **Exact Match Rate**: Perfectly correct predictions
   - Very strict metric

---

## Task Comparison

| Aspect | Firm → Chains | Chain → Firms |
|--------|---------------|---------------|
| **Questions** | 3,187 | 47 |
| **Avg Answers** | 1.4 | 107.6 |
| **Max Answers** | 23 | 341 |
| **Difficulty** | Moderate | Very Hard |
| **Best For** | Basic knowledge | Comprehensive knowledge |
| **Recall Challenge** | Low | Very High |
| **Precision Challenge** | Moderate | High |
| **Exact Match** | Achievable (~65%) | Very Rare (<5%) |

---

## Use Cases

### Firm → Chains
✓ Quick knowledge assessment
✓ Company classification
✓ Industry relationship mapping
✓ Multi-label classification benchmarks

### Chain → Firms
✓ Comprehensive knowledge testing
✓ Large-scale recall evaluation
✓ RAG system benchmarking
✓ Hallucination detection
✓ Domain expertise assessment

---

## Data Sources

All data is sourced from:
- **Neo4j Graph Database**: Taiwan value chain graph
- **Nodes**: ValueChain, Category, Subcategory, SubSubcategory, Company
- **Relationships**: CONTAINS (hierarchy), INCLUDES (company membership)
- **Query Pattern**: Traverse 1-10 levels through hierarchy

---

## Example QA Pairs

### Firm → Chains (Easier)

**Single-chain example:**
```json
{
  "question": "列出公司 A&D 所屬的所有產業鏈。",
  "company": "A&D",
  "answer": ["醫療器材產業鏈"],
  "answer_count": 1,
  "is_foreign": true
}
```

**Multi-chain example:**
```json
{
  "question": "列出公司 91APP*-KY 所屬的所有產業鏈。",
  "company": "91APP*-KY",
  "answer": [
    "人工智慧產業鏈",
    "大數據產業鏈",
    "金融科技產業鏈",
    "雲端運算產業鏈",
    "電子商務產業鏈"
  ],
  "answer_count": 5,
  "is_foreign": false
}
```

### Chain → Firms (Harder)

**Medium chain:**
```json
{
  "question": "列出產業鏈 交通運輸及航運產業鏈 包含的所有公司。",
  "chain": "交通運輸及航運產業鏈",
  "answer": ["Angelicoussis Shipping Group", "中保科", "...64 total"],
  "answer_count": 64,
  "local_count": 39,
  "foreign_count": 25
}
```

**Large chain:**
```json
{
  "question": "列出產業鏈 人工智慧產業鏈 包含的所有公司。",
  "chain": "人工智慧產業鏈",
  "answer": ["91APP*-KY", "Anthropic", "Google", "...104 total"],
  "answer_count": 104,
  "local_count": 80,
  "foreign_count": 24
}
```

---

## Directory Structure

```
scrape_tw_value_chains/
├── QA Datasets
│   ├── firm_chains_qa_local.jsonl (2,309 questions)
│   ├── firm_chains_qa_foreign.jsonl (878 questions)
│   ├── firm_chains_qa.jsonl (3,187 questions)
│   └── chain_firms_qa_large.jsonl (47 questions)
│
├── Generation Scripts
│   ├── generate_firm_to_chains_qa.py
│   └── generate_chain_to_firms_qa.py
│
├── Evaluation Scripts
│   ├── evaluate_openai_on_firm_chains.py
│   ├── test_openai_evaluation.py
│   └── compare_models.py
│
└── Documentation
    ├── FIRM_CHAINS_QA_README.md
    ├── CHAIN_FIRMS_QA_README.md
    ├── EVALUATION_README.md
    └── OPENAI_EVALUATION_QUICKSTART.md
```

---

## Quick Start

### 1. Generate Datasets (One-Time)
```bash
# Set Neo4j password
$env:NEO4J_PASSWORD = 'AIoT2018*'

# Generate Firm→Chains QA
python generate_firm_to_chains_qa.py

# Generate Chain→Firms QA
python generate_chain_to_firms_qa.py
```

### 2. Evaluate OpenAI Models
```bash
# Set API key
$env:OPENAI_API_KEY = 'sk-your-key'

# Quick test
python test_openai_evaluation.py

# Full evaluation
python evaluate_openai_on_firm_chains.py --dataset firm_chains_qa_local.jsonl
```

### 3. Compare Models
```bash
python compare_models.py \
  --dataset firm_chains_qa_local.jsonl \
  --models gpt-4o-mini gpt-4o \
  --max-samples 100
```

---

## Cost Estimates (OpenAI GPT-4o-mini)

| Dataset | Questions | Est. Cost |
|---------|-----------|-----------|
| Test (5 samples) | 5 | $0.01 |
| firm_chains_qa_local | 2,309 | $0.50-1.00 |
| firm_chains_qa_foreign | 878 | $0.20-0.40 |
| firm_chains_qa | 3,187 | $0.70-1.40 |
| chain_firms_qa_large | 47 | $0.05-0.10 |

**Note**: Chain→Firms is cheaper (fewer questions) but harder to answer correctly.

---

## Performance Expectations

### Firm → Chains

| Performance | F1 Score | Exact Match | Interpretation |
|-------------|----------|-------------|----------------|
| Strong | >0.75 | >0.65 | Excellent domain knowledge |
| Good | 0.60-0.75 | 0.45-0.65 | Solid knowledge, minor gaps |
| Moderate | 0.40-0.60 | 0.25-0.45 | Partial knowledge |
| Weak | <0.40 | <0.25 | Limited knowledge |

### Chain → Firms

| Performance | F1 Score | Recall | Interpretation |
|-------------|----------|--------|----------------|
| Strong | >0.50 | >0.60 | Comprehensive knowledge |
| Good | 0.35-0.50 | 0.45-0.60 | Good coverage |
| Moderate | 0.20-0.35 | 0.30-0.45 | Knows major players |
| Weak | <0.20 | <0.30 | Very limited knowledge |

---

## Key Findings

From initial testing:

1. **Firm→Chains is more feasible** for current LLMs
   - Smaller answer sets (1-5 chains typically)
   - Higher exact match rates possible
   - Good for quick assessment

2. **Chain→Firms is very challenging** even for strong models
   - Large answer sets (20-300 companies)
   - Requires comprehensive knowledge
   - Better for RAG evaluation
   - Exact matches extremely rare

3. **Local vs Foreign Companies**
   - Models tend to perform better on local companies
   - Foreign companies have less coverage
   - Consider separate evaluation tracks

4. **Chain Size Matters**
   - Performance degrades with chain size
   - Small chains: Recall >0.7 possible
   - Large chains: Recall <0.3 typical

---

## Future Work

Potential extensions:

1. **Additional Tasks**
   - Category→Subcategory hierarchy
   - Company→Company relationships
   - Multi-hop reasoning queries

2. **Additional Metrics**
   - Semantic similarity (beyond exact match)
   - Coverage metrics
   - Hallucination rates

3. **Additional Models**
   - Claude, Gemini, Llama evaluation
   - Fine-tuned models
   - RAG systems

4. **Temporal Analysis**
   - Track model improvements over time
   - Dataset versioning

---

## Citation

If you use these datasets in your research, please cite appropriately and acknowledge the Taiwan Economic Journal (TEJ) as the original data source.

---

## Contact & Support

For questions, issues, or contributions:
- Check the detailed README files for each dataset
- Review the evaluation documentation
- See example scripts and outputs

**Last Updated**: October 21, 2025
**Version**: 1.0
