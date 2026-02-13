# Chain-to-Firms QA Dataset

## Overview

The chain-to-firms question generation script generates QA datasets for evaluating LLM performance on listing companies within industry value chains.

## Dataset Variants

1. **All Companies** (`chain_firms_qa.jsonl`): Contains all companies (both local and foreign)
2. **Local Companies Only** (`chain_firms_qa_local.jsonl`): Only Taiwan domestic companies

## Usage Examples

### Generate Both Datasets (Default)

```bash
python src/qa_generation/generate_chain_to_firms_qa.py
```

This will generate:
- `datasets/qa/chain_firms_qa.jsonl` (all companies)
- `datasets/qa/chain_firms_qa_local.jsonl` (local companies only)

### Generate Only All-Companies Dataset

```bash
python src/qa_generation/generate_chain_to_firms_qa.py --no-local
```

### Generate Only Local Companies Dataset

```bash
python src/qa_generation/generate_chain_to_firms_qa.py --no-all
```

## Output Format

### Local-Only QA Entry

```json
{
  "question": "列出產業鏈 半導體產業鏈 包含的所有公司。",
  "chain": "半導體產業鏈",
  "answer": ["台積電", "聯發科", "日月光"],
  "answer_count": 3,
  "local_companies": ["台積電", "聯發科", "日月光"],
  "foreign_companies": [],
  "local_count": 3,
  "foreign_count": 0
}
```

### All Companies QA Entry (for comparison)

```json
{
  "question": "列出產業鏈 半導體產業鏈 包含的所有公司。",
  "chain": "半導體產業鏈",
  "answer": ["台積電", "聯發科", "日月光", "Intel", "Samsung"],
  "answer_count": 5,
  "local_companies": ["台積電", "聯發科", "日月光"],
  "foreign_companies": ["Intel", "Samsung"],
  "local_count": 3,
  "foreign_count": 2
}
```

## Key Differences from All-Companies Dataset

| Feature | All Companies | Local Only |
|---------|--------------|------------|
| `answer` | All companies (local + foreign) | Local companies only |
| `answer_count` | Total count | Local count only |
| `foreign_companies` | List of foreign companies | Empty list |
| `foreign_count` | Foreign company count | 0 |
| Focus | Complete value chain coverage | Taiwan domestic participation |

## Alignment with Firm-to-Chains Dataset

This provides **symmetry** between the QA datasets:

### Firm-to-Chains
- `firm_chains_qa_local.jsonl` - Local companies → their chains
- `firm_chains_qa_foreign.jsonl` - Foreign companies → their chains

### Chain-to-Firms
- `chain_firms_qa.jsonl` - All chains → all companies
- `chain_firms_qa_local.jsonl` - All chains → local companies only

## Use Cases

1. **Taiwan-Focused Evaluation**: Test LLM knowledge specifically on Taiwan's domestic industry ecosystem
2. **Fair Comparison**: Compare with `firm_chains_qa_local.jsonl` results using the same company set
3. **Local Market Research**: Analyze local company participation across value chains
4. **Reduced Complexity**: Evaluate on smaller, more focused answer sets

## Metadata

Each generated dataset includes a `.meta.json` file with:
- Dataset type and variant
- System prompt for evaluation
- Uncertainty patterns
- Chain names included
- Statistics (total questions, total chains)

Example: `chain_firms_qa_local.meta.json`

## Evaluation

Use the local dataset with evaluation scripts:

```bash
python src/evaluation/evaluate_langchain_models.py \
  --dataset datasets/qa/chain_firms_qa_local.jsonl \
  --provider openai \
  --model gpt-4o-mini
```

## Notes

- The local version only includes chains that have at least one local company
- Chains with only foreign companies are excluded from the local variant
- The question text remains the same across all variants
- Answers are filtered to match the variant's focus
