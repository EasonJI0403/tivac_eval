# Chain → Firms QA Dataset

This dataset contains question-answer pairs for evaluating LLMs on the **Chain → Firms** task, which is the reverse of the Firm→Chains task. Given a value chain (產業鏈), the model must list all companies that belong to that chain.

## Generated Files

### `chain_firms_qa.jsonl`
- **Description**: All chains with all companies (local + foreign)
- **Chains**: 47 value chains
- **Total Companies**: 5,225 unique companies
  - Local: 4,020 (77%)
  - Foreign: 1,205 (23%)

### `chain_firms_qa_local.jsonl`
- **Description**: All chains with local (Taiwan) companies only
- **Chains**: 47 value chains
- **Total Companies**: 4,020 local companies only

### Chain Size Distribution

| Size Category | Company Range | Number of Chains | Percentage |
|---------------|---------------|------------------|------------|
| Medium        | 11-50         | 8                | 17.0%      |
| Large         | 51-100        | 22               | 46.8%      |
| X-Large       | >100          | 17               | 36.2%      |

**Note**: No chains with ≤10 companies exist in this dataset (all chains are large).

## Data Format

Each line in the JSONL file contains one QA pair:

```json
{
  "question": "列出產業鏈 人工智慧產業鏈 包含的所有公司。",
  "chain": "人工智慧產業鏈",
  "answer": [
    "91APP*-KY",
    "Anthropic",
    "Google",
    "OpenAI",
    ...
  ],
  "answer_count": 104,
  "local_companies": ["91APP*-KY", "中保科", ...],
  "foreign_companies": ["Anthropic", "Google", "OpenAI", ...],
  "local_count": 80,
  "foreign_count": 24
}
```

### Fields

- **question**: The question in Chinese asking for all companies in the chain
- **chain**: The value chain name
- **answer**: Complete list of all companies (sorted alphabetically)
- **answer_count**: Total number of companies
- **local_companies**: List of local/domestic companies only
- **foreign_companies**: List of foreign companies only
- **local_count**: Number of local companies
- **foreign_count**: Number of foreign companies

## Key Characteristics

### Task Difficulty

This task is **significantly more challenging** than Firm→Chains because:

1. **Many-to-One Relationship**: Each chain has many companies (11-341 companies)
2. **Large Answer Sets**: Average ~85-110 companies per chain
3. **Recall Challenge**: Models must remember/retrieve dozens to hundreds of companies
4. **Precision Challenge**: Easy to hallucinate company names

### Evaluation Focus

This dataset is ideal for evaluating:

1. **Comprehensive Knowledge**: Does the model know the full scope of each chain?
2. **Recall Capability**: Can it retrieve many related entities?
3. **Precision**: Does it avoid making up companies?
4. **Domain Coverage**: Which industries does the model know well?

### Notable Chains

**Largest Chains (>100 companies):**
- 其他產業鏈: 341 companies
- 生技醫療產業鏈: 242 companies
- 電子零組件產業鏈: 199 companies
- 資訊服務產業鏈: 148 companies
- 物流產業鏈: 135 companies

**Medium-Sized Chains (20-50 companies):**
- 生物辨識產業鏈: 47 companies
- 紡織產業鏈: 42 companies
- 遊戲產業鏈: 36 companies

**AI & Tech Chains:**
- 人工智慧產業鏈: 104 companies
- 半導體產業鏈: 118 companies
- 雲端運算產業鏈: 74 companies

## Generation Script

Use `generate_chain_to_firms_qa.py` to regenerate or customize the dataset:

### Basic Usage

```bash
# Generate both datasets (all companies + local only) - default
python generate_chain_to_firms_qa.py

# Generate only all-companies dataset
python generate_chain_to_firms_qa.py --no-local

# Generate only local companies dataset  
python generate_chain_to_firms_qa.py --no-all

# Only chains with at least 20 companies
python generate_chain_to_firms_qa.py --min-companies 20
```

### Advanced Options

```bash
# Export detailed statistics
python generate_chain_to_firms_qa.py --export-stats

# Custom output files
python generate_chain_to_firms_qa.py \
  --output custom_all.jsonl \
  --local-output custom_local.jsonl
```

## Evaluation Metrics

This dataset can be used to evaluate:

### Standard Metrics

1. **Recall**: What fraction of actual companies were found?
   - `Recall = |Predicted ∩ Actual| / |Actual|`
   - Critical for this task (many companies to recall)

2. **Precision**: What fraction of predicted companies are correct?
   - `Precision = |Predicted ∩ Actual| / |Predicted|`
   - Important to avoid hallucination

3. **F1 Score**: Harmonic mean of precision and recall
   - `F1 = 2 × (Precision × Recall) / (Precision + Recall)`

4. **mAP**: For ranked/ordered predictions

### Specialized Metrics

5. **Coverage**: Percentage of chains with recall > threshold (e.g., >0.5)
6. **Top-K Accuracy**: Did model find the K most representative companies?
7. **By Company Type**: Performance on local vs foreign companies (use `chain_firms_qa_local.jsonl` for local-only evaluation)

## Example QA Pairs

### Medium-Sized Chain Example
```json
{
  "question": "列出產業鏈 交通運輸及航運產業鏈 包含的所有公司。",
  "chain": "交通運輸及航運產業鏈",
  "answer": [
    "Angelicoussis Shipping Group",
    "Golden Ocean Group",
    "中保科",
    "中國鐵路總公司",
    "中櫃",
    "...64 companies total..."
  ],
  "answer_count": 64,
  "local_count": 39,
  "foreign_count": 25
}
```

### Large Chain Example
```json
{
  "question": "列出產業鏈 人工智慧產業鏈 包含的所有公司。",
  "chain": "人工智慧產業鏈",
  "answer": [
    "91APP*-KY",
    "Anthropic",
    "Google",
    "OpenAI",
    "中保科",
    "...104 companies total..."
  ],
  "answer_count": 104,
  "local_count": 80,
  "foreign_count": 24
}
```

### X-Large Chain Example
```json
{
  "question": "列出產業鏈 其他產業鏈 包含的所有公司。",
  "chain": "其他產業鏈",
  "answer": [
    "三捷科技",
    "三洋電",
    "三貝德",
    "...341 companies total..."
  ],
  "answer_count": 341,
  "local_count": 341,
  "foreign_count": 0
}
```

## Neo4j Query

The data is extracted from Neo4j using:

```cypher
MATCH (vc:ValueChain)-[:CONTAINS*1..10]->(node)-[:INCLUDES]->(comp:Company)
RETURN DISTINCT vc.name AS value_chain_name,
                comp.name AS company_name,
                comp.is_foreign AS is_foreign
ORDER BY vc.name, comp.name
```

This query:
1. Finds all value chains (top-level categories)
2. Traverses 1-10 levels down through subcategories
3. Collects all companies connected via INCLUDES relationships
4. Returns unique chain-company pairs with foreign status

## Evaluation Challenges

### For LLMs

1. **Knowledge Breadth**: Must know hundreds of Taiwan companies
2. **Industry Classification**: Must correctly map companies to chains
3. **Completeness**: Must recall all companies, not just major ones
4. **No Hallucination**: Must not invent company names
5. **Multi-Chain Companies**: Some companies appear in multiple chains

### For RAG Systems

1. **Retrieval Coverage**: Must retrieve all relevant companies
2. **Ranking Quality**: Most important companies should rank higher
3. **Deduplication**: Handle companies in multiple chains
4. **Scalability**: Handle chains with 100+ companies

## Use Cases

1. **Knowledge Completeness Testing**: Test how comprehensive LLM knowledge is
2. **Recall Evaluation**: Test model's ability to retrieve many related entities
3. **RAG Benchmarking**: Evaluate retrieval systems on complex queries
4. **Domain Expertise**: Assess domain-specific knowledge depth
5. **Hallucination Detection**: Identify when models make up company names

## Comparison: Chain→Firms vs Firm→Chains

| Aspect | Firm→Chains | Chain→Firms |
|--------|-------------|-------------|
| **Questions** | 3,187 companies | 47 chains |
| **Avg Answer Size** | 1.4 chains | 107.6 companies |
| **Max Answer Size** | 23 chains | 341 companies |
| **Task Difficulty** | Easier | Much Harder |
| **Recall Challenge** | Low (few chains) | High (many companies) |
| **Precision Challenge** | Moderate | High (easy to hallucinate) |
| **Best For** | Basic knowledge test | Comprehensive knowledge test |

## Statistics Summary

- **Total Chains**: 47
- **Total Companies**: 5,055 unique
  - Local: 3,866 (76.5%)
  - Foreign: 1,189 (23.5%)
- **Average Companies per Chain**: 107.6
- **Median Companies per Chain**: 78
- **Largest Chain**: 其他產業鏈 (341 companies)
- **Smallest Chain**: ~11-20 companies (1 chain only)

## Performance Expectations

### Strong Performance (F1 > 0.6)
- Model has comprehensive Taiwan industry knowledge
- Can recall large sets of related entities
- Suitable for industry analysis applications

### Moderate Performance (0.3 < F1 < 0.6)
- Model knows major companies in each chain
- May miss smaller/niche companies
- Useful with RAG augmentation

### Weak Performance (F1 < 0.3)
- Limited domain knowledge
- Struggles with recall on large answer sets
- Requires fine-tuning or specialized RAG

## Notes

- All value chain and company names are in Traditional Chinese
- Answers are sorted alphabetically for consistency
- Foreign companies include both foreign HQ and foreign-registered companies
- Large chains (>100 companies) are particularly challenging
- Perfect exact matches are extremely rare on this dataset

## Citation

If you use this dataset in your research, please cite appropriately.
