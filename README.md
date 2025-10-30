# Indexer Score Dashboard - The Graph Network

> **A simplified, delegator-focused scoring system for evaluating indexer performance on The Graph Network**

## 🎯 Overview

The Indexer Score Dashboard provides a transparent and simple way to evaluate indexers based on what matters most: **query fee generation** and **fair reward sharing**. We've removed complexity to focus on the metrics that directly impact delegators' returns.

---

## ✨ What Makes This Score Simple

Unlike complex multi-factor scoring systems, our approach is straightforward:

- **Single Performance Metric**: Query Fee Ratio (QFR) - how efficiently an indexer generates query fees relative to their allocated stake
- **Clear Reward Requirement**: Delegator rewards percentage - how much indexers share with their delegators
- **Transparent Thresholds**: Easy-to-understand tier requirements
- **Automatic Penalties**: Built-in protections against underserving indexers

---

## 🛡️ Protecting Delegators

### Delegator Rewards Thresholds

The scoring system is designed to **protect delegators** by enforcing fair reward distribution:

| Delegator Rewards % | Impact | Tier Eligibility |
|---------------------|--------|------------------|
| **≥ 30%** | ✅ Fair reward sharing | Can achieve **Excellent** tier |
| **10-29%** | ⚠️ Below recommended | Maximum **Fair** tier |
| **< 10%** | 🔴 Unfair to delegators | **Automatic Poor** classification |

**Key Protection:** Any indexer sharing less than 10% of rewards with delegators is **automatically classified as Poor**, regardless of their technical performance. This ensures delegators avoid indexers who keep more than 90% of rewards for themselves.

### Supporting Informed Decisions

The dashboard helps delegators by:
- 🎯 **Highlighting generous indexers** - Top tier requires ≥30% reward sharing
- 🚫 **Filtering out unfair indexers** - Automatic Poor rating for <10% sharing
- 📊 **Transparent metrics** - All data visible including exact reward percentages
- 🏆 **Easy comparison** - Sorted by tier and reward generosity

---

## 🌐 Network Health Focus

### Query Fee Ratio (QFR)

We care about the **health of The Graph Network** by evaluating indexers who actually serve the network through meaningful query fee generation.

**QFR Formula:**
```
QFR = Query Fees Earned / Total Allocated Stake
```

**Why QFR Matters:**
- Measures real network contribution through query serving
- Reflects indexer's subgraph curation quality
- Indicates actual demand for indexer's services
- Shows revenue generation capability

**Network Health Benefits:**
- Rewards indexers who serve active subgraphs
- Encourages proper subgraph selection
- Supports indexers providing value to dApp developers
- Promotes sustainable network economics

---

## ⚠️ Underserving Penalty

To maintain network decentralization and quality of service, we apply a **significant penalty** to indexers serving too few subgraphs.

### Penalty Details

**Threshold:** Indexers must serve at least **10 subgraphs**

**Penalty Formula:**
```
Penalty = 3.0 × (10 - number_of_subgraphs) / 10
```

**Penalty Examples:**

| Subgraphs Served | Penalty Applied | Impact |
|------------------|-----------------|--------|
| 1 subgraph | +2.70 points | 🔴 Severe |
| 3 subgraphs | +2.10 points | 🔴 High |
| 5 subgraphs | +1.50 points | 🟡 Moderate |
| 7 subgraphs | +0.90 points | 🟡 Low |
| 10+ subgraphs | 0.00 points | ✅ No penalty |

**Why This Matters:**
- Encourages network diversity
- Prevents over-concentration on few subgraphs
- Supports broader ecosystem health
- Incentivizes proper resource allocation

---

## 📐 Score Calculation

### Step 1: Calculate QFR
```
QFR = Query Fees Earned / Allocated Stake
```

### Step 2: Normalize QFR (1-10 scale)
```
Normalized QFR = 1 + 9 × (min(QFR, 0.3) / 0.3)
```
- QFR capped at 0.3 (30%) for fairness
- Scale: 1 (worst) to 10 (best)

### Step 3: Invert for Final Score (1 = best, 10 = worst)
```
Final Score = 11 - Normalized QFR
```

### Step 4: Apply Underserving Penalty (if applicable)
```
If (subgraphs < 10):
    Final Score = min(10.0, Final Score + Penalty)
```

### Step 5: Assign Performance Tier
```
If (Delegator Rewards < 10%):
    Tier = Poor 🔴
Else If (Final Score ≤ 9.92 AND Delegator Rewards ≥ 30%):
    Tier = Excellent 🟢
Else If (Final Score ≤ 9.97):
    Tier = Fair 🟡
Else:
    Tier = Poor 🔴
```

---

## 📊 Complete Formula

```
FINAL_SCORE = 11 - (1 + 9 × (min(QFR, 0.3) / 0.3)) + PENALTY

Where:
    QFR = Query Fees Earned / Allocated Stake
    PENALTY = 3.0 × (10 - subgraphs) / 10  (if subgraphs < 10, else 0)
    
TIER = {
    Poor 🔴:      if Delegator Rewards < 10% OR Final Score > 9.97
    Excellent 🟢: if Delegator Rewards ≥ 30% AND Final Score ≤ 9.92
    Fair 🟡:      otherwise
}
```

---

## 💡 Examples

### Example 1: Excellent Indexer

**Indexer: dataservices.eth**
- Query Fees: 222,106 GRT
- Allocated Stake: 87,323,802 GRT
- Subgraphs: 930
- Delegator Rewards: 81.14%

**Calculation:**
```
QFR = 222,106 / 87,323,802 = 0.00254 (0.254%)
Normalized QFR = 1 + 9 × (0.00254 / 0.3) = 1.08
Final Score = 11 - 1.08 = 9.92
Penalty = 0 (serves 930 subgraphs)
Delegator Rewards = 81.14% ≥ 30% ✅

Result: Excellent 🟢
```

**Why Excellent:**
- ✅ Generates meaningful query fees
- ✅ Shares 81.14% with delegators (exceptional!)
- ✅ Serves many subgraphs (930)
- ✅ Final score 9.92 ≤ threshold

---

### Example 2: Fair Indexer

**Indexer: streamingfastindexer.eth**
- Query Fees: 4,691,469 GRT
- Allocated Stake: 18,434,122 GRT
- Subgraphs: 750
- Delegator Rewards: 22.09%

**Calculation:**
```
QFR = 4,691,469 / 18,434,122 = 0.2545 (25.45%)
Normalized QFR = 1 + 9 × (0.2545 / 0.3) = 8.63
Final Score = 11 - 8.63 = 2.37
Penalty = 0 (serves 750 subgraphs)
Delegator Rewards = 22.09% < 30% ⚠️

Result: Fair 🟡
```

**Why Fair (not Excellent):**
- ✅ Excellent query fee generation (top performer!)
- ❌ Delegator rewards 22.09% < 30% threshold
- ✅ Serves many subgraphs (750)

---

### Example 3: Poor Indexer (Underserving)

**Indexer: Example Indexer**
- Query Fees: 1,000 GRT
- Allocated Stake: 5,000,000 GRT
- Subgraphs: 3
- Delegator Rewards: 35%

**Calculation:**
```
QFR = 1,000 / 5,000,000 = 0.0002 (0.02%)
Normalized QFR = 1 + 9 × (0.0002 / 0.3) = 1.01
Final Score = 11 - 1.01 = 9.99
Penalty = 3.0 × (10 - 3) / 10 = 2.10
Final Score with Penalty = min(10.0, 9.99 + 2.10) = 10.0

Result: Poor 🔴
```

**Why Poor:**
- ❌ Very low query fee generation
- ❌ Serves only 3 subgraphs (underserving)
- ✅ Good delegator rewards (35%) but not enough to overcome poor performance

---

### Example 4: Poor Indexer (Low Delegator Rewards)

**Indexer: pinax2.eth**
- Query Fees: 2,856,044 GRT
- Allocated Stake: 15,999,999 GRT
- Subgraphs: 600
- Delegator Rewards: 0.00%

**Calculation:**
```
QFR = 2,856,044 / 15,999,999 = 0.1785 (17.85%)
Normalized QFR = 1 + 9 × (0.1785 / 0.3) = 6.36
Final Score = 11 - 6.36 = 4.64
Penalty = 0 (serves 600 subgraphs)
Delegator Rewards = 0.00% < 10% 🔴

Result: Poor 🔴 (Automatic)
```

**Why Poor:**
- ✅ Excellent query fee generation (17.85%!)
- ✅ Serves many subgraphs (600)
- 🔴 **AUTOMATIC POOR**: Shares 0% with delegators
- **Critical Failure**: Keeps 100% of rewards for themselves

---

## 🏆 Performance Tiers Summary

| Tier | Requirements | Typical Profile |
|------|-------------|-----------------|
| **🟢 Excellent** | Score ≤9.92 + Delegator Rewards ≥30% | Top performers who share fairly |
| **🟡 Fair** | Score ≤9.97 OR Delegator Rewards 10-29% | Decent performance or moderate sharing |
| **🔴 Poor** | Score >9.97 OR Delegator Rewards <10% | Low performance or unfair sharing |

---

## 🎯 Key Takeaways

1. **Simplicity First**: One primary metric (QFR) plus reward fairness
2. **Delegator Protection**: Automatic Poor rating for <10% reward sharing
3. **Excellence Requires Fairness**: Need ≥30% reward sharing for top tier
4. **Network Health**: Rewards real query fee generation
5. **Diversity Incentive**: Penalty for serving too few subgraphs
6. **Transparent Scoring**: Clear formulas, no black boxes

---

## 📚 Resources

- **Live Dashboard**: [https://indexerscore.com](https://indexerscore.com)
- **Documentation**: View `docs.html` in the reports folder
- **GitHub Repository**: [https://github.com/pdiomede/indexerscore](https://github.com/pdiomede/indexerscore)
- **Forum Discussion**: [The Graph Forum Thread](https://forum.thegraph.com/t/introducing-the-indexer-score/6501)

---

## 🛠️ Technical Details

### Configuration

Key thresholds can be configured via `.env.costs` file:

```bash
# Delegator rewards threshold for Excellent tier
DELEGATOR_REWARDS_THRESHOLD=30.0

# Underserving subgraphs threshold
UNDERSERVING_SUBGRAPHS_COUNT=10

# Indexer size thresholds (GRT)
SMALL_INDEXER=1000000
MEDIUM_INDEXER=20000000
LARGE_INDEXER=50000000
```

### Running the Script

```bash
# Install dependencies
pip install -r requirements.txt

# Run the metrics fetch
python fetch_indexers_metrics.py

# Output files generated:
# - reports/index.html (dashboard)
# - reports/docs.html (documentation)
# - reports/indexers_output.csv (raw data)
```

---

## 📊 Current Network Statistics

**Total Indexers**: 82 with allocations  
**Performance Distribution**:
- 🟢 Excellent: 20 indexers (24.4%)
- 🟡 Fair: 11 indexers (13.4%)
- 🔴 Poor: 51 indexers (62.2%)

*Updated every 6 hours*

---

## 🤝 Contributing

This is an open-source project aimed at improving transparency in The Graph Network. Contributions, suggestions, and feedback are welcome!

---

## 📄 License

MIT License - See LICENSE file for details

---

## 👨‍💻 Author

**Paolo Diomede** ([@pdiomede](https://x.com/pdiomede))  
Made with ❤️ for The Graph ecosystem 👨‍🚀

---

**Version**: 2.0.0  
**Last Updated**: October 30, 2025
