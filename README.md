# Indexer Score Dashboard — The Graph Network
This project generates a detailed performance dashboard and CSV report for **indexers** on [The Graph Network](https://thegraph.com). It fetches on-chain data via The Graph Network, evaluates indexers using the custom **Indexer Score** metric, and builds a modern, responsive HTML dashboard.

**Live Dashboard:**  
🔗 [indexerscore.com](https://indexerscore.com)

🧪 This dashboard is part of [**Graph Tools Pro**](https://graphtools.pro), a community-driven initiative to provide useful, independent analytics tools for The Graph ecosystem.

---

## 📊 Features

- Scores indexers based on allocation efficiency and query fee generation
- Fetches indexer metadata including ENS and avatars
- Ranks indexers by score and assigns performance flags
- Generates:
  - A responsive `index.html` dashboard
  - A `CSV` file for offline data analysis
  - Daily JSON metric snapshots for historical tracking
  - A secondary HTML file explaining methodology (`docs.html`)
- Light/dark mode toggle and performance filter buttons
- Cached ENS results and clean logs

---

## 🧮 What Is the Indexer Score?

The **Indexer Score** is a synthetic performance metric for The Graph’s indexers. It combines two main factors:

- **AER (Allocation Efficiency Ratio)** – how effectively GRT is distributed across subgraphs (70% weight)
- **QFR (Query Fee Ratio)** – how many query fees are generated per GRT allocated (30% weight)

Scores are normalized and adjusted to reflect both performance and behavior. Final scores range from **1.0 (best)** to **10.0 (worst)**, and indexers are labeled as:

- 🟢 **Excellent** (1.0 – 1.25)
- 🟡 **Fair** (1.26 – 2.5)
- 🔴 **Poor** (2.51 – 10.0)

Indexers serving fewer than the minimum number of subgraphs also receive a penalty.

See the full documentation here:  
📘 **[Indexer Score Docs](https://indexerscore.com/docs.html)**  
📄 **[Whitepaper PDF](https://indexerscore.com/indexer_score_documentation_v1.1.0.pdf)**

---
# Abstract

This project generates PDF statements for **indexers** in [The Graph Network](https://thegraph.com). 
It fetches on-chain data via subgraph queries and creates a formatted, branded statement including metrics like stake, rewards, query fees, and delegator share.

---

## 📌 Features

- Fetches active indexers via The Graph Network
- Generates a local JSON with detailed indexer metrics
- Downloads and caches avatar images dynamically
- Outputs one PDF per indexer with full branding and metrics
- Supports avatar image format detection (PNG, JPG, etc.)
- Logs all actions to a timestamped log file

---

## 📂 File Structure
📦 indexerscore/
- 📜 generate_indexer_statements.py
- 📜 .env                        # Contains your GRAPH_API_KEY
- 📂 logs/
  - 📜 indexer_statements_log.txt
- 📂 reports/
  - 📂 images/                  # Cached avatars and banner
  - 📂 statements/              # Output PDFs per indexer
  - 📜 indexers_metrics.json    # Metrics cache
- 📜 all_indexers.json          # Indexer ID list from subgraph

---

## 🚀 How to Use

1. **Install dependencies**:
`pip install python-dotenv fpdf filetype requests`

2.	Prepare your .env file:
Create a .env file in the root directory with this line:
`GRAPH_API_KEY=your_graph_api_key_here`

3.	Run the script:
`python generate_indexer_statements.py`

The script will:
- Fetch indexers from the subgraph
- Fetch their metrics
- Generate PDFs in ./reports/statements/

## 📊 Powered By
- 🧠 [The Graph](https://thegraph.com)
- 📛 ENS (Ethereum Name Service)
- 🧩 Python, HTML5, CSS3
- 🌐 GitHub Pages / any static web host
