import os
import json
from datetime import datetime, timezone
from dotenv import load_dotenv
from dataclasses import dataclass
from typing import Optional
import requests
from fpdf import FPDF

# v0.0.1 / 30-Apr-2025
# PDF Statement Generation for Graph Indexers
# Author: Paolo Diomede
SCRIPT_VERSION="0.0.1"


# Logging function
def log_message(message):
    timestamp = datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M:%S UTC")
    log = f"[{timestamp}] {message}"
    print(log)
    with open("logs/indexer_statements_log.txt", "a") as f:
        f.write(log + "\n")


# Load environment variables
load_dotenv()
API_KEY = os.getenv("GRAPH_API_KEY")
if not API_KEY:
    log_message("❌ GRAPH_API_KEY is not set. Please check your .env file.")
    exit(1)


# Basic Indexer structure
@dataclass
class IndexerData:
    id: str
    name: Optional[str] = None
    avatar_url: Optional[str] = None
    total_stake: float = 0.0
    allocated_stake: float = 0.0
    query_fees_earned: float = 0.0
    indexing_rewards: float = 0.0
    total_rewards: float = 0.0

def load_indexers_from_file(file_path="all_indexers.json"):
    if not os.path.exists(file_path):
        log_message(f"❌ File not found: {file_path}")
        return []

    try:
        with open(file_path, "r", encoding="utf-8") as f:
            indexers = json.load(f)
            log_message(f"✅ Loaded {len(indexers)} indexer addresses.")
            return indexers
    except json.JSONDecodeError:
        log_message("❌ Failed to parse JSON file.")
        return []

def generate_indexers_json_from_subgraph(output_file="all_indexers.json"):
    query = """
    {
      indexers(first: 150, where: {stakedTokens_gt: "0"}) {
        id
      }
    }
    """
    url = f"https://gateway.thegraph.com/api/{API_KEY}/subgraphs/id/DZz4kDTdmzWLWsV373w2bSmoar3umKKH9y82SUKr5qmp"
    headers = {"Content-Type": "application/json"}
    

    try:
        response = requests.post(url, headers=headers, json={"query": query})
        response.raise_for_status()
        data = response.json()

        if "errors" in data:
            log_message("❌ GraphQL error while fetching indexers.")
            return

        indexers = [item["id"] for item in data["data"]["indexers"]]
        with open(output_file, "w", encoding="utf-8") as f:
            json.dump(indexers, f, indent=2)
        log_message(f"✅ Saved {len(indexers)} indexers to {output_file}")
    except requests.RequestException as e:
        log_message(f"❌ Request failed: {e}")

from fpdf import FPDF

def generate_indexer_statements(indexer_ids, output_dir="./reports/statements"):
    os.makedirs(output_dir, exist_ok=True)

    indexer_data_map = {}
    metrics_file = "indexers_metrics.json"
    if os.path.exists(metrics_file):
        try:
            with open(metrics_file, "r", encoding="utf-8") as f:
                indexer_data_map = json.load(f)
                log_message(f"📊 Loaded metrics from {metrics_file}")
        except json.JSONDecodeError:
            log_message(f"❌ Failed to parse {metrics_file}")

    for indexer_id in indexer_ids:
        pdf = FPDF()
        pdf.add_page()

        # Add banner image at top (replace with actual path)
        banner_path = "./reports/images/banner.jpeg"
        if os.path.exists(banner_path):
            pdf.image(banner_path, x=10, y=8, w=190)
        # Heading just below the banner
        pdf.set_font("Arial", style="B", size=16)
        pdf.set_y(75)  # Move lower to avoid overlap with banner and avatar
        pdf.cell(200, 12, txt="Indexer Statement", ln=True, align="C")
        pdf.set_font("Arial", size=12)  # Reset to normal size for body text
        pdf.ln(5)
        pdf.cell(200, 10, txt=f"Indexer ID: {indexer_id}", ln=True)

        data = indexer_data_map.get(indexer_id, {})

        # Add indexer avatar/logo if available and cache it with correct extension
        if data.get("avatar_url"):
            try:
                import urllib.request
                import filetype
                from urllib.parse import urlparse

                os.makedirs("./reports/images", exist_ok=True)
                temp_path = os.path.join("./reports/images", f"{indexer_id}_temp")

                # Download raw image first
                urllib.request.urlretrieve(data["avatar_url"], temp_path)

                # Detect actual image type
                kind = filetype.guess(temp_path)
                if not kind:
                    raise ValueError("Unable to determine image type")
                image_type = kind.extension

                final_path = os.path.join("./reports/images", f"{indexer_id}.{image_type}")

                # Rename to proper extension if needed
                if not os.path.exists(final_path):
                    os.rename(temp_path, final_path)
                    log_message(f"🖼️  Cached avatar for {indexer_id} as {image_type}")
                else:
                    os.remove(temp_path)

                if image_type in ("png", "jpeg", "jpg"):
                    pdf.image(final_path, x=165, y=10, w=30)
                else:
                    log_message(f"⚠️ Skipped avatar for {indexer_id} — unsupported format: {image_type}")

            except Exception as e:
                log_message(f"⚠️ Failed to load avatar for {indexer_id}: {e}")

        def fmt(val):
            try:
                return f"{float(val):,.0f}"
            except (ValueError, TypeError):
                return "N/A"

        pdf.cell(200, 10, txt=f"Total Stake: {fmt(data.get('total_stake'))}", ln=True)
        pdf.cell(200, 10, txt=f"Allocated Stake: {fmt(data.get('allocated_stake'))}", ln=True)
        pdf.cell(200, 10, txt=f"Query Fees Earned: {fmt(data.get('query_fees_earned'))}", ln=True)
        pdf.cell(200, 10, txt=f"Total Rewards: {fmt(data.get('total_rewards'))}", ln=True)
        pdf.cell(200, 10, txt=f"Delegator Rewards: {fmt(data.get('delegator_rewards'))}", ln=True)
        pdf.cell(200, 10, txt=f"Indexing Rewards: {fmt(data.get('indexing_rewards'))}", ln=True)

        try:
            total_rewards = float(data.get('total_rewards', 0))
            delegator_rewards = float(data.get('delegator_rewards', 0))
            if total_rewards > 0:
                delegator_pct = (delegator_rewards / total_rewards) * 100
                pdf.cell(200, 10, txt=f"Delegator Reward Share: {delegator_pct:.1f}%", ln=True)
            else:
                pdf.cell(200, 10, txt="Delegator Reward Share: N/A", ln=True)
        except Exception as e:
            log_message(f"⚠️ Failed to calculate delegator reward percentage: {e}")
            pdf.cell(200, 10, txt="Delegator Reward Share: N/A", ln=True)


        filename = os.path.join(output_dir, f"{indexer_id}.pdf")
        pdf.output(filename)
        log_message(f"📄 Generated statement for indexer: {indexer_id}")


# Generate indexer metrics JSON
def generate_indexer_metrics_json(output_file="indexers_metrics.json"):
    query = """
    {
        indexers(first: 150, where: {stakedTokens_gt: "0" }) {
            id
            allocationCount
            allocatedTokens
            stakedTokens
            delegatedTokens
            queryFeesCollected
            rewardsEarned
            indexerIndexingRewards
            delegatorIndexingRewards
            url
            account {
            defaultName
            metadata {
                image
            }
            }
        }
    }
    """
    url = f"https://gateway.thegraph.com/api/{API_KEY}/subgraphs/id/DZz4kDTdmzWLWsV373w2bSmoar3umKKH9y82SUKr5qmp"
    headers = {"Content-Type": "application/json"}

    try:
        response = requests.post(url, headers=headers, json={"query": query})
        response.raise_for_status()
        data = response.json()

        if "errors" in data:
            log_message("❌ GraphQL error while fetching metrics.")
            return

        metrics = {}
        for indexer in data["data"]["indexers"]:
            account = indexer.get("account") or {}
            metadata = account.get("metadata") or {}
            metrics[indexer["id"]] = {
                "name": account.get("defaultName"),
                "avatar_url": metadata.get("image"),
                "total_stake": float(indexer.get("stakedTokens", 0)) / 1e18,
                "allocated_stake": float(indexer.get("allocatedTokens", 0)) / 1e18,
                "query_fees_earned": float(indexer.get("queryFeesCollected", 0)) / 1e18,
                "indexing_rewards": float(indexer.get("indexerIndexingRewards", 0)) / 1e18,
                "total_rewards": float(indexer.get("rewardsEarned", 0)) / 1e18,
                "delegator_rewards": float(indexer.get("delegatorIndexingRewards", 0)) / 1e18,
            }

        with open(output_file, "w", encoding="utf-8") as f:
            json.dump(metrics, f, indent=2)
        log_message(f"✅ Saved indexer metrics to {output_file}")
    except requests.RequestException as e:
        log_message(f"❌ Request failed: {e}")


if __name__ == "__main__":
    generate_indexers_json_from_subgraph()
    generate_indexer_metrics_json()
    indexer_ids = load_indexers_from_file()
    for i, idx in enumerate(indexer_ids[:5]):
        log_message(f"Sample Indexer #{i+1}: {idx}")
    generate_indexer_statements(indexer_ids)