import os
import json
import csv
import requests
from datetime import datetime, timezone, timedelta
from dotenv import load_dotenv
from dataclasses import dataclass, asdict
from typing import Optional

# v1.2.1 / 6-May-2025
# Author: Paolo Diomede
DASHBOARD_VERSION="1.2.1"


# Function that writes in the log file
def log_message(message):
    timestamped = f"[{datetime.now(timezone.utc).strftime('%Y-%m-%d %H:%M:%S UTC')}] {message}"
    print(timestamped)
    with open(log_file, "a") as log:
        log.write(timestamped + "\n")
# End Function 'log_message'


# Load environment variables from the .env file
load_dotenv()
API_KEY = os.getenv("GRAPH_API_KEY")

# Load additional configuration variables
load_dotenv(dotenv_path=".env.costs") 
SMALL_INDEXER_THRESHOLD = float(os.getenv("SMALL_INDEXER"))
MEDIUM_INDEXER_THRESHOLD = float(os.getenv("MEDIUM_INDEXER"))
LARGE_INDEXER_THRESHOLD = float(os.getenv("LARGE_INDEXER"))
UNDERSERVING_SUBGRAPHS_THRESHOLD = float(os.getenv("UNDERSERVING_SUBGRAPHS_COUNT"))
SMALL_INDEXER_SUBGRAPH_ALLOCATION_THRESHOLD = float(os.getenv("SMALL_INDEXER_SUBGRAPH_ALLOCATION_THRESHOLD"))
MEDIUM_INDEXER_SUBGRAPH_ALLOCATION_THRESHOLD = float(os.getenv("MEDIUM_INDEXER_SUBGRAPH_ALLOCATION_THRESHOLD"))
LARGE_INDEXER_SUBGRAPH_ALLOCATION_THRESHOLD = float(os.getenv("LARGE_INDEXER_SUBGRAPH_ALLOCATION_THRESHOLD"))
MEGA_INDEXER_SUBGRAPH_ALLOCATION_THRESHOLD = float(os.getenv("MEGA_INDEXER_SUBGRAPH_ALLOCATION_THRESHOLD"))

# Load ENS cache file path
ENS_CACHE_FILE = os.getenv("ENS_CACHE_FILE", "ens_cache.json")
if not os.getenv("ENS_CACHE_FILE"):
    log_message("⚠️ ENS_CACHE_FILE not set in .env — using default: 'ens_cache.json'")

# Load ENS cache expiry duration
try:
    ENS_CACHE_EXPIRY_HOURS = int(os.getenv("ENS_CACHE_EXPIRY_HOURS", 24))
    if not os.getenv("ENS_CACHE_EXPIRY_HOURS"):
        log_message("⚠️ ENS_CACHE_EXPIRY_HOURS not set in .env — using default: 24h")
except ValueError:
    ENS_CACHE_EXPIRY_HOURS = 24
    log_message("⚠️ ENS_CACHE_EXPIRY_HOURS is not a valid integer — using default: 24h")


# If set to 1 we exclude the Upgrade Indxer from the result
EXCLUDE_UPGRADE_INDEXER = int(os.getenv("EXCLUDE_UPGRADE_INDEXER"))

# List of all used subgraphs
SUBGRAPH_URL = f"https://gateway.thegraph.com/api/{API_KEY}/subgraphs/id/DZz4kDTdmzWLWsV373w2bSmoar3umKKH9y82SUKr5qmp"
ENS_SUBGRAPH_URL = f"https://gateway.thegraph.com/api/{API_KEY}/subgraphs/id/5XqPmWe6gjyrJtFn9cLy237i4cWw2j9HcUJEXsP5qGtH"

# Global counters for indexer stats
TOTAL_INDEXERS_COUNT = 0
ACTIVE_INDEXERS_COUNT = 0
ALLOCATED_INDEXERS_COUNT = 0
SMALL_INDEXERS_ACTIVE_COUNT = 0
MEDIUM_INDEXERS_ACTIVE_COUNT = 0
LARGE_INDEXERS_ACTIVE_COUNT = 0
MEGA_INDEXERS_ACTIVE_COUNT = 0
INDEXERS_UNDERSERVING_COUNT = 0
EXCELLENT_INDEXERS_COUNT = 0
FAIR_INDEXERS_COUNT = 0
POOR_INDEXERS_COUNT = 0



# === Helper Functions ===
def format_grt(grt):
    if grt >= 1_000_000:
        return f"{grt / 1_000_000:.0f}M"
    elif grt >= 1_000:
        return f"{grt / 1_000:.0f}K"
    return str(grt)


# Indexer Dataclass to store the indexers data from the Graph Network
@dataclass
class IndexerData:
    id: str
    name: Optional[str] = None
    avatar_url: Optional[str] = None

    # Stake metrics
    total_stake: float = 0.0
    own_stake: float = 0.0
    delegated_stake: float = 0.0
    allocated_stake: float = 0.0
    available_stake: float = 0.0  # total - allocated

    # Activity
    number_of_allocations: int = 0
    number_of_subgraphs: int = 0

    # Rewards
    query_fees_earned: float = 0.0
    indexing_rewards: float = 0.0
    total_rewards: float = 0.0

    # Behavioral flags
    is_active: bool = True
    is_underserving: bool = False  # Allocates little vs. capacity
    is_suspicious: bool = False    # Custom flag based on logic

    indexer_size: Optional[str] = None
    aer: Optional[float] = None
    qfr: Optional[float] = None

    aer_normalized: Optional[float] = None
    qfr_normalized: Optional[float] = None
    final_score: Optional[float] = None
    performance_flag: Optional[str] = None  # New field for the flag
    penalty: Optional[float] = 0.0  # New field for the penalty

    delegator_rewards: Optional[float] = None
    delegator_rewards_percentage:  Optional[float] = None

    # Computed properties
    def allocation_ratio(self) -> float:
        return self.allocated_stake / self.total_stake if self.total_stake else 0.0

    def calculate_aer(self) -> float:
        if self.indexer_size == "small":
            avg_grt_per_allocation = SMALL_INDEXER_SUBGRAPH_ALLOCATION_THRESHOLD
        elif self.indexer_size == "medium":
            avg_grt_per_allocation = MEDIUM_INDEXER_SUBGRAPH_ALLOCATION_THRESHOLD
        elif self.indexer_size == "large":
            avg_grt_per_allocation = LARGE_INDEXER_SUBGRAPH_ALLOCATION_THRESHOLD
        elif self.indexer_size == "mega":
            avg_grt_per_allocation = MEGA_INDEXER_SUBGRAPH_ALLOCATION_THRESHOLD
        else:
            return 0.0  # fallback

        if self.number_of_allocations == 0:
            return 0.0  # avoid division by zero

        expected_allocation = self.number_of_allocations * avg_grt_per_allocation
        return self.allocated_stake / expected_allocation if expected_allocation else 0.0

    def calculate_qfr(self) -> float:
        if self.allocated_stake == 0:  # Avoid division by zero
            return 0.0
        return self.query_fees_earned / self.allocated_stake
    
    # Calculate the GRT given to delegators
    def calculate_delegator_rewards(self) -> float:
        return self.total_rewards - self.indexing_rewards

    # Calculate the percentage of total rewards given to delegators
    def calculate_delegator_rewards_percentage(self) -> float:
        if self.total_rewards == 0:  # Avoid division by zero
            return 0.0
        return (self.delegator_rewards / self.total_rewards) * 100   
    
    def to_dict(self):
        return asdict(self)


# Create REPORTS directory if it doesn't exist
report_dir = "reports"
os.makedirs(report_dir, exist_ok=True)

# Create LOGS directory if it doesn't exist
log_dir = "logs"
os.makedirs(log_dir, exist_ok=True)
log_file = os.path.join(log_dir, f"metrics_log_{datetime.now(timezone.utc).strftime('%Y-%m-%d')}.txt")

# Create METRICS directory if it doesn't exist
metrics_dir = "metrics"
os.makedirs(metrics_dir, exist_ok=True)


# Get data to be used in the log and report files
timestamp = datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M:%S UTC")
date_suffix = datetime.now(timezone.utc).strftime("%m%d%Y")


# GraphQL query to retrieve main stats and all the active Indexers
queryAllIndexers = """
{
  graphNetwork(id: "1") {
    indexerCount
    stakedIndexersCount
    subgraphCount
    activeSubgraphCount
  }
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

queryIndexer_excluding_UpgradeIndexer = """
{
  graphNetwork(id: "1") {
    indexerCount
    stakedIndexersCount
    subgraphCount
    activeSubgraphCount
  }
indexers(first: 150, where: {stakedTokens_gt: "0", id_not: "0xbdfb5ee5a2abf4fc7bb1bd1221067aef7f9de491"}) {
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

# GraphQL query to retrieve ENS
queryENS = """
    query GetEnsName($address: String!) {
      domains(where: { resolvedAddress: $address, name_ends_with: ".eth" }, first: 1) {
        name
      }
    }
"""


# FUNCTION SECTION

# Updated fetch_metrics function
def fetch_metrics():
    global TOTAL_INDEXERS_COUNT, ACTIVE_INDEXERS_COUNT, ALLOCATED_INDEXERS_COUNT, SMALL_INDEXERS_ACTIVE_COUNT, MEDIUM_INDEXERS_ACTIVE_COUNT, LARGE_INDEXERS_ACTIVE_COUNT, MEGA_INDEXERS_ACTIVE_COUNT, INDEXERS_UNDERSERVING_COUNT, EXCLUDE_UPGRADE_INDEXER, EXCELLENT_INDEXERS_COUNT, FAIR_INDEXERS_COUNT, POOR_INDEXERS_COUNT
    
    log_message("⚙️  Fetching data from subgraph...")
    
    if EXCLUDE_UPGRADE_INDEXER == 1:
        log_message("(Excluding the Upgrade Indexer)")

    headers = {
        "Content-Type": "application/json"
    }

    if EXCLUDE_UPGRADE_INDEXER == 1:
        response = requests.post(SUBGRAPH_URL, headers=headers, json={"query": queryIndexer_excluding_UpgradeIndexer})
    else:
        response = requests.post(SUBGRAPH_URL, headers=headers, json={"query": queryAllIndexers})
    
    result = response.json()

    if "errors" in result:
        log_message("❌ GraphQL error: " + json.dumps(result["errors"], indent=2))
        return

    data = result["data"]["graphNetwork"]
    data["timestamp"] = timestamp

    # Set TOTAL_INDEXERS_COUNT from the GraphQL response
    TOTAL_INDEXERS_COUNT = int(data.get("indexerCount", 0))

    if EXCLUDE_UPGRADE_INDEXER == 1:
        TOTAL_INDEXERS_COUNT -= 1  # Adjust for the excluded Upgrade Indexer
        log_message(f"✔️ Adjusted total indexer count to exclude Upgrade Indexer: {TOTAL_INDEXERS_COUNT}")

    indexers_data = []
    
    for indexer_raw in result["data"]["indexers"]:
        
        allocated_tokens = float(indexer_raw.get("allocatedTokens", 0)) / 1e18
        allocation_count = int(indexer_raw.get("allocationCount", 0))
        
        if allocated_tokens <= 0:
            continue
        
        ens_name = fetch_ens_name2(indexer_raw["id"]) or indexer_raw["id"]
        account = indexer_raw.get("account") or {}
        metadata = account.get("metadata") or {}
        avatar_url = metadata.get("image")
    
        own_stake = float(indexer_raw.get("stakedTokens", 0)) / 1e18
        delegated_stake = float(indexer_raw.get("delegatedTokens", 0)) / 1e18
        total_stake = own_stake + delegated_stake
        is_underserving = allocation_count <= UNDERSERVING_SUBGRAPHS_THRESHOLD
        
        if is_underserving:
            INDEXERS_UNDERSERVING_COUNT += 1
                
        if total_stake < SMALL_INDEXER_THRESHOLD:
            size = "small"
        elif SMALL_INDEXER_THRESHOLD <= total_stake < MEDIUM_INDEXER_THRESHOLD:
            size = "medium"
        elif MEDIUM_INDEXER_THRESHOLD <= total_stake < LARGE_INDEXER_THRESHOLD:
            size = "large"
        else:
            size = "mega"

        indexer = IndexerData(
            id=indexer_raw["id"],
            name=ens_name,
            avatar_url=avatar_url,
            own_stake=own_stake,
            delegated_stake=delegated_stake,
            total_stake=total_stake,
            allocated_stake=allocated_tokens,
            available_stake=total_stake - allocated_tokens,
            number_of_allocations=allocation_count,
            number_of_subgraphs=allocation_count,
            query_fees_earned=float(indexer_raw.get("queryFeesCollected", 0)) / 1e18,
            indexing_rewards=float(indexer_raw.get("indexerIndexingRewards", 0)) / 1e18,
            total_rewards=float(indexer_raw.get("rewardsEarned", 0)) / 1e18,
            is_active=True,
            is_underserving=is_underserving,
            is_suspicious=is_underserving,
            indexer_size=size
        )
        indexer.aer = indexer.calculate_aer()
        indexer.qfr = indexer.calculate_qfr()
        indexer.delegator_rewards = indexer.calculate_delegator_rewards()
        indexer.delegator_rewards_percentage = indexer.calculate_delegator_rewards_percentage()
        indexers_data.append(indexer)
  
    ACTIVE_INDEXERS_COUNT = int(data.get("stakedIndexersCount", 0))

    if EXCLUDE_UPGRADE_INDEXER == 1:
        ACTIVE_INDEXERS_COUNT -= 1
    
    ALLOCATED_INDEXERS_COUNT = len(indexers_data)

    SMALL_INDEXERS_ACTIVE_COUNT = len([i for i in indexers_data if i.is_active and i.allocated_stake < SMALL_INDEXER_THRESHOLD])
    MEDIUM_INDEXERS_ACTIVE_COUNT = len([i for i in indexers_data if i.is_active and SMALL_INDEXER_THRESHOLD <= i.allocated_stake < MEDIUM_INDEXER_THRESHOLD])
    LARGE_INDEXERS_ACTIVE_COUNT = len([i for i in indexers_data if i.is_active and i.allocated_stake >= MEDIUM_INDEXER_THRESHOLD and i.allocated_stake < LARGE_INDEXER_THRESHOLD])
    MEGA_INDEXERS_ACTIVE_COUNT = len([i for i in indexers_data if i.is_active and i.allocated_stake >= LARGE_INDEXER_THRESHOLD])

    log_message(f"Fetched {len(indexers_data)} indexers")
    log_message("📊 Indexer Stats:")
    log_message(f"Total Indexers: {TOTAL_INDEXERS_COUNT}")
    log_message(f"Active Indexers with staked GRT: {ACTIVE_INDEXERS_COUNT}")
    log_message(f"Allocated Indexers: {ALLOCATED_INDEXERS_COUNT}")
    log_message(f"Small Active Indexers: {SMALL_INDEXERS_ACTIVE_COUNT} with less than {SMALL_INDEXER_THRESHOLD} GRT staked")
    log_message(f"Medium Active Indexers: {MEDIUM_INDEXERS_ACTIVE_COUNT} with more than {SMALL_INDEXER_THRESHOLD} and less than {MEDIUM_INDEXER_THRESHOLD} GRT staked")
    log_message(f"Large Active Indexers: {LARGE_INDEXERS_ACTIVE_COUNT} with more than {MEDIUM_INDEXER_THRESHOLD} and less than {LARGE_INDEXER_THRESHOLD} GRT staked")
    log_message(f"Mega Active Indexers: {MEGA_INDEXERS_ACTIVE_COUNT} with more than {LARGE_INDEXER_THRESHOLD} GRT staked")

    # Normalize AER using the capped method (1 = best, 10 = worst)
    aer_values = [i.aer for i in indexers_data]
    normalized_aer = normalize_aer_capped(aer_values, cap=500.0)
    for i, score in zip(indexers_data, normalized_aer):
        i.aer_normalized = score if score is not None else 1.0

    # Normalize QFR using the inverted method (10 = best, 1 = worst)
    qfr_values = [i.qfr for i in indexers_data]
    normalized_qfr = normalize_qfr_inverted(qfr_values, cap=1.0)
    for i, score in zip(indexers_data, normalized_qfr):
        i.qfr_normalized = score if score is not None else 1.0

    # Calculate final score, apply penalty, and assign performance flag
    for i in indexers_data:
        # Ensure normalized values are not None
        aer_norm = i.aer_normalized if i.aer_normalized is not None else 1.0
        qfr_norm = i.qfr_normalized if i.qfr_normalized is not None else 1.0
        
        # Calculate initial final score
        qfr_adjusted = 11 - qfr_norm
        i.final_score = round((aer_norm * 0.7) + (qfr_adjusted * 0.3), 2)

        # Apply underserving penalty if applicable and store the penalty value
        if i.number_of_subgraphs < UNDERSERVING_SUBGRAPHS_THRESHOLD:
            i.penalty = 2.0 * (UNDERSERVING_SUBGRAPHS_THRESHOLD - i.number_of_subgraphs) / UNDERSERVING_SUBGRAPHS_THRESHOLD
            i.final_score = min(10.0, i.final_score + i.penalty)
            i.final_score = round(i.final_score, 2)  # Round again after applying penalty
            # log_message(f"Indexer {i.id}: Applied penalty of {i.penalty:.2f} for serving {i.number_of_subgraphs} subgraphs (threshold: {UNDERSERVING_SUBGRAPHS_THRESHOLD}). New final_score={i.final_score}")
        else:
            i.penalty = 0.0  # Explicitly set penalty to 0 for non-underserving Indexers

        # Assign performance flag based on final score (after penalty)
        if i.final_score is not None:
            if 1.0 <= i.final_score <= 1.25:
                i.performance_flag = "Excellent 🟢"
            elif 1.26 <= i.final_score <= 2.5:
                i.performance_flag = "Fair 🟡"
            else:  # 2.51 to 10.0
                i.performance_flag = "Poor 🔴"
            # Debug: Log the assignment
            # log_message(f"Indexer {i.id}: final_score={i.final_score}, performance_flag={i.performance_flag}")
        else:
            i.performance_flag = "Unknown ⚪"
            log_message(f"Indexer {i.id}: final_score is None, performance_flag={i.performance_flag}")
    
    # Count performance flags
    EXCELLENT_INDEXERS_COUNT = len([i for i in indexers_data if i.performance_flag and i.performance_flag.startswith("Excellent")])
    FAIR_INDEXERS_COUNT = len([i for i in indexers_data if i.performance_flag and i.performance_flag.startswith("Fair")])
    POOR_INDEXERS_COUNT = len([i for i in indexers_data if i.performance_flag and i.performance_flag.startswith("Poor")])
    
    log_message(f"Performance Categories — Excellent: {EXCELLENT_INDEXERS_COUNT}, Fair: {FAIR_INDEXERS_COUNT}, Poor: {POOR_INDEXERS_COUNT}")
    
    # Save snapshot metric to metrics_dir
    metric_data = {
        "timestamp": timestamp,
        "total_subgraphs": int(data.get("subgraphCount", 0)),
        "active_subgraphs": int(data.get("activeSubgraphCount", 0)),
        "total_indexers": TOTAL_INDEXERS_COUNT,
        "active_indexers": ACTIVE_INDEXERS_COUNT,
        "allocated_indexers": ALLOCATED_INDEXERS_COUNT,
        "excellent_count": EXCELLENT_INDEXERS_COUNT,
        "fair_count": FAIR_INDEXERS_COUNT,
        "poor_count": POOR_INDEXERS_COUNT
    }

    metric_filename = f"metric_{datetime.now(timezone.utc).strftime('%Y%m%d_%H%M%S')}.json"
    metric_path = os.path.join(metrics_dir, metric_filename)
    with open(metric_path, "w") as metric_file:
        json.dump(metric_data, metric_file, indent=2)
    log_message(f"📁 Saved metric snapshot to {metric_path}")

    return indexers_data
# End Function 'fetch_metrics'


# Function that fetches ENS names
def fetch_ens_name(address):
    headers = {
        "Content-Type": "application/json"
    }

    response = requests.post(ENS_SUBGRAPH_URL, headers=headers, json={"query": queryENS, "variables": {"address": address}})
       
    if response.status_code == 200:
        result = response.json()
        domains = result.get("data", {}).get("domains", [])
        if domains and "name" in domains[0]:
            return domains[0]["name"]
    return ""
# End Function 'fetch_ens_name'


def fetch_ens_name2(address: str) -> str:
    global ENS_SUBGRAPH_URL
    headers = {"Content-Type": "application/json"}
    address = address.lower()

    # Load cache from file
    if os.path.exists(ENS_CACHE_FILE):
        with open(ENS_CACHE_FILE, "r") as f:
            ens_cache = json.load(f)
    else:
        ens_cache = {}

    # Check cache and freshness (24h)
    record = ens_cache.get(address)
    if record:
        last_updated = datetime.fromisoformat(record["timestamp"]).replace(tzinfo=timezone.utc)
        if datetime.now(timezone.utc) - last_updated < timedelta(hours=ENS_CACHE_EXPIRY_HOURS):
            log_message(f"🧠 Using cached ENS for {address}: {record['ens'] or 'no ENS'}")
            return record["ens"]

    # Build GraphQL query
    payload = {
        "query": queryENS,
        "variables": { "address": address }
    }

    try:
        response = requests.post(ENS_SUBGRAPH_URL, headers=headers, json=payload)
        response.raise_for_status()
        result = response.json()
        domains = result.get("data", {}).get("domains", [])

        # Match the structure used in fetch_ens_name
        if domains and "name" in domains[0]:
            ens_name = domains[0]["name"]
        else:
            ens_name = ""
        ens_cache[address] = {
            "ens": ens_name,
            "timestamp": datetime.now(timezone.utc).isoformat()
        }

        # Save updated cache
        with open(ENS_CACHE_FILE, "w") as f:
            json.dump(ens_cache, f, indent=2)

        log_message(f"🔍 Fetched ENS for {address}: {ens_name or 'no ENS'}")
        return ens_name

    except requests.RequestException as e:
        log_message(f"⚠️ ENS lookup failed for {address}: {e}")
        return ""
    

# Normalize AER values on a scale from 1 to 10 with a cap at 500.
# 1 is best (lowest AER), 10 is worst (highest AER).
# Returns a list of normalized AER values between 1 and 10
def normalize_aer_capped(values, cap=500.0):
    normalized = []
    for v in values:
        capped_value = min(v, cap)
        norm_value = 1 + 9 * (capped_value / cap)
        norm_value = max(1.0, min(10.0, norm_value))
        normalized.append(round(norm_value, 2))
    return normalized
# End Function 'normalize_aer_capped'


# Normalize QFR values on a scale from 10 to 1 with a cap at 1.0.
# 10 is best (highest QFR), 1 is worst (lowest QFR).
# Returns a list of normalized QFR values between 1 and 10
def normalize_qfr_inverted(values, cap=1.0):
    normalized = []
    for v in values:
        capped_value = min(v, cap)
        norm_value = 10 - 9 * (capped_value / cap)
        norm_value = max(1.0, min(10.0, norm_value))
        normalized.append(round(norm_value, 2))
    return normalized
# End Function 'normalize_qfr_inverted'


# Function that writes the CSV report file
def generate_indexers_to_csv(indexers, filename="indexers_output.csv"):
    csv_path = os.path.join(report_dir, filename)
    with open(csv_path, "w", newline="", encoding="utf-8") as csvfile:
        fieldnames = list(indexers[0].to_dict().keys())
        if "delegator_rewards_percentage" not in fieldnames:
            fieldnames.append("delegator_rewards_percentage")

        writer = csv.DictWriter(csvfile, fieldnames=fieldnames, delimiter=';')
        writer.writeheader()
        for indexer in indexers:
            # Create a copy of the indexer's dictionary
            indexer_dict = indexer.to_dict()
            # Remove the emoji from performance_flag for CSV output
            if indexer_dict["performance_flag"]:
                # Split on the space and take the first part (e.g., "Excellent 🟢" -> "Excellent")
                indexer_dict["performance_flag"] = indexer_dict["performance_flag"].split(" ")[0]
            # Round the penalty to 2 decimal places for CSV output
            indexer_dict["penalty"] = round(indexer_dict["penalty"], 2)
            # Ensure delegator_rewards_percentage is present and formatted
            indexer_dict["delegator_rewards_percentage"] = f"{indexer_dict.get('delegator_rewards_percentage', 0.0):.2f}%"
            writer.writerow(indexer_dict)
    log_message(f"✅ Saved CSV file: {csv_path}")
# End Function 'generate_indexers_to_csv'


# Function that writes the HTML report file
def generate_indexers_to_html(indexers, filename="index.html"):
    global TOTAL_INDEXERS_COUNT, ACTIVE_INDEXERS_COUNT, ALLOCATED_INDEXERS_COUNT, SMALL_INDEXERS_ACTIVE_COUNT, MEDIUM_INDEXERS_ACTIVE_COUNT, LARGE_INDEXERS_ACTIVE_COUNT, MEGA_INDEXERS_ACTIVE_COUNT, INDEXERS_UNDERSERVING_COUNT, EXCLUDE_UPGRADE_INDEXER, EXCELLENT_INDEXERS_COUNT, FAIR_INDEXERS_COUNT, POOR_INDEXERS_COUNT
    
    small_thresh_fmt = format_grt(SMALL_INDEXER_THRESHOLD)
    medium_thresh_fmt = format_grt(MEDIUM_INDEXER_THRESHOLD)
    large_thresh_fmt = format_grt(LARGE_INDEXER_THRESHOLD)
    
    html_file = os.path.join(report_dir, filename)
    heading_note = " (excluding the Upgrade Indexer)" if EXCLUDE_UPGRADE_INDEXER == 1 else ""

    html = f"""
    <html>
    <head>
        <meta charset='UTF-8' />
        <meta name="viewport" content="width=device-width, initial-scale=1.0">
        
        <title>Indexer Score Dashboard - The Graph Network</title>
        <meta name="description" content="Explore The Graph Indexer Score Dashboard to analyze indexer performance, allocations, and rewards.">
        <meta name="keywords" content="The Graph, Indexer Score, blockchain indexing, GRT, subgraph allocations, query fees, indexer performance">
        <meta name="robots" content="index, follow">

        <link rel="canonical" href="https://indexerscore.com/index.html">

        <meta property="og:title" content="Indexer Score Dashboard - The Graph Network">
        <meta property="og:description" content="Explore The Graph Indexer Score Dashboard to analyze indexer performance, allocations, and rewards.">
        <meta property="og:image" content="https://graphtools.pro/IndexerScore.jpg">
        <meta property="og:url" content="https://indexerscore.com/index.html">
        <meta property="og:type" content="website">

        <meta name="twitter:card" content="summary_large_image">
        <meta name="twitter:title" content="Indexer Score Dashboard - The Graph Network">
        <meta name="twitter:description" content="Analyze indexer performance and rewards on The Graph Network with our dashboard.">
        <meta name="twitter:image" content="https://graphtools.pro/IndexerScore.jpg">
        <meta name="twitter:site" content="@graphtronauts_c">

        <link rel="icon" type="image/png" href="https://graphtools.pro/favicon.ico">

        <style>
            :root {{
                --bg-color: #111;
                --text-color: #fff;
                --table-bg: #1e1e1e;
                --header-bg: #333;
                --link-color: #fff;
            }}
            
            .light-mode {{
                --bg-color: #f0f2f5;
                --text-color: #000;
                --table-bg: #ffffff;
                --header-bg: #ddd;
                --link-color: #000;
            }}
            
            body {{
                font-family: Arial, sans-serif;
                padding: 10px 20px 20px 20px;
                margin-top: 0;
                background-color: var(--bg-color);
                color: var(--text-color);
                transition: all 0.3s ease;
            }}

            .header-container {{
                display: flex;
                justify-content: space-between;
                align-items: center;
                margin-bottom: 15px;
                line-height: 1;
            }}
            
            .breadcrumb {{
                font-size: 0.9em;
                margin: 0;
                padding: 0;
                display: flex;
                align-items: center;
            }}

            .toggle-container {{
                display: flex;
                align-items: center;
                margin: 0;
                padding: 0;
            }}

            .toggle-switch {{
                position: relative;
                width: 50px;
                height: 24px;
                margin-right: 10px;
            }}

            .toggle-switch input {{
                opacity: 0;
                width: 0;
                height: 0;
            }}

            .toggle-switch .slider {{
                position: absolute;
                top: 0; left: 0;
                right: 0; bottom: 0;
                background: #ccc;
                transition: 0.4s;
                border-radius: 34px;
            }}

            .toggle-switch .slider:before {{
                position: absolute;
                content: "";
                height: 18px;
                width: 18px;
                left: 4px;
                bottom: 3px;
                background: white;
                transition: 0.4s;
                border-radius: 50%;
            }}

            .toggle-switch input:checked + .slider {{
                background: #2196F3;
            }}

            .toggle-switch input:checked + .slider:before {{
                transform: translateX(24px);
            }}

            #toggle-icon {{
                font-size: 1.5rem;
                line-height: 1;
            }}

            .divider {{
                border: 0;
                height: 2px;
                background: linear-gradient(to right, rgba(255, 255, 255, 0), rgba(255, 255, 255, 0.5), rgba(255, 255, 255, 0));
                margin: 15px 0;
            }}

            .light-mode .divider {{
                background: linear-gradient(to right, rgba(0, 0, 0, 0), rgba(0, 0, 0, 0.7), rgba(0, 0, 0, 0));
            }}
    
            .tooltip-wrapper {{
                position: relative;
                display: inline-block;
                cursor: help;
                border-bottom: 1px dotted #999;
            }}
            .tooltip-wrapper::after {{
                font-size: 50%;
            }}

            .tooltip-text {{
                visibility: hidden;
                background-color: #333;
                color: #fff;
                text-align: left;
                padding: 6px 10px;
                border-radius: 4px;
                position: absolute;
                z-index: 1000;
                top: -5px;
                left: 50%;
                transform: translateX(-50%);
                opacity: 0;
                transition: opacity 0.3s;
                width: max-content;
                max-width: 220px;
                font-size: 13px;
                pointer-events: none;
                white-space: normal;
            }}

            .tooltip-wrapper:hover .tooltip-text {{
                visibility: visible;
                opacity: 1;
                display: block;
                position: absolute;
           }}

            .highlight-delegator {{
                color: #0066ff !important;
            }}
            .dark-mode .highlight-delegator {{
                color: #66b3ff !important;
            }}

            .box {{
                padding: 10px;
                margin-top: 20px;
                border-left: 5px solid;
                width: 50%;
            }}

            .thresholds-box {{
                background-color: #e1f5fe;
                border-color: #2196f3;
            }}

            .network-box {{
                background-color: #e8f5e9;
                border-color: #4caf50;
            }}

            .dark-mode .thresholds-box {{
                background-color: #1e2a38;
                color: #eee;
            }}

            .dark-mode .network-box {{
                background-color: #1e3022;
                color: #eee;
            }}

            a {{
                color: var(--link-color);
                text-decoration: none;
            }}
            a:hover {{
                text-decoration: underline;
            }}

            table {{
                border-collapse: collapse;
                width: 100%;
                border-radius: 12px;
                overflow: hidden;
                border: 1px solid #ccc;
            }}

            th, td {{
                border: 1px solid #ccc;
                padding: 8px;
                text-align: left;
            }}

            th {{
                background-color: #f2f2f2;
                cursor: pointer;
                font-size: 0.85em;
                /* Remove underlining effect */
                text-decoration: none;
                border-bottom: none;
            }}

            th span {{
                text-decoration: none !important;
                border-bottom: none !important;
            }}

            tr:nth-child(even) {{
                background-color: #fafafa;
            }}

            .dark-mode table {{
                border-color: #555;
            }}

            .dark-mode th {{
                background-color: #444;
                color: #fff;
            }}

            .dark-mode tr:nth-child(even) {{
                background-color: #2c2c2c;
            }}

            .light-mode table {{
                background-color: #ffffff;
                color: #000;
            }}

            .light-mode th, .light-mode td {{
                background-color: #ffffff;
                color: #000;
            }}

            .switch {{
                position: relative;
                display: inline-block;
                width: 50px;
                height: 24px;
            }}

            .switch input {{
                opacity: 0;
                width: 0;
                height: 0;
            }}

            .slider {{
                position: absolute;
                cursor: pointer;
                top: 0;
                left: 0;
                right: 0;
                bottom: 0;
                background-color: #ccc;
                transition: .4s;
                border-radius: 24px;
            }}

            .slider:before {{
                position: absolute;
                content: "";
                height: 18px;
                width: 18px;
                left: 3px;
                bottom: 3px;
                background-color: white;
                transition: .4s;
                border-radius: 50%;
            }}

            input:checked + .slider {{
                background-color: #2196f3;
            }}

            input:checked + .slider:before {{
                transform: translateX(26px);
            }}

            th.sorted-column {{
                background-color: #d1ecf1 !important;
                color: #000 !important;
            }}

            th.sorted-column::after {{
                content: attr(data-sort-direction);
                margin-left: 6px;
                font-size: 12px;
            }}

            .sorted-column {{
                background-color: #d1ecf1 !important;
            }}

            .version {{ 
                font-size: 12px; 
            }}

            .download-button {{ 
                padding: 5px 10px; background-color: #4CAF50; color: white; border: none; border-radius: 3px; cursor: pointer;
            }}

            .download-button:hover {{ 
                background-color: #45a049; 
            }}

            .performance-filter a {{
                text-decoration: none;
            }}

            .footer {{
                    text-align: center;
                    margin: 10px 0 40px;
                    font-size: 0.9rem;
                    opacity: 0.9;
            }}

            .footer a {{
                color: #80bfff;
                text-decoration: none;
                transition: color 0.3s ease;
            }}

            .footer a:hover {{
                color: #4d94ff;
            }}

            .light-mode .footer a {{
                color: #0066cc;
            }}

            .light-mode .footer a:hover {{
            color: #0033ff;
            }}

            .footer-divider {{
                border: none;
                border-bottom: 1px solid rgba(200, 200, 200, 0.2);
                margin: 40px 0 10px;
                opacity: 0.8;
            }}

            .current-page-title {{
                color: #00bcd4;
                font-weight: bold;
            }}

            .light-mode .current-page-title {{
                color: #1a73e8;
            }}
            
            .filter-button.active-filter {{
                background-color: #ffeb3b;
                padding: 2px 6px;
                border-radius: 4px;
                color: black;
                font-weight: bold;
            }}

        </style>

        <!-- Plausible Analytics -->
        <script defer data-domain="indexerscore.com" src="https://plausible.io/js/script.file-downloads.hash.outbound-links.pageview-props.tagged-events.js"></script>
        <script>
            window.plausible = window.plausible || function() {{
                (window.plausible.q = window.plausible.q || []).push(arguments)
            }}
        </script>

    </head>
    
    <body>
        <!-- Header with breadcrumb and toggle -->

        <div class="header-container">
            <div class="breadcrumb" style="font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif; font-weight: 500; font-size: 0.85em; letter-spacing: 0.3px; text-shadow: 0 1px 2px rgba(0,0,0,0.15);">
                <a href="https://graphtools.pro" class="home-link" style="text-decoration: none;">🏠 Home</a>&nbsp;&nbsp;&raquo;&nbsp;&nbsp;
                <span class="current-page-title">📊 Indexer Score Dashboard</span>    
            </div>

            <div class="toggle-container">
                <label class="toggle-switch">
                  <input type="checkbox">
                  <span class="slider"></span>
                </label>
                <span id="toggle-icon">🌙</span>
            </div>
        </div>

        
        <hr class="divider">
        <div style="text-align: center;">       
            <h1 style="margin-bottom: 4px;">Indexer Score Dashboard</h1>
            <div style="text-align: center; font-size: 0.8em; color: var(--text-color); margin-top: 0; margin-bottom: 30px;">
                Generated on: {timestamp} - (updated every 6 hours) - v{DASHBOARD_VERSION}
                <p style="margin-top: 10px;">
                    💬 <a target="_blank" href="https://forum.thegraph.com/t/introducing-the-indexer-score/6501">Join the Forum Discussion</a>
                    | 📚 <a href="docs.html" target="_blank">How the Dashboard Works</a>
                    | 📄 <a  target="_blank" href="./indexer_score_documentation_v1.1.0.pdf" download>Download PDF Documentation</a>
                </p>
            </div>
        </div>

 
        <div style="display: flex; gap: 20px; flex-wrap: wrap;">
            <!-- Network Stats Box -->
            <div style="padding: 6px 12px; background-color: #d6cbd3; border-left: 5px solid #4caf50; width: fit-content; color: black !important;">
                <ul style="margin: 0; padding-left: 20px;">
                    <li><strong>Active Indexers:</strong> {ACTIVE_INDEXERS_COUNT}  <span style="font-size: 0.8em;">{heading_note}</span></li>
                    <li><strong>Indexers with allocations:</strong> {ALLOCATED_INDEXERS_COUNT}</li>
                    <li><strong>Indexers serving <span style="color: #c83349;">less than</span> {int(UNDERSERVING_SUBGRAPHS_THRESHOLD)} allocations:</strong> {INDEXERS_UNDERSERVING_COUNT} <span style="font-size: 0.8em;">→ check the 'Underserving' column</span></li>
                </ul>
            </div>
        </div>

        <br /> 
        <br />

        <div style="display: flex; align-items: center; gap: 12px; flex-wrap: wrap; margin-bottom: 10px;">
            <input type="text" id="searchBox" placeholder="Search indexers..." style="padding: 5px; width: 300px;" />
            
            <button class="download-button" onclick="downloadCSV()">Download CSV</button>

            <div class="performance-filter" style="display: flex; align-items: center; gap: 8px; font-size: 0.8em;">
                &nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;<strong style="margin-right: 6px;">Filter by Performance: </strong>
                <a href="#" class="filter-button" onclick="filterByFlag('Excellent')">🟢 Excellent ({EXCELLENT_INDEXERS_COUNT})</a>
                | <a href="#" class="filter-button" onclick="filterByFlag('Fair')">🟡 Fair ({FAIR_INDEXERS_COUNT})</a>
                | <a href="#" class="filter-button" onclick="filterByFlag('Poor')">🔴 Poor ({POOR_INDEXERS_COUNT})</a>
                | <a href="#" class="filter-button" onclick="filterByFlag('All')">🧹 Clear Filter</a>
            </div>

        </div>
        
        <div style="overflow-x: auto; border-radius: 12px; border: 1px solid #ccc;">
        <table>
            <tr>
                <th>
                  <span class="tooltip-wrapper">ID<span class="tooltip-text">ETH address or ENS of the Indexer</span></span>
                </th>
                <th>
                  <span class="tooltip-wrapper">Allocated Stake<span class="tooltip-text">GRT currently allocated by the Indexer</span></span>
                </th>
                <th>
                  <span class="tooltip-wrapper"># Allocations<span class="tooltip-text">Number of active allocations the Indexer has open</span></span>
                </th>
                <th>
                  <span class="tooltip-wrapper highlight-delegator">Delegator Reward %<span class="tooltip-text">Percentage of total rewards given to Delegators</span></span>
                </th>
                <th>
                  <span class="tooltip-wrapper">Delegator Rewards<span class="tooltip-text">Delegators portion of indexing rewards</span></span>
                </th>
                <th>
                  <span class="tooltip-wrapper">Indexing Rewards<span class="tooltip-text">Indexer portion of indexing rewards</span></span>
                </th>
                <th>
                  <span class="tooltip-wrapper">Total Rewards<span class="tooltip-text">Total rewards earned (Indexer + delegators)</span></span>
                </th>
                <th>
                  <span class="tooltip-wrapper">Underserving<span class="tooltip-text">Yes if Indexer is serving fewer than 10 subgraphs</span></span>
                </th>
                <th>
                  <span class="tooltip-wrapper">Indexer Score<span class="tooltip-text">Final Indexer Score (1 = best, 10 = worst)</span></span>
                </th>
                <th data-sort-direction="asc" class="sorted-column">
                  <span class="tooltip-wrapper">Performance<span class="tooltip-text">Performance flag based on score</span></span>
                </th>
            </tr>
    """

    for i in indexers:
        html += f"""
        <tr>
            
            <td>
                {f"<img src='{i.avatar_url}' style='width:20px; height:20px; border-radius:50%; vertical-align:middle; margin-right:6px;' />" if i.avatar_url else ""}
                <a target="_blank" href="https://thegraph.com/explorer/profile/{i.id}?view=Overview&chain=arbitrum-one"><span style="font-size: 0.85em;">{i.name}</span></a>
            <!--
                <a href="statements/{i.id}.pdf" target="_blank">
                    <img src="images/pdf.png" alt="PDF" style="width:14px; height:14px; vertical-align:middle; margin-left:6px;" />
                </a>
            -->
            </td>

            <td data-value="{i.allocated_stake}"><span style="font-size: 0.85em;">{i.allocated_stake:,.0f}</span></td>
            <td data-value="{i.number_of_allocations}"><span style="font-size: 0.85em;">{i.number_of_allocations}</span></td>
            <td data-value="{i.delegator_rewards_percentage:.2f}"><span style="font-size: 0.85em; color: #0066ff;">{i.delegator_rewards_percentage:.2f}%</span></td>
            <td data-value="{i.delegator_rewards}"><span style="font-size: 0.85em;">{i.delegator_rewards:,.0f}</span></td>
            <td data-value="{i.indexing_rewards}"><span style="font-size: 0.85em;">{i.indexing_rewards:,.0f}</span></td>
            <td data-value="{i.total_rewards}"><span style="font-size: 0.85em;">{i.total_rewards:,.0f}</span></td>
            <td style="color: {'#c83349' if i.is_underserving else 'inherit'};"><span style="font-size: 0.85em;">{'Yes' if i.is_underserving else 'No'}</span></td>
            <td data-value="{i.final_score}"><span style="font-size: 0.85em;">{i.final_score:,.2f}</span></td>
            <td data-value="{i.final_score}"><span style="font-size: 0.85em;">{i.performance_flag}</span></td>
        </tr>
        """

    html += """
        </table>
        </div>

        <hr class="footer-divider">
        <div class="footer">
            ©<script>document.write(new Date().getFullYear())</script> 
            <a href="https://graphtools.pro">Graph Tools Pro</a> :: Made with ❤️ by 
            <a href="https://x.com/graphtronauts_c" target="_blank">Graphtronauts</a>
            for <a href="https://x.com/graphprotocol" target="_blank">The Graph</a> ecosystem 👨‍🚀
            <div style="margin-top: 8px;">
                <span style="font-size: 0.8rem;">For Info: <a href="https://x.com/pdiomede" target="_blank">@pdiomede</a> & <a href="https://x.com/PaulBarba12" target="_blank">@PaulBarba12</a></span>
            </div>
        </div>

        <script>

            document.addEventListener('DOMContentLoaded', () => {
                // Automatic sorting by Performance column (10th column, index 9)
                const table = document.querySelector('table');
                const rows = Array.from(table.querySelectorAll('tr:nth-child(n+2)'));
                rows.sort((a, b) => {
                    const aVal = parseFloat(a.children[9].getAttribute('data-value') || 0);
                    const bVal = parseFloat(b.children[9].getAttribute('data-value') || 0);
                    return aVal - bVal;
                });
                rows.forEach(row => table.appendChild(row));
                table.querySelector('th:nth-child(10)').classList.add('sorted-column');
                table.querySelector('th:nth-child(10)').setAttribute('data-sort-direction', 'asc');

                // Theme toggle logic
                const toggle = document.querySelector('input[type="checkbox"]');
                const htmlEl = document.documentElement;
                toggle.checked = true;
                htmlEl.classList.add('dark-mode');

                toggle.addEventListener('change', () => {
                    htmlEl.classList.toggle('light-mode');
                    htmlEl.classList.toggle('dark-mode');
                    document.body.classList.toggle('light-mode');
                    document.body.classList.toggle('dark-mode');
                });

                document.querySelectorAll('th').forEach((header, columnIndex) => {
                    header.addEventListener('click', () => {
                        const table = header.closest('table');
                        const rows = Array.from(table.querySelectorAll('tr:nth-child(n+2)'));
                        // Updated numericColumns to include all columns that should sort numerically
                        const numericColumns = [1, 2, 3, 4, 5, 6, 8, 9];
                        const isNumeric = numericColumns.includes(columnIndex);
                        const currentDirection = header.getAttribute('data-sort-direction');
                        const newDirection = currentDirection === 'asc' ? 'desc' : 'asc';
                        table.querySelectorAll('th').forEach(th => {
                            th.removeAttribute('data-sort-direction');
                            th.classList.remove('sorted-column');
                        });
                        header.setAttribute('data-sort-direction', newDirection);
                        header.classList.add('sorted-column');
                        rows.sort((a, b) => {
                            let aVal = a.children[columnIndex].getAttribute('data-value') || a.children[columnIndex].textContent.trim();
                            let bVal = b.children[columnIndex].getAttribute('data-value') || b.children[columnIndex].textContent.trim();
                            // Special handling for Underserving column ("Yes"/"No")
                            if (columnIndex === 7) {
                                aVal = aVal === "Yes" ? 1 : 0;
                                bVal = bVal === "Yes" ? 1 : 0;
                                aVal = Number(aVal);
                                bVal = Number(bVal);
                            } else if (isNumeric) {
                                aVal = parseFloat(aVal.replace(/,/g, '')) || 0;
                                bVal = parseFloat(bVal.replace(/,/g, '')) || 0;
                            }
                            if (aVal < bVal) return newDirection === 'asc' ? -1 : 1;
                            if (aVal > bVal) return newDirection === 'asc' ? 1 : -1;
                            return 0;
                        });
                        rows.forEach(row => table.appendChild(row));
                    });
                });
                const searchBox = document.getElementById('searchBox');
                searchBox.addEventListener('input', () => {
                    const query = searchBox.value.toLowerCase();
                    const rows = document.querySelectorAll('table tr:nth-child(n+2)');
                    rows.forEach(row => {
                        const text = row.textContent.toLowerCase();
                        row.style.display = text.includes(query) ? '' : 'none';
                    });
                });
            });

            function downloadCSV() {
                const link = document.createElement('a');
                link.href = 'indexers_output.csv';
                link.download = 'indexers_output.csv';
                document.body.appendChild(link);
                link.click();
                document.body.removeChild(link);
            }

            function filterByFlag(flag) {
                const rows = document.querySelectorAll('table tr:nth-child(n+2)');
                rows.forEach(row => {
                    const performanceCell = row.children[9];  // 9th column (Performance)
                    const flagText = performanceCell.textContent.trim();
                    if (flag === 'All') {
                        row.style.display = '';
                    } else {
                        row.style.display = flagText.startsWith(flag) ? '' : 'none';
                    }
                });

                // Update active class on buttons
                document.querySelectorAll('.filter-button').forEach(btn => {
                    btn.classList.remove('active-filter');
                });
                if (flag !== 'All') {
                    document.querySelectorAll('.filter-button').forEach(btn => {
                        if (btn.textContent.includes(flag)) {
                            btn.classList.add('active-filter');
                        }
                    });
                }
            }
        
        </script>

    </body>

    </html>

    """

    with open(html_file, "w", encoding="utf-8") as f:
        f.write(html)

    log_message(f"✅ Saved simple indexers HTML table: {html_file}")
# End Function 'generate_indexers_to_html'


# Function that generates the documentation file
def generate_docs_html(filename="docs.html"):
    global TOTAL_INDEXERS_COUNT, ACTIVE_INDEXERS_COUNT, ALLOCATED_INDEXERS_COUNT, SMALL_INDEXERS_ACTIVE_COUNT, MEDIUM_INDEXERS_ACTIVE_COUNT, LARGE_INDEXERS_ACTIVE_COUNT, MEGA_INDEXERS_ACTIVE_COUNT, INDEXERS_UNDERSERVING_COUNT, EXCLUDE_UPGRADE_INDEXER, EXCELLENT_INDEXERS_COUNT, FAIR_INDEXERS_COUNT, POOR_INDEXERS_COUNT

    small_thresh_fmt = format_grt(SMALL_INDEXER_THRESHOLD)
    medium_thresh_fmt = format_grt(MEDIUM_INDEXER_THRESHOLD)
    large_thresh_fmt = format_grt(LARGE_INDEXER_THRESHOLD)

    html_file = os.path.join(report_dir, filename)
    
    html = f"""
<html>
    <head>
        <meta charset='UTF-8'>
        <meta name="viewport" content="width=device-width, initial-scale=1.0">
        
        <title>Indexer Score Documentation - The Graph Network</title>
        <meta name="description" content="Learn how the Indexer Score is calculated for The Graph Network with detailed documentation.">
        <meta name="keywords" content="The Graph, Indexer Score, blockchain indexing, GRT, subgraph allocations, query fees, indexer performance">
        <meta name="robots" content="index, follow">

        <link rel="canonical" href="https://indexerscore.com/docs.html">

        <meta property="og:title" content="Indexer Score Documentation - The Graph Network">
        <meta property="og:description" content="Understand the Indexer Score calculation for The Graph Network with in-depth documentation.">
        <meta property="og:image" content="https://graphtools.pro/IndexerScore.jpg">
        <meta property="og:url" content="https://indexerscore.com/docs.html">
        <meta property="og:type" content="article">

        <meta name="twitter:card" content="summary_large_image">
        <meta name="twitter:title" content="Indexer Score Documentation - The Graph Network">
        <meta name="twitter:description" content="Understand the Indexer Score calculation for The Graph Network with in-depth documentation.">
        <meta name="twitter:image" content="https://graphtools.pro/IndexerScore.jpg">
        <meta name="twitter:site" content="@graphtronauts_c">

        <link rel="icon" type="image/png" href="https://graphtools.pro/favicon.ico">

        <style>
            body {{
                font-family: Arial, sans-serif;
                padding: 20px;
                background-color: #ffffff;
                color: #000;
            }}
        </style>

        <!-- Plausible Analytics -->
        <script defer data-domain="indexerscore.com" src="https://plausible.io/js/script.file-downloads.hash.outbound-links.pageview-props.tagged-events.js"></script>
        <script>
            window.plausible = window.plausible || function() {{
                (window.plausible.q = window.plausible.q || []).push(arguments)
            }}
        </script>

    </head>

    <body>
        <h1>📚 Documentation</h1>

        <p>
        💬 <a target="_blank" href="https://forum.thegraph.com/t/introducing-the-indexer-score/6501">Join the Forum Discussion</a>
        | 📄 <a  target="_blank" href="./indexer_score_documentation_v1.1.0.pdf" download>Download PDF Documentation</a>
        | 📊 <a href="./index.html">View the Dashboard</a>
        </p>
       
        <div style="margin-top: 20px; padding: 10px; background-color: #e1f5fe; border-left: 5px solid #2196f3; width: 60%; color: black !important; font-family: 'Courier New', Courier, monospace;">
            <h3>📏 Thresholds Used</h3>
            <ul>
                <li><strong>Total Indexers:</strong> {TOTAL_INDEXERS_COUNT}</li>
                <li><strong>Active Indexers:</strong> {ACTIVE_INDEXERS_COUNT}</li>
                <li><strong>Indexers with allocations:</strong> {ALLOCATED_INDEXERS_COUNT}</li>
                <br />
                <li><strong>Underserving Subgraphs Threshold:</strong> {UNDERSERVING_SUBGRAPHS_THRESHOLD:.0f} subgraphs</li>
                <li><strong>Indexers serving <span style="color: #c83349;">less than</span> {int(UNDERSERVING_SUBGRAPHS_THRESHOLD)} allocations:</strong> {INDEXERS_UNDERSERVING_COUNT}</li>
                <br />
                <li><strong>Small Active Indexers:</strong> {SMALL_INDEXERS_ACTIVE_COUNT} <span style="font-size: 90%;">(less than {small_thresh_fmt} GRT allocated)</span></li>
                <li><strong>Medium Active Indexers:</strong> {MEDIUM_INDEXERS_ACTIVE_COUNT} <span style="font-size: 90%;">(between {small_thresh_fmt} and {medium_thresh_fmt} GRT allocated)</span></li>
                <li><strong>Large Active Indexers:</strong> {LARGE_INDEXERS_ACTIVE_COUNT} <span style="font-size: 90%;">(more than {medium_thresh_fmt} and {large_thresh_fmt} GRT allocated)</span></li>
                <li><strong>Mega Active Indexers:</strong> {MEGA_INDEXERS_ACTIVE_COUNT} <span style="font-size: 90%;">(more than {large_thresh_fmt} GRT allocated)</span></li>
            </ul>
        </div>

        <div style="margin-top: 30px; padding: 15px; background-color: #fff3e0; border-left: 5px solid #ff9800; font-family: 'Courier New', Courier, monospace;">
            <h2>📈 How the Indexer Score is Calculated</h2>
            <p>
                The <strong>Indexer Score</strong> is a weighted combination of two critical performance metrics:
            </p>
            <ul>
                <li><strong>AER (Allocation Efficiency Ratio):</strong> Reflects how efficiently an indexer spreads its GRT across subgraphs. <strong>Weight: 70%</strong></li>
                <li><strong>QFR (Query Fee Ratio):</strong> Indicates how efficiently an indexer generates query fees relative to its allocated GRT. <strong>Weight: 30%</strong></li>
            </ul>
            <p>
                This blended metric evaluates an indexer’s overall performance, balancing allocation efficiency and query fee generation.
            </p>
        </div>

        <p>
            In the following sections, we break down the methodology used to compute both <strong>AER</strong> and <strong>QFR</strong>, 
            including their normalization processes. The final score is adjusted to a uniform scale where <strong>1 represents the best performance</strong> 
            and <strong>10 the worst</strong>, despite differing normalization directions for AER and QFR.
        </p>

        <div style="margin-top: 30px; padding: 15px; background-color: #e9f5ff; border-left: 5px solid #007bff; font-family: 'Courier New', Courier, monospace;">
            <h2 style="margin-top: 0;">📐 How Allocation Efficiency Ratio (AER) is calculated:</h2>
            <div style="margin: 20px 0; font-size: 16px; line-height: 1.6;">
                <div style="display: inline-block; text-align: center;">
                    <div><strong style="font-family: 'Courier New', Courier, monospace;">Total GRT Allocated</strong></div>
                    <div style="border-top: 2px solid #333; margin: 4px 0;"></div>
                    <div><strong style="font-family: 'Courier New', Courier, monospace;">(Number of Allocations * Average GRT per Allocation)</strong></div>
                </div>
            </div>
            <p>
                The Allocation Efficiency Ratio (AER) measures how effectively an indexer distributes their staked GRT across subgraphs.
                <br />
                ‼️ A <b>lower AER</b> reflects more efficient allocation, while a higher AER suggests over-concentration.
            </p>
            <p>
                Average allocation targets are based on indexer size:
            </p>
            <ul style="margin-left: 20px;">
                <li><strong>Small Indexers:</strong> 5,000 GRT per subgraph</li>
                <li><strong>Medium Indexers:</strong> 10,000 GRT per subgraph</li>
                <li><strong>Large Indexers:</strong> 20,000 GRT per subgraph</li>
                <li><strong>Mega Indexers:</strong> 40,000 GRT per subgraph</li>
            </ul>
        </div>

        <div style="margin-top: 30px; padding: 15px; background-color: #e9f5ff; border-left: 5px solid #007bff; font-family: 'Courier New', Courier, monospace;">
            <h2 style="margin-top: 0;">📏 How AER is Normalized:</h2>
            <div style="margin: 20px 0; font-size: 16px; line-height: 1.6;">
                <div style="display: inline-block; text-align: center;">
                    <div><strong style="font-family: 'Courier New', Courier, monospace;">Normalized AER = 1 + 9 × (min(AER, 500) / 500)</strong></div>
                </div>
            </div>
            <p>
                AER is normalized to a scale from <strong>1 (best)</strong> to <strong>10 (worst)</strong> to account for varying efficiency levels across indexers:
            </p>
            <ul style="margin-left: 20px;">
                <li><strong>Capping:</strong> AER values are capped at 500 to limit the impact of extreme outliers (e.g., highly concentrated allocations).</li>
                <li><strong>Scaling:</strong> The capped AER is scaled linearly from 0 to 500 onto a 1-to-10 range using the formula above.</li>
                <li><strong>Interpretation:</strong>
                    <ul>
                        <li>AER = 0 → Normalized = 1 (most efficient, best)</li>
                        <li>AER = 500 or higher → Normalized = 10 (least efficient, worst)</li>
                        <li>Example: AER = 23.788 → Normalized = 1 + 9 × (23.788 / 500) ≈ 1.43</li>
                    </ul>
                </li>
            </ul>
            <p>
                This ensures AER values are fairly compared, with lower ratios (better performance) resulting in lower scores.
            </p>
        </div>

        <div style="margin-top: 30px; padding: 15px; background-color: #fff3e0; border-left: 5px solid #ff9800; font-family: 'Courier New', Courier, monospace;">
            <h2 style="margin-top: 0;">🔍 How Query Fee Ratio (QFR) is calculated:</h2>
            <div style="margin: 20px 0; font-size: 16px; line-height: 1.6;">
                <div style="display: inline-block; text-align: center;">
                    <div><strong style="font-family: 'Courier New', Courier, monospace;">Query Fees Generated</strong></div>
                    <div style="border-top: 2px solid #333; margin: 4px 0;"></div>
                    <div><strong style="font-family: 'Courier New', Courier, monospace;">Total GRT Allocated</strong></div>
                </div>
            </div>
            <p>
                The Query Fee Ratio (QFR) measures how efficiently an indexer generates query fees per unit of allocated GRT.
                <br />
                ‼️ A <b>higher QFR</b> indicates better performance, as it means more query fees are earned per GRT allocated.
            </p>
        </div>

        <div style="margin-top: 30px; padding: 15px; background-color: #fff3e0; border-left: 5px solid #ff9800; font-family: 'Courier New', Courier, monospace;">
            <h2 style="margin-top: 0;">📏 How QFR is Normalized:</h2>
            <div style="margin: 20px 0; font-size: 16px; line-height: 1.6;">
                <div style="display: inline-block; text-align: center;">
                    <div><strong style="font-family: 'Courier New', Courier, monospace;">Normalized QFR = 10 - 9 × (min(QFR, 1.0) / 1.0)</strong></div>
                </div>
            </div>
            <p>
                QFR is normalized to a scale from <strong>10 (best)</strong> to <strong>1 (worst)</strong> to reflect its efficiency metric, where higher raw QFR values are better:
            </p>
            <ul style="margin-left: 20px;">
                <li><strong>Capping:</strong> QFR is capped at 1.0, representing a theoretical maximum where query fees equal the allocated GRT (a rare but ideal scenario).</li>
                <li><strong>Scaling:</strong> The capped QFR is scaled inversely from 0 to 1.0 onto a 10-to-1 range using the formula above. This inversion ensures that higher QFR values (better performance) result in higher normalized scores.</li>
                <li><strong>Interpretation:</strong>
                    <ul>
                        <li>QFR = 1 or higher → Normalized = 10 (most efficient, best)</li>
                        <li>QFR = 0 → Normalized = 1 (least efficient, worst)</li>
                        <li>Example: QFR = 0.002856 → Normalized = 10 - 9 × (0.002856 / 1) ≈ 9.97</li>
                    </ul>
                </li>
            </ul>
            <p>
                This normalization preserves QFR’s meaning: indexers generating more query fees relative to their allocations receive higher (better) scores, 
                while those with little to no fees score lower.
            </p>
        </div>

        <div style="margin-top: 30px; padding: 15px; background-color: #ede7f6; border-left: 5px solid #673ab7; font-family: 'Courier New', Courier, monospace;">
            <h2 style="margin-top: 0;">📊 How the Final Indexer Score is calculated:</h2>
            <p style="font-size: 16px; line-height: 1.6;">
                The final Indexer Score combines <strong>AER</strong> (70%) and <strong>QFR</strong> (30%) into a unified scale from <strong>1 (best)</strong> to <strong>10 (worst)</strong>. 
                Since AER and QFR have opposite normalization directions, QFR is adjusted before combining:
            </p>
            <ul style="font-size: 16px; line-height: 1.6; padding-left: 20px;">
                <li><strong>AER Normalized</strong>: Ranges from 1 (best, low AER) to 10 (worst, high AER).</li>
                <li><strong>QFR Normalized</strong>: Ranges from 10 (best, high QFR) to 1 (worst, low QFR). To align with the final score’s direction, it’s adjusted using: <strong>QFR Adjusted = 11 - Normalized QFR</strong>, flipping it to 1 (best) to 10 (worst).</li>
            </ul>
            <div style="margin: 20px 0; font-size: 16px; line-height: 1.6; text-align: center;">
                <div><strong>Final Score = (Normalized AER × 0.7) + ((11 - Normalized QFR) × 0.3)</strong></div>
            </div>
            <p style="font-size: 16px; line-height: 1.6;">
                Here’s how it works:
            </p>
            <ul style="margin-left: 20px;">
                <li><strong>AER Contribution:</strong> Normalized AER (1 to 10) is multiplied by 0.7, contributing 70% to the final score. A lower AER (better efficiency) lowers the score.</li>
                <li><strong>QFR Contribution:</strong> Normalized QFR (10 to 1) is inverted to QFR Adjusted (1 to 10) by subtracting it from 11, then multiplied by 0.3, contributing 30%. A higher raw QFR (better fee generation) results in a lower adjusted value, lowering the final score.</li>
                <li><strong>Final Scaling:</strong> The weighted sum naturally falls between 1 and 10, with 1 indicating the best performance (low AER, high QFR) and 10 the worst (high AER, low QFR).</li>
            </ul>
            <p style="font-size: 16px; line-height: 1.6;">
                <strong>Examples:</strong>
            </p>
            <ul style="margin-left: 20px;">
                <li>Best Case: AER = 0 (Normalized = 1), QFR = 1 (Normalized = 10, Adjusted = 1) → Final = (1 × 0.7) + (1 × 0.3) = 1.0</li>
                <li>Worst Case: AER = 500 (Normalized = 10), QFR = 0 (Normalized = 1, Adjusted = 10) → Final = (10 × 0.7) + (10 × 0.3) = 10.0</li>
                <li>Mixed Case: AER = 23.788 (Normalized ≈ 1.43), QFR = 0.002856 (Normalized ≈ 9.97, Adjusted ≈ 1.03) → Final = (1.43 × 0.7) + (1.03 × 0.3) ≈ 1.31</li>
            </ul>
            <p style="font-size: 16px; line-height: 1.6;">
                This method ensures that efficient allocation and high query fee generation both drive the final score toward 1 (best), 
                while poor performance in either metric increases it toward 10 (worst).
            </p>
            <p style="font-size: 16px; line-height: 1.6;">
                <strong>⚠️ 🔴 Underserving Penalty:</strong> Indexers serving fewer than 10 subgraphs are considered underserving and receive a penalty to their final score. 
                The penalty is calculated as <code>Penalty = 2.0 × (10 - number_of_subgraphs) / 10</code>, and the new score is capped at 10. 
                For example, an Indexer with 1 subgraph receives a penalty of 1.8, while an Indexer with 5 subgraphs receives a penalty of 1.0. 
                This ensures that Indexers are incentivized to support a diverse set of subgraphs, contributing to the health and decentralization of The Graph Network.
            </p>
        </div>

        <div style="margin-top: 30px; padding: 15px; background-color: #f0f4f8; border-left: 5px solid #607d8b; font-family: 'Courier New', Courier, monospace;">
            <h2 style="margin-top: 0;">🏅 Performance Flags</h2>
            <p style="font-size: 16px; line-height: 1.6;">
                Each indexer is assigned a <strong>Performance Flag</strong> based on its final Indexer Score, 
                providing a quick visual indicator of its overall efficiency and effectiveness:
            </p>
            <ul style="margin-left: 20px;">
                <li><strong>Excellent 🟢 (1.0 - 1.25):</strong> Top-tier performance with highly efficient allocation and strong query fee generation.</li>
                <li><strong>Fair 🟡 (1.26 - 2.5):</strong> Average performance with moderate inefficiencies or lower query fee generation, indicating room for improvement.</li>
                <li><strong>Poor 🔴 (2.51 - 10.0):</strong> Poor performance with significant inefficiencies or negligible query fees, requiring attention.</li>
            </ul>
            <p style="font-size: 16px; line-height: 1.6;">
                These ranges are designed to reflect the distribution of scores across the network, 
                distinguishing top performers (🟢) from those needing optimization (🔴).
            </p>
        </div>
    </body>
    </html>
    """

    with open(html_file, "w", encoding="utf-8") as f:
        f.write(html)

    log_message(f"✅ Saved HTML documentation file: {html_file}")
# End Function 'generate_docs_html'


if __name__ == "__main__":
    indexers = fetch_metrics()
    generate_indexers_to_csv(indexers)
    generate_indexers_to_html(indexers)
    generate_docs_html()