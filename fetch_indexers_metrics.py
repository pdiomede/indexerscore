import os
import json
import csv
import requests
from datetime import datetime, timezone, timedelta
from dotenv import load_dotenv
from dataclasses import dataclass, asdict
from typing import Optional

# v2.0.0 / 30-Oct-2025
# Author: Paolo Diomede
DASHBOARD_VERSION="2.0.0"


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

# Load delegator rewards threshold (default 30% if not set)
try:
    DELEGATOR_REWARDS_THRESHOLD = float(os.getenv("DELEGATOR_REWARDS_THRESHOLD", 30.0))
except ValueError:
    DELEGATOR_REWARDS_THRESHOLD = 30.0

# Load ENS cache file path
ENS_CACHE_FILE = os.getenv("ENS_CACHE_FILE", "ens_cache.json")

# Load ENS cache expiry duration
try:
    ENS_CACHE_EXPIRY_HOURS = int(os.getenv("ENS_CACHE_EXPIRY_HOURS", 24))
except ValueError:
    ENS_CACHE_EXPIRY_HOURS = 24


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
    log_message(f"📊 Delegator Rewards Threshold: {DELEGATOR_REWARDS_THRESHOLD:.0f}% (Excellent tier requirement)")
    
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

    SMALL_INDEXERS_ACTIVE_COUNT = len([i for i in indexers_data if i.is_active and i.total_stake < SMALL_INDEXER_THRESHOLD])
    MEDIUM_INDEXERS_ACTIVE_COUNT = len([i for i in indexers_data if i.is_active and SMALL_INDEXER_THRESHOLD <= i.total_stake < MEDIUM_INDEXER_THRESHOLD])
    LARGE_INDEXERS_ACTIVE_COUNT = len([i for i in indexers_data if i.is_active and i.total_stake >= MEDIUM_INDEXER_THRESHOLD and i.total_stake < LARGE_INDEXER_THRESHOLD])
    MEGA_INDEXERS_ACTIVE_COUNT = len([i for i in indexers_data if i.is_active and i.total_stake >= LARGE_INDEXER_THRESHOLD])

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
    normalized_qfr = normalize_qfr_inverted(qfr_values, cap=0.3)
    for i, score in zip(indexers_data, normalized_qfr):
        i.qfr_normalized = score if score is not None else 1.0

    # Calculate final score, apply penalty, and assign performance flag
    for i in indexers_data:
        # Ensure normalized values are not None
        qfr_norm = i.qfr_normalized if i.qfr_normalized is not None else 1.0
        
        # Calculate initial final score using only QFR
        # QFR is normalized 1-10 where 10 is best, so invert it to make 1 best and 10 worst
        qfr_adjusted = 11 - qfr_norm
        i.final_score = round(qfr_adjusted, 2)

        # Apply underserving penalty if applicable and store the penalty value
        if i.number_of_subgraphs < UNDERSERVING_SUBGRAPHS_THRESHOLD:
            # More severe penalty: 3.0 multiplier ensures underserving indexers are heavily penalized
            i.penalty = 3.0 * (UNDERSERVING_SUBGRAPHS_THRESHOLD - i.number_of_subgraphs) / UNDERSERVING_SUBGRAPHS_THRESHOLD
            i.final_score = min(10.0, i.final_score + i.penalty)
            i.final_score = round(i.final_score, 2)  # Round again after applying penalty
            log_message(f"⚠️ Underserving Penalty: Indexer {i.name} serving {i.number_of_subgraphs} subgraphs (threshold: {int(UNDERSERVING_SUBGRAPHS_THRESHOLD)}) → Penalty: +{i.penalty:.2f} → Final Score: {i.final_score}")
        else:
            i.penalty = 0.0  # Explicitly set penalty to 0 for non-underserving Indexers

        # Assign performance flag based on final score (after penalty) AND delegator rewards threshold
        if i.final_score is not None:
            # CRITICAL: Indexers with < 10% delegator rewards are automatically Poor
            if i.delegator_rewards_percentage < 10.0:
                i.performance_flag = "Poor 🔴"
                log_message(f"🔴 POOR (Low Delegator Rewards): Indexer {i.name} shares only {i.delegator_rewards_percentage:.2f}% with delegators (< 10% minimum) → Classified as Poor")
            # Check delegator rewards percentage for Excellent tier eligibility (>= 30%)
            # RELAXED: Increased threshold from 9.7 to 9.92 to allow more indexers into Excellent tier
            elif 1.0 <= i.final_score <= 9.92 and i.delegator_rewards_percentage >= DELEGATOR_REWARDS_THRESHOLD:
                i.performance_flag = "Excellent 🟢"
            elif 1.0 <= i.final_score <= 9.92 and i.delegator_rewards_percentage < DELEGATOR_REWARDS_THRESHOLD:
                # Good QFR but doesn't meet delegator rewards threshold - downgrade to Fair
                i.performance_flag = "Fair 🟡"
                log_message(f"⚠️ Delegator Rewards: Indexer {i.name} has good QFR (score: {i.final_score}) but delegator rewards {i.delegator_rewards_percentage:.2f}% < {DELEGATOR_REWARDS_THRESHOLD}% threshold → Downgraded to Fair")
            elif 9.93 <= i.final_score <= 9.97:
                i.performance_flag = "Fair 🟡"
            else:  # 9.98 to 10.0
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


# Normalize QFR values on a scale from 1 to 10 with a cap at 0.3.
# 10 is best (highest QFR), 1 is worst (lowest QFR).
# Returns a list of normalized QFR values between 1 and 10
def normalize_qfr_inverted(values, cap=0.3):
    normalized = []
    for v in values:
        capped_value = min(v, cap)
        norm_value = 1 + 9 * (capped_value / cap)
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

        writer = csv.DictWriter(csvfile, fieldnames=fieldnames, delimiter=',')
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

            @keyframes pulse {{
                0%, 100% {{
                    opacity: 1;
                    transform: scale(1);
                }}
                50% {{
                    opacity: 0.8;
                    transform: scale(1.05);
                }}
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
            <h1 style="margin-bottom: 4px;">
                Indexer Score Dashboard 
                <a href="whats-new.html" target="_blank" style="text-decoration: none;">
                    <span style="background: linear-gradient(135deg, #667eea 0%, #764ba2 100%); color: white; padding: 4px 12px; border-radius: 20px; font-size: 0.35em; font-weight: bold; margin-left: 10px; display: inline-block; box-shadow: 0 2px 8px rgba(102, 126, 234, 0.4); vertical-align: middle; animation: pulse 2s infinite; cursor: pointer;">✨ NEW</span>
                </a>
            </h1>
            <div style="text-align: center; font-size: 0.8em; color: var(--text-color); margin-top: 0; margin-bottom: 30px;">
                Generated on: {timestamp} - (updated every 6 hours) - v{DASHBOARD_VERSION}
                <p style="margin-top: 10px;">
                    💬 <a target="_blank" href="https://forum.thegraph.com/t/introducing-the-indexer-score/6501">Join the Forum Discussion</a>
                    | 📚 <a href="docs.html" target="_blank">How the Dashboard Works</a>
                    | 📄 <a  target="_blank" href="./indexer_score_documentation_v2.0.0.pdf" download>Download PDF Documentation</a>
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
                  <span class="tooltip-wrapper">Query Fees<span class="tooltip-text">Total query fees earned by the Indexer</span></span>
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
            <td data-value="{i.query_fees_earned}"><span style="font-size: 0.85em;">{i.query_fees_earned:,.0f}</span></td>
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
            <a href="https://x.com/pdiomede" target="_blank">pdiomede</a>
            for <a href="https://x.com/graphprotocol" target="_blank">The Graph</a> ecosystem 👨‍🚀
            <div style="margin-top: 4px; text-align: center;">
                <span style="font-size: 0.8rem;">
                    <img src="./github.png" style="width: 15px; height: 15px; vertical-align: middle; margin-left: 6px; margin-right: 4px;" />
                    <a href="https://github.com/pdiomede/indexerscore" target="_blank">GitHub repo</a>
                </span>
            </div>
        </div>

        <script>

            document.addEventListener('DOMContentLoaded', () => {
                // Automatic sorting by Performance tier first, then by Delegator Rewards % descending
                const table = document.querySelector('table');
                const rows = Array.from(table.querySelectorAll('tr:nth-child(n+2)'));
                
                // Define tier priority (lower number = higher priority)
                const tierPriority = {
                    'Excellent': 1,
                    'Fair': 2,
                    'Poor': 3
                };
                
                rows.sort((a, b) => {
                    // Primary sort: Performance tier
                    const aPerformance = a.children[10].textContent.trim();
                    const bPerformance = b.children[10].textContent.trim();
                    const aTier = aPerformance.split(' ')[0]; // Extract 'Excellent', 'Fair', or 'Poor'
                    const bTier = bPerformance.split(' ')[0];
                    const tierDiff = tierPriority[aTier] - tierPriority[bTier];
                    
                    if (tierDiff !== 0) return tierDiff;
                    
                    // Secondary sort: Delegator Rewards % descending
                    const aDelegatorRewards = parseFloat(a.children[4].getAttribute('data-value') || 0);
                    const bDelegatorRewards = parseFloat(b.children[4].getAttribute('data-value') || 0);
                    return bDelegatorRewards - aDelegatorRewards; // Descending order
                });
                
                rows.forEach(row => table.appendChild(row));
                table.querySelector('th:nth-child(11)').classList.add('sorted-column');
                table.querySelector('th:nth-child(11)').setAttribute('data-sort-direction', 'asc');

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
                        const numericColumns = [1, 2, 3, 4, 5, 6, 7, 9, 10];
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
                            if (columnIndex === 8) {
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
                    const performanceCell = row.children[10];  // 11th column (Performance)
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
        | 📄 <a  target="_blank" href="./indexer_score_documentation_v2.0.0.pdf" download>Download PDF Documentation</a>
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
                The <strong>Indexer Score</strong> is based on a single critical performance metric:
            </p>
            <ul>
                <li><strong>QFR (Query Fee Ratio):</strong> Measures how efficiently an indexer generates query fees relative to its allocated GRT. <strong>Weight: 100%</strong></li>
            </ul>
            <p>
                This metric directly evaluates an indexer's ability to generate revenue, making it the most important indicator of real-world performance and value to delegators.
            </p>
        </div>

        <p>
            In the following sections, we break down the methodology used to compute <strong>QFR</strong>, 
            including its normalization process. The final score is adjusted to a uniform scale where <strong>1 represents the best performance</strong> 
            (highest query fees relative to allocated stake) and <strong>10 the worst</strong> (little to no query fees generated).
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
                    <div><strong style="font-family: 'Courier New', Courier, monospace;">Normalized QFR = 1 + 9 × (min(QFR, 0.3) / 0.3)</strong></div>
                </div>
            </div>
            <p>
                QFR is normalized to a scale from <strong>1 (worst)</strong> to <strong>10 (best)</strong> to reflect its efficiency metric, where higher raw QFR values are better:
            </p>
            <ul style="margin-left: 20px;">
                <li><strong>Capping:</strong> QFR is capped at 0.3 (30%), representing excellent query fee performance based on the actual distribution observed in The Graph Network. This cap was chosen to properly differentiate between indexers, as most perform in the 0-10% range, with top performers reaching 10-30%.</li>
                <li><strong>Scaling:</strong> The capped QFR is scaled linearly from 0 to 0.3 onto a 1-to-10 range using the formula above. This ensures that higher QFR values (better performance) result in higher normalized scores.</li>
                <li><strong>Interpretation:</strong>
                    <ul>
                        <li>QFR = 0.3 or higher → Normalized = 10 (exceptional performance)</li>
                        <li>QFR = 0 → Normalized = 1 (no query fees generated)</li>
                        <li>Example: QFR = 0.2545 → Normalized = 1 + 9 × (0.2545 / 0.3) ≈ 8.64</li>
                        <li>Example: QFR = 0.014 → Normalized = 1 + 9 × (0.014 / 0.3) ≈ 1.42</li>
                    </ul>
                </li>
            </ul>
            <p>
                This normalization preserves QFR's meaning: indexers generating more query fees relative to their allocations receive higher (better) scores, 
                while those with little to no fees score lower.
            </p>
        </div>

        <div style="margin-top: 30px; padding: 15px; background-color: #ede7f6; border-left: 5px solid #673ab7; font-family: 'Courier New', Courier, monospace;">
            <h2 style="margin-top: 0;">📊 How the Final Indexer Score is calculated:</h2>
            <p style="font-size: 16px; line-height: 1.6;">
                The final Indexer Score is based solely on <strong>QFR (Query Fee Ratio)</strong>, scaled from <strong>1 (best)</strong> to <strong>10 (worst)</strong>.
            </p>
            <ul style="font-size: 16px; line-height: 1.6; padding-left: 20px;">
                <li><strong>QFR Normalized</strong>: Ranges from 1 (worst, low QFR) to 10 (best, high QFR).</li>
                <li><strong>Score Adjustment</strong>: To align with the scoring convention where 1 = best and 10 = worst, the normalized QFR is inverted using: <strong>Final Score = 11 - Normalized QFR</strong></li>
            </ul>
            <div style="margin: 20px 0; font-size: 16px; line-height: 1.6; text-align: center;">
                <div><strong>Final Score = 11 - Normalized QFR</strong></div>
            </div>
            <p style="font-size: 16px; line-height: 1.6;">
                <strong>How it works:</strong>
            </p>
            <ul style="margin-left: 20px;">
                <li><strong>High Query Fees:</strong> An indexer generating high query fees relative to allocated stake gets a high normalized QFR (e.g., 10), which becomes a low final score (11 - 10 = 1) = Excellent performance.</li>
                <li><strong>Low Query Fees:</strong> An indexer generating few or no query fees gets a low normalized QFR (e.g., 1), which becomes a high final score (11 - 1 = 10) = Poor performance.</li>
            </ul>
            <p style="font-size: 16px; line-height: 1.6;">
                <strong>Examples:</strong>
            </p>
            <ul style="margin-left: 20px;">
                <li>Best Case: QFR = 0.3+ (30%+) → Normalized = 10 → Final Score = 11 - 10 = <strong>1.0</strong> 🟢</li>
                <li>Excellent: QFR = 0.2545 (25.45%) → Normalized ≈ 8.64 → Final Score = 11 - 8.64 = <strong>2.36</strong> 🟢</li>
                <li>Fair: QFR = 0.05 (5%) → Normalized ≈ 2.5 → Final Score = 11 - 2.5 = <strong>8.5</strong> 🟡</li>
                <li>Poor: QFR = 0.014 (1.4%) → Normalized ≈ 1.42 → Final Score = 11 - 1.42 = <strong>9.58</strong> 🔴</li>
                <li>Worst Case: QFR = 0 (0%) → Normalized = 1 → Final Score = 11 - 1 = <strong>10.0</strong> 🔴</li>
            </ul>
            <p style="font-size: 16px; line-height: 1.6;">
                This method ensures that indexers who generate the most query fees relative to their allocated stake receive the best scores,
                providing a clear and direct measure of indexer performance that matters most to delegators.
            </p>
            <p style="font-size: 16px; line-height: 1.6;">
                <strong>⚠️ 🔴 Underserving Penalty:</strong> Indexers serving fewer than 10 subgraphs are considered underserving and receive a significant penalty to their final score. 
                The penalty is calculated as <code>Penalty = 3.0 × (10 - number_of_subgraphs) / 10</code>, and the new score is capped at 10. 
                For example, an Indexer with 1 subgraph receives a penalty of <strong>+2.7</strong>, while an Indexer with 5 subgraphs receives a penalty of <strong>+1.5</strong>. 
                This severe penalty ensures that Indexers are strongly incentivized to support a diverse set of subgraphs, contributing to the health and decentralization of The Graph Network. 
                Underserving indexers will typically fall into the Poor or Fair categories, regardless of their query fee performance.
            </p>
        </div>

        <div style="margin-top: 30px; padding: 15px; background-color: #f0f4f8; border-left: 5px solid #607d8b; font-family: 'Courier New', Courier, monospace;">
            <h2 style="margin-top: 0;">🏅 Performance Flags</h2>
            <p style="font-size: 16px; line-height: 1.6;">
                Each indexer is assigned a <strong>Performance Flag</strong> based on its final Indexer Score AND delegator rewards percentage, 
                providing a quick visual indicator of both query fee generation efficiency and delegator-friendliness:
            </p>
            <ul style="margin-left: 20px;">
                <li><strong>Excellent 🟢 (Score: 1.0 - 9.92 + Delegator Rewards ≥ {DELEGATOR_REWARDS_THRESHOLD:.0f}%):</strong> Average to exceptional query fee generation <strong>AND</strong> generous delegator rewards. These indexers demonstrate efficiency in converting their allocated stake into query fee revenue while sharing at least {DELEGATOR_REWARDS_THRESHOLD:.0f}% of total rewards with delegators. They generate meaningful query fees (typically 0.08-30% of allocated stake) and provide strong value for delegators through both performance and fair reward sharing. Most importantly, they share rewards generously with their delegators.</li>
                <li><strong>Fair 🟡 (Score: 9.93 - 9.97 OR Delegator Rewards 10-{int(DELEGATOR_REWARDS_THRESHOLD)-1}%):</strong> Below-average query fee generation OR insufficient delegator rewards. These indexers either generate small fees relative to their allocation (typically 0.005-0.08% of allocated stake), or they have good performance but share between 10-{int(DELEGATOR_REWARDS_THRESHOLD)-1}% of rewards with delegators. There is significant room for optimization through better subgraph selection or improved reward sharing.</li>
                <li><strong>Poor 🔴 (Score: 9.98 - 10.0 OR Delegator Rewards < 10%):</strong> Minimal query fee generation OR unfair reward distribution. These indexers either generate negligible fees relative to their allocation (<0.005% of allocated stake) OR share less than 10% of rewards with delegators. <strong style="color: #c83349;">CRITICAL: Any indexer sharing less than 10% of rewards with delegators is automatically classified as Poor</strong>, regardless of query fee performance. These indexers should prioritize both optimization and fair reward sharing.</li>
            </ul>
            <p style="font-size: 16px; line-height: 1.6;">
                <strong>🔴 Critical Rules:</strong>
            </p>
            <ul style="margin-left: 20px;">
                <li><strong>Automatic Poor Classification:</strong> Delegator Rewards < 10% → Poor tier (regardless of QFR score)</li>
                <li><strong>Excellent Tier Requirement:</strong> Delegator Rewards ≥ {DELEGATOR_REWARDS_THRESHOLD:.0f}% + Reasonable QFR (score ≤ 9.92) [Relaxed threshold]</li>
                <li><strong>Fair Tier Range:</strong> Delegator Rewards 10-{int(DELEGATOR_REWARDS_THRESHOLD)-1}% OR Score 9.93-9.97</li>
            </ul>
            <p style="font-size: 16px; line-height: 1.6;">
                These thresholds ensure that only indexers who provide meaningful value to delegators through BOTH strong performance AND fair reward sharing can achieve top ratings. 
                Indexers who keep more than 90% of rewards for themselves are considered unfair to delegators and will always be rated as Poor, 
                regardless of their technical performance.
            </p>
        </div>
    </body>
    </html>
    """

    with open(html_file, "w", encoding="utf-8") as f:
        f.write(html)

    log_message(f"✅ Saved HTML documentation file: {html_file}")
# End Function 'generate_docs_html'


# Function that generates the "What's New" HTML page
def generate_whats_new_html(filename="whats-new.html"):
    html_file = os.path.join(report_dir, filename)
    
    html = f"""
<html>
    <head>
        <meta charset='UTF-8'>
        <meta name="viewport" content="width=device-width, initial-scale=1.0">
        
        <title>What's New - Indexer Score Dashboard v2.0</title>
        <meta name="description" content="Discover the new simplified scoring system for The Graph Network indexers - focused on delegator protection and network health.">
        
        <link rel="icon" type="image/png" href="https://graphtools.pro/favicon.ico">
        
        <style>
            body {{
                font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, 'Helvetica Neue', Arial, sans-serif;
                line-height: 1.6;
                max-width: 1100px;
                margin: 0 auto;
                padding: 20px;
                background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
                color: #333;
            }}
            
            .container {{
                background: white;
                border-radius: 12px;
                padding: 40px;
                box-shadow: 0 10px 40px rgba(0,0,0,0.1);
            }}
            
            h1 {{
                color: #667eea;
                border-bottom: 3px solid #667eea;
                padding-bottom: 10px;
                margin-top: 0;
            }}
            
            h2 {{
                color: #764ba2;
                margin-top: 40px;
                padding-left: 10px;
                border-left: 4px solid #764ba2;
            }}
            
            h3 {{
                color: #667eea;
                margin-top: 25px;
            }}
            
            .badge {{
                background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
                color: white;
                padding: 6px 14px;
                border-radius: 20px;
                font-size: 0.9em;
                font-weight: bold;
                display: inline-block;
                margin-left: 10px;
            }}
            
            table {{
                width: 100%;
                border-collapse: collapse;
                margin: 20px 0;
                box-shadow: 0 2px 8px rgba(0,0,0,0.1);
            }}
            
            th, td {{
                padding: 12px;
                text-align: left;
                border-bottom: 1px solid #ddd;
            }}
            
            th {{
                background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
                color: white;
                font-weight: bold;
            }}
            
            tr:hover {{
                background-color: #f5f5f5;
            }}
            
            .highlight-box {{
                background: #f8f9fa;
                border-left: 4px solid #667eea;
                padding: 20px;
                margin: 20px 0;
                border-radius: 4px;
            }}
            
            .warning-box {{
                background: #fff3cd;
                border-left: 4px solid #ffc107;
                padding: 20px;
                margin: 20px 0;
                border-radius: 4px;
            }}
            
            .success-box {{
                background: #d4edda;
                border-left: 4px solid #28a745;
                padding: 20px;
                margin: 20px 0;
                border-radius: 4px;
            }}
            
            .danger-box {{
                background: #f8d7da;
                border-left: 4px solid #dc3545;
                padding: 20px;
                margin: 20px 0;
                border-radius: 4px;
            }}
            
            code {{
                background: #f4f4f4;
                padding: 2px 6px;
                border-radius: 3px;
                font-family: 'Courier New', monospace;
                font-size: 0.9em;
            }}
            
            pre {{
                background: #2d2d2d;
                color: #f8f8f2;
                padding: 20px;
                border-radius: 6px;
                overflow-x: auto;
                font-size: 0.9em;
            }}
            
            pre code {{
                background: none;
                color: inherit;
                padding: 0;
            }}
            
            .back-link {{
                display: inline-block;
                margin-bottom: 20px;
                color: white;
                background: #667eea;
                padding: 10px 20px;
                border-radius: 6px;
                text-decoration: none;
                font-weight: bold;
            }}
            
            .back-link:hover {{
                background: #764ba2;
            }}
            
            ul {{
                line-height: 1.8;
            }}
            
            li {{
                margin: 8px 0;
            }}
        </style>
    </head>
    
    <body>
        <div class="container">
            <a href="index.html" class="back-link">← Back to Dashboard</a>
            
            <h1>✨ What's New in Indexer Score v2.0</h1>
            
            <div class="success-box">
                <h3 style="margin-top:0;">🎉 A Simpler, Delegator-Focused Scoring System</h3>
                <p>We've completely redesigned the Indexer Score to focus on what matters most: <strong>query fee generation</strong> and <strong>fair reward sharing</strong>. We removed complexity to create a transparent system that protects delegators and promotes network health.</p>
            </div>
            
            <h2>🎯 The New Simple Approach</h2>
            <p>Unlike complex multi-factor scoring systems, our approach is straightforward:</p>
            <ul>
                <li><strong>Single Performance Metric:</strong> Query Fee Ratio (QFR) - how efficiently an indexer generates query fees relative to their allocated stake</li>
                <li><strong>Clear Reward Requirement:</strong> Delegator rewards percentage - how much indexers share with their delegators</li>
                <li><strong>Transparent Thresholds:</strong> Easy-to-understand tier requirements</li>
                <li><strong>Automatic Penalties:</strong> Built-in protections against underserving indexers</li>
            </ul>
            
            <h2>🛡️ Protecting Delegators</h2>
            
            <h3>Delegator Rewards Thresholds</h3>
            <p>The scoring system is designed to <strong>protect delegators</strong> by enforcing fair reward distribution:</p>
            
            <table>
                <tr>
                    <th>Delegator Rewards %</th>
                    <th>Impact</th>
                    <th>Tier Eligibility</th>
                </tr>
                <tr>
                    <td><strong>≥ {DELEGATOR_REWARDS_THRESHOLD:.0f}%</strong></td>
                    <td>✅ Fair reward sharing</td>
                    <td>Can achieve <strong>Excellent</strong> tier</td>
                </tr>
                <tr>
                    <td><strong>10-{int(DELEGATOR_REWARDS_THRESHOLD)-1}%</strong></td>
                    <td>⚠️ Below recommended</td>
                    <td>Maximum <strong>Fair</strong> tier</td>
                </tr>
                <tr>
                    <td><strong>&lt; 10%</strong></td>
                    <td>🔴 Unfair to delegators</td>
                    <td><strong>Automatic Poor</strong> classification</td>
                </tr>
            </table>
            
            <div class="danger-box">
                <h3 style="margin-top:0;">🔴 Key Protection</h3>
                <p><strong>Any indexer sharing less than 10% of rewards with delegators is automatically classified as Poor</strong>, regardless of their technical performance. This ensures delegators avoid indexers who keep more than 90% of rewards for themselves.</p>
            </div>
            
            <h3>Supporting Informed Decisions</h3>
            <p>The dashboard helps delegators by:</p>
            <ul>
                <li>🎯 <strong>Highlighting generous indexers</strong> - Top tier requires ≥{DELEGATOR_REWARDS_THRESHOLD:.0f}% reward sharing</li>
                <li>🚫 <strong>Filtering out unfair indexers</strong> - Automatic Poor rating for &lt;10% sharing</li>
                <li>📊 <strong>Transparent metrics</strong> - All data visible including exact reward percentages</li>
                <li>🏆 <strong>Easy comparison</strong> - Sorted by tier and reward generosity</li>
            </ul>
            
            <h2>🌐 Network Health Focus</h2>
            
            <h3>Query Fee Ratio (QFR)</h3>
            <p>We care about the <strong>health of The Graph Network</strong> by evaluating indexers who actually serve the network through meaningful query fee generation.</p>
            
            <div class="highlight-box">
                <h4>QFR Formula:</h4>
                <pre><code>QFR = Query Fees Earned / Total Allocated Stake</code></pre>
            </div>
            
            <p><strong>Why QFR Matters:</strong></p>
            <ul>
                <li>Measures real network contribution through query serving</li>
                <li>Reflects indexer's subgraph curation quality</li>
                <li>Indicates actual demand for indexer's services</li>
                <li>Shows revenue generation capability</li>
            </ul>
            
            <p><strong>Network Health Benefits:</strong></p>
            <ul>
                <li>Rewards indexers who serve active subgraphs</li>
                <li>Encourages proper subgraph selection</li>
                <li>Supports indexers providing value to dApp developers</li>
                <li>Promotes sustainable network economics</li>
            </ul>
            
            <h2>⚠️ Underserving Penalty</h2>
            <p>To maintain network decentralization and quality of service, we apply a <strong>significant penalty</strong> to indexers serving too few subgraphs.</p>
            
            <div class="warning-box">
                <h4>Penalty Formula:</h4>
                <pre><code>Penalty = 3.0 × (10 - number_of_subgraphs) / 10</code></pre>
                <p><strong>Threshold:</strong> Indexers must serve at least <strong>10 subgraphs</strong></p>
            </div>
            
            <table>
                <tr>
                    <th>Subgraphs Served</th>
                    <th>Penalty Applied</th>
                    <th>Impact</th>
                </tr>
                <tr>
                    <td>1 subgraph</td>
                    <td>+2.70 points</td>
                    <td>🔴 Severe</td>
                </tr>
                <tr>
                    <td>3 subgraphs</td>
                    <td>+2.10 points</td>
                    <td>🔴 High</td>
                </tr>
                <tr>
                    <td>5 subgraphs</td>
                    <td>+1.50 points</td>
                    <td>🟡 Moderate</td>
                </tr>
                <tr>
                    <td>7 subgraphs</td>
                    <td>+0.90 points</td>
                    <td>🟡 Low</td>
                </tr>
                <tr>
                    <td>10+ subgraphs</td>
                    <td>0.00 points</td>
                    <td>✅ No penalty</td>
                </tr>
            </table>
            
            <h2>📐 Score Calculation</h2>
            
            <h3>Step-by-Step Breakdown</h3>
            
            <div class="highlight-box">
                <h4>Step 1: Calculate QFR</h4>
                <pre><code>QFR = Query Fees Earned / Allocated Stake</code></pre>
                
                <h4>Step 2: Normalize QFR (1-10 scale)</h4>
                <pre><code>Normalized QFR = 1 + 9 × (min(QFR, 0.3) / 0.3)</code></pre>
                <p>QFR capped at 0.3 (30%) for fairness<br>Scale: 1 (worst) to 10 (best)</p>
                
                <h4>Step 3: Invert for Final Score (1 = best, 10 = worst)</h4>
                <pre><code>Final Score = 11 - Normalized QFR</code></pre>
                
                <h4>Step 4: Apply Underserving Penalty (if applicable)</h4>
                <pre><code>If (subgraphs < 10):
    Final Score = min(10.0, Final Score + Penalty)</code></pre>
                
                <h4>Step 5: Assign Performance Tier</h4>
                <pre><code>If (Delegator Rewards < 10%):
    Tier = Poor 🔴
Else If (Final Score ≤ 9.92 AND Delegator Rewards ≥ {DELEGATOR_REWARDS_THRESHOLD:.0f}%):
    Tier = Excellent 🟢
Else If (Final Score ≤ 9.97):
    Tier = Fair 🟡
Else:
    Tier = Poor 🔴</code></pre>
            </div>
            
            <h2>💡 Real Examples</h2>
            
            <h3>Example 1: Excellent Indexer (dataservices.eth)</h3>
            <div class="success-box">
                <ul style="margin: 0;">
                    <li><strong>Query Fees:</strong> 222,106 GRT</li>
                    <li><strong>Allocated Stake:</strong> 87,323,802 GRT</li>
                    <li><strong>Subgraphs:</strong> 930</li>
                    <li><strong>Delegator Rewards:</strong> 81.14%</li>
                </ul>
                
                <h4>Calculation:</h4>
                <pre><code>QFR = 222,106 / 87,323,802 = 0.00254 (0.254%)
Normalized QFR = 1 + 9 × (0.00254 / 0.3) = 1.08
Final Score = 11 - 1.08 = 9.92
Penalty = 0 (serves 930 subgraphs)
Delegator Rewards = 81.14% ≥ {DELEGATOR_REWARDS_THRESHOLD:.0f}% ✅

<strong style="color: #28a745;">Result: Excellent 🟢</strong></code></pre>
                
                <p><strong>Why Excellent:</strong> Generates meaningful query fees, shares 81.14% with delegators (exceptional!), serves many subgraphs, and meets all requirements.</p>
            </div>
            
            <h3>Example 2: Fair Indexer (streamingfastindexer.eth)</h3>
            <div class="warning-box">
                <ul style="margin: 0;">
                    <li><strong>Query Fees:</strong> 4,691,469 GRT</li>
                    <li><strong>Allocated Stake:</strong> 18,434,122 GRT</li>
                    <li><strong>Subgraphs:</strong> 750</li>
                    <li><strong>Delegator Rewards:</strong> 22.09%</li>
                </ul>
                
                <h4>Calculation:</h4>
                <pre><code>QFR = 4,691,469 / 18,434,122 = 0.2545 (25.45%)
Normalized QFR = 1 + 9 × (0.2545 / 0.3) = 8.63
Final Score = 11 - 8.63 = 2.37
Penalty = 0 (serves 750 subgraphs)
Delegator Rewards = 22.09% < {DELEGATOR_REWARDS_THRESHOLD:.0f}% ⚠️

<strong style="color: #ffc107;">Result: Fair 🟡</strong></code></pre>
                
                <p><strong>Why Fair (not Excellent):</strong> Excellent query fee generation (top performer!), BUT delegator rewards 22.09% &lt; {DELEGATOR_REWARDS_THRESHOLD:.0f}% threshold. Still serves many subgraphs (750).</p>
            </div>
            
            <h3>Example 3: Poor Indexer - Unfair Rewards (pinax2.eth)</h3>
            <div class="danger-box">
                <ul style="margin: 0;">
                    <li><strong>Query Fees:</strong> 2,856,044 GRT</li>
                    <li><strong>Allocated Stake:</strong> 15,999,999 GRT</li>
                    <li><strong>Subgraphs:</strong> 600</li>
                    <li><strong>Delegator Rewards:</strong> 0.00%</li>
                </ul>
                
                <h4>Calculation:</h4>
                <pre><code>QFR = 2,856,044 / 15,999,999 = 0.1785 (17.85%)
Normalized QFR = 1 + 9 × (0.1785 / 0.3) = 6.36
Final Score = 11 - 6.36 = 4.64
Penalty = 0 (serves 600 subgraphs)
Delegator Rewards = 0.00% < 10% 🔴

<strong style="color: #dc3545;">Result: Poor 🔴 (AUTOMATIC)</strong></code></pre>
                
                <p><strong>Why Poor:</strong> Excellent query fee generation (17.85%!), serves many subgraphs (600), BUT <strong>AUTOMATIC POOR</strong> because shares 0% with delegators. Critical failure: keeps 100% of rewards for themselves.</p>
            </div>
            
            <h2>🏆 Performance Tiers Summary</h2>
            
            <table>
                <tr>
                    <th>Tier</th>
                    <th>Requirements</th>
                    <th>Typical Profile</th>
                </tr>
                <tr>
                    <td><strong>🟢 Excellent</strong></td>
                    <td>Score ≤9.92 + Delegator Rewards ≥{DELEGATOR_REWARDS_THRESHOLD:.0f}%</td>
                    <td>Top performers who share fairly</td>
                </tr>
                <tr>
                    <td><strong>🟡 Fair</strong></td>
                    <td>Score ≤9.97 OR Delegator Rewards 10-{int(DELEGATOR_REWARDS_THRESHOLD)-1}%</td>
                    <td>Decent performance or moderate sharing</td>
                </tr>
                <tr>
                    <td><strong>🔴 Poor</strong></td>
                    <td>Score >9.97 OR Delegator Rewards &lt;10%</td>
                    <td>Low performance or unfair sharing</td>
                </tr>
            </table>
            
            <h2>🎯 Key Takeaways</h2>
            
            <div class="highlight-box">
                <ol>
                    <li><strong>Simplicity First:</strong> One primary metric (QFR) plus reward fairness</li>
                    <li><strong>Delegator Protection:</strong> Automatic Poor rating for &lt;10% reward sharing</li>
                    <li><strong>Excellence Requires Fairness:</strong> Need ≥{DELEGATOR_REWARDS_THRESHOLD:.0f}% reward sharing for top tier</li>
                    <li><strong>Network Health:</strong> Rewards real query fee generation</li>
                    <li><strong>Diversity Incentive:</strong> Penalty for serving too few subgraphs</li>
                    <li><strong>Transparent Scoring:</strong> Clear formulas, no black boxes</li>
                </ol>
            </div>
            
            <h2>📊 Current Network Statistics</h2>
            
            <p><strong>Total Indexers:</strong> {ALLOCATED_INDEXERS_COUNT} with allocations<br>
            <strong>Performance Distribution:</strong></p>
            <ul>
                <li>🟢 Excellent: {EXCELLENT_INDEXERS_COUNT} indexers ({EXCELLENT_INDEXERS_COUNT/ALLOCATED_INDEXERS_COUNT*100:.1f}%)</li>
                <li>🟡 Fair: {FAIR_INDEXERS_COUNT} indexers ({FAIR_INDEXERS_COUNT/ALLOCATED_INDEXERS_COUNT*100:.1f}%)</li>
                <li>🔴 Poor: {POOR_INDEXERS_COUNT} indexers ({POOR_INDEXERS_COUNT/ALLOCATED_INDEXERS_COUNT*100:.1f}%)</li>
            </ul>
            
            <p style="text-align: center; margin-top: 40px;">
                <a href="index.html" class="back-link">← Back to Dashboard</a>
            </p>
            
            <hr style="margin: 40px 0; border: none; border-top: 1px solid #ddd;">
            
            <p style="text-align: center; color: #666; font-size: 0.9em;">
                <strong>Version:</strong> {DASHBOARD_VERSION} | <strong>Last Updated:</strong> {timestamp}<br>
                Made with ❤️ by <a href="https://x.com/pdiomede" target="_blank">Paolo Diomede</a> for The Graph ecosystem 👨‍🚀
            </p>
        </div>
    </body>
</html>
    """
    
    with open(html_file, "w", encoding="utf-8") as f:
        f.write(html)
    
    log_message(f"✅ Saved What's New HTML file: {html_file}")
# End Function 'generate_whats_new_html'


if __name__ == "__main__":
    indexers = fetch_metrics()
    generate_indexers_to_csv(indexers)
    generate_indexers_to_html(indexers)
    generate_docs_html()
    generate_whats_new_html()