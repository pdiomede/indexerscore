# Changelog

All notable changes to the Indexer Score Dashboard project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [2.0.0] - 2025-10-30

### 🎉 Major Release - Complete Scoring System Redesign

This release represents a complete overhaul of the Indexer Score system, focusing on simplicity, delegator protection, and network health.

### ✨ Added

#### New Scoring System
- **Single-Metric Scoring**: Simplified to use only Query Fee Ratio (QFR) as the primary performance metric (100% weight)
- **Delegator Rewards Requirement**: Added mandatory delegator reward percentage thresholds for tier classification
- **Automatic Poor Classification**: Indexers sharing <10% with delegators are automatically rated as Poor
- **Relaxed QFR Thresholds**: Score ≤9.92 for Excellent tier (previously more restrictive)
- **Enhanced Underserving Penalty**: Increased penalty multiplier from 2.0x to 3.0x for indexers serving <10 subgraphs

#### New Features
- **"What's New" Page**: Beautiful HTML page (`whats-new.html`) explaining all v2.0 changes with examples
- **Animated NEW Badge**: Pulsing badge on dashboard title linking to the What's New page
- **Enhanced Default Sorting**: Dashboard now sorts by performance tier first, then by delegator rewards percentage (descending)
- **Delegator Rewards Threshold Configuration**: New `DELEGATOR_REWARDS_THRESHOLD` parameter in `.env.costs` (default: 30%)
- **Comprehensive README.md**: Complete documentation with formulas, examples, and explanations

#### UI/UX Improvements
- Added pulse animation for the NEW badge
- Improved visual hierarchy with color-coded content boxes
- Enhanced code block styling with syntax highlighting
- Better mobile responsiveness

### 🔄 Changed

#### Scoring Methodology
- **Removed AER (Allocation Efficiency Ratio)**: No longer part of the final score calculation
- **QFR-Only Scoring**: Changed from 70% QFR + 30% AER to 100% QFR
- **Tier Thresholds Updated**:
  - Excellent: Score ≤9.92 + Delegator Rewards ≥30%
  - Fair: Score 9.93-9.97 OR Delegator Rewards 10-29%
  - Poor: Score ≥9.98 OR Delegator Rewards <10%

#### Classification Logic
- Performance tier now requires BOTH good QFR score AND fair delegator rewards
- Indexers can no longer achieve Excellent tier without sharing ≥30% with delegators
- Delegator reward sharing is now a key factor in all tier assignments

#### Display and Sorting
- Dashboard default sort: Performance tier (Excellent→Fair→Poor) then Delegator Rewards % (descending)
- Performance flag column now sorted by default on page load
- Enhanced logging showing which indexers are penalized or downgraded

### 🐛 Fixed

- **Bug Fix**: Corrected indexer size counting to use `total_stake` instead of `allocated_stake`
  - Small/Medium/Large/Mega indexer counts now accurately reflect total stake (own + delegated)
  - This fixes inconsistencies between indexer categorization and statistics

### 📚 Documentation

#### New Documentation Files
- `README.md`: Comprehensive guide with formulas, examples, and configuration
- `CHANGELOG.md`: This file - complete version history
- `reports/whats-new.html`: Interactive HTML page explaining v2.0 changes

#### Updated Documentation
- `reports/docs.html`: Updated with new scoring methodology
- Inline code comments improved throughout `fetch_indexers_metrics.py`
- Better explanation of delegator rewards thresholds

### 🔧 Technical Changes

#### Configuration
- Added `DELEGATOR_REWARDS_THRESHOLD` environment variable (default: 30.0)
- Improved error handling for environment variable loading
- Removed early logging that caused initialization errors

#### Code Structure
- Added `generate_whats_new_html()` function for the What's New page
- Enhanced `fetch_metrics()` with delegator rewards threshold logging
- Improved tier assignment logic with clear conditions and logging
- Better separation of concerns in scoring calculations

#### Performance
- Optimized default sorting algorithm for dashboard
- Improved caching for ENS lookups
- Better memory management for large indexer datasets

### 📊 Statistics (Current Network)

- **Total Indexers**: 82 with allocations
- **Performance Distribution**:
  - 🟢 Excellent: 20 indexers (24.4%)
  - 🟡 Fair: 11 indexers (13.4%)
  - 🔴 Poor: 51 indexers (62.2%)

### 🎯 Key Improvements for Delegators

1. **Clear Protection**: <10% delegator rewards = automatic Poor rating
2. **Fair Requirement**: ≥30% delegator rewards needed for Excellent tier
3. **Transparent Metrics**: All reward percentages visible in dashboard
4. **Easy Comparison**: Best indexers (by both performance and fairness) appear first
5. **Educational Content**: Comprehensive What's New page and README

### 🌐 Network Health Benefits

1. **Real Value Focus**: QFR directly measures network contribution
2. **Active Subgraph Incentive**: Underserving penalty encourages diversity
3. **Sustainable Economics**: Rewards indexers generating actual query fees
4. **Quality Curation**: Incentivizes proper subgraph selection

### ⚙️ Configuration Example

```bash
# .env.costs
DELEGATOR_REWARDS_THRESHOLD=30.0
UNDERSERVING_SUBGRAPHS_COUNT=10
SMALL_INDEXER=1000000
MEDIUM_INDEXER=20000000
LARGE_INDEXER=50000000
```

### 🔗 Links

- **Dashboard**: https://indexerscore.com
- **Forum Discussion**: https://forum.thegraph.com/t/introducing-the-indexer-score/6501
- **GitHub Repository**: https://github.com/pdiomede/indexerscore

---

## [1.1.0] - 2025-05-20

### Added
- Initial public release
- Combined AER + QFR scoring (30% AER, 70% QFR)
- Basic HTML dashboard
- CSV export functionality
- ENS name resolution
- Underserving penalty (2.0x multiplier)

### Changed
- Improved table sorting
- Enhanced mobile responsiveness

---

## [1.0.4] - 2025-04-01

### Added
- Beta release
- Core scoring functionality
- Basic reporting

---

## Contributing

We welcome contributions! Please see our [GitHub repository](https://github.com/pdiomede/indexerscore) for more information.

## Authors

- **Paolo Diomede** - [@pdiomede](https://x.com/pdiomede)

## Acknowledgments

- The Graph community for feedback and suggestions
- Delegators who helped test and validate the scoring system
- All indexers committed to fair reward sharing

---

**Made with ❤️ for The Graph ecosystem 👨‍🚀**

