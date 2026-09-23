# Arenix market-data integrity protocol

Arenix stores sportsbook quotes as immutable observations. The first quote observed by Arenix is first_seen, not necessarily the bookmaker true opening line. A quote becomes a closing observation only when it was captured before event start and no later pregame quote is available.

## Leakage rules
- Never use a quote captured after kickoff for pregame inference.
- Never use 2026 live outcomes to tune historical thresholds.
- 2025 remains evaluation-only for the current NFL v1 protocol.
- Market probabilities are benchmark/gate inputs, not training features for the pure-model ensemble.
- Historical gate thresholds must be selected on pre-test data and frozen before test evaluation.

## Capture health
Each run records status, event count, new quotes, API quota telemetry, timestamps and errors. Unchanged quotes are deduplicated by provider/source hash.

## Stages
opening means first observed by Arenix; intermediate means a changed observation between first and latest; current is the latest stored observation. Closing will be assigned only from the final valid pre-kickoff observation.

## Economics
Synthetic no-vig prices are diagnostics only. Real ROI requires an actually observed sportsbook price. CLV compares the price taken or observed at decision time with the final valid pregame closing price from the same market/bookmaker or a separately defined consensus benchmark.
