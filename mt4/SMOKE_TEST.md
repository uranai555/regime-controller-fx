# Phase 4A — Real MT4 Smoke Test

This is the final instrumentation gate before a 30-minute passive XAUUSD capture.

## 1. Compile

Open `QuoteCaptureEA.mq4` in the broker MT4 MetaEditor and compile it.

**PASS:**

```text
0 errors, 0 warnings
```

Do not continue if MetaEditor reports any error or warning.

## 2. Run on the same Windows host

Use at least two MT4 terminals on the same physical Windows installation.

For each terminal:

- attach the EA to the broker's actual gold/USD chart (`XAUUSD`, `XAUUSDm`, `GOLD`, etc.)
- set `CanonicalSymbol = XAUUSD`
- use a unique `SourceId`, e.g. `broker_a`, `broker_b`
- AutoTrading may remain OFF; this collector contains no order calls
- keep the PC awake and do not reboot during capture

Capture for roughly 1–2 minutes under normal liquid conditions. Do not start with CPI/NFP.

Logs are written through `FILE_COMMON` to the MT4 shared Common/Files directory.

## 3. Validate the real MQL4-produced binary files

Run the validator from the repository root. Module invocation is preferred and works in a clean checkout without first installing the package:

```bash
python -m scripts.validate_mt4_smoke \
  --source broker_a="/path/to/microstructure_broker_a_XAUUSD_....bin" \
  --source broker_b="/path/to/microstructure_broker_b_XAUUSD_....bin" \
  --symbol XAUUSD \
  --output-dir output/microstructure-smoke
```

Direct script execution is also supported:

```bash
python scripts/validate_mt4_smoke.py --help
```

On Windows/PowerShell:

```powershell
python -m scripts.validate_mt4_smoke `
  --source "broker_a=C:\...\Common\Files\microstructure_broker_a_XAUUSD_....bin" `
  --source "broker_b=C:\...\Common\Files\microstructure_broker_b_XAUUSD_....bin" `
  --symbol XAUUSD `
  --output-dir output\microstructure-smoke
```

The validator checks:

- at least two distinct source aliases
- at least two distinct resolved capture files (symlink/path aliases cannot fake a second broker)
- real file exists
- `8 + N*84` binary layout
- schema/header/parser compatibility
- sequence continuity
- finite/valid tick values through the parser
- capture duration >= 30s
- positive median spread
- cross-terminal `GetTickCount()` wrap alignment
- capture overlap >= 95% of the shorter capture
- end-to-end report generation

**PASS:** final line is:

```text
SMOKE_PASS
```

Expected output files:

```text
broker_lag_matrix.csv
broker_fingerprint.json
lag_stability.json
cost_stress.json
microstructure_report.md
```

## 4. Manual checks that code cannot prove

Before accepting `SMOKE_PASS`, confirm manually:

- each MT4 terminal is connected to the intended broker/account environment
- each chart represents the same economic instrument (spot/CFD gold vs USD)
- broker symbol aliases have not accidentally selected a different contract
- both terminals ran on the same Windows host
- no terminal reconnect/restart occurred during the capture

## 5. Decision

### GO_30MIN_CAPTURE

Only if all are true:

```text
MetaEditor compile: 0 errors / 0 warnings
validate_mt4_smoke: SMOKE_PASS
manual instrument/host checks: PASS
```

### BLOCKED_MT4

Any compile problem, parser/schema mismatch, write failure, invalid sequence, duplicate capture file, or missing output.

### BLOCKED_ENV

Different hosts, bad capture overlap, wrong gold instrument, sleep/reboot/reconnect contamination.

The 30-minute capture is still instrumentation validation, not proof of profitability or permission for live trading.
