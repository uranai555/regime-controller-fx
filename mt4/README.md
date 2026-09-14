# QuoteCaptureEA (Phase 4A)

Passive MT4 quote collector. It never sends orders and does not use DLLs.

## Usage

1. Compile `QuoteCaptureEA.mq4` in MetaEditor.
2. Run one MT4 terminal per broker on the **same Windows host**.
3. Attach the EA to the broker's XAUUSD chart and give each terminal a unique non-sensitive `SourceId` alias (`broker_a`, `broker_b`, ...).
4. Keep AutoTrading unnecessary/off; the EA has no trade calls.
5. Logs are written with `FILE_COMMON` under the shared terminal Common/Files directory.

The file format is schema v1: 8-byte header + fixed 84-byte records. Cross-terminal timing uses `GetTickCount()` (ms since Windows boot). `GetMicrosecondCount()` is recorded only for intra-EA diagnostics and must not be compared directly between terminals.
