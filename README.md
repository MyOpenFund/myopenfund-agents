# myopenfund-agents

Agents of the MyOpenFund data pipeline. The first one, A1, triages each night of harvesting runs: a
deterministic summary of the night goes to a language model, which returns schema-validated findings that
become GitHub issues or comments. A regression bench replays known nights and traps through any engine and
scores it against fixed thresholds.

- [Engines, settings and bench commands](docs/engines.md)
