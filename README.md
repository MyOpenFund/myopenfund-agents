# myopenfund-agents

Agents of the MyOpenFund data pipeline. The first one, A1, triages each night of harvesting runs: the code
detects the problems and plans the GitHub issues or comments, and a language model only explains them, with
quotes the code checks. An alert never waits for the model. A regression bench replays known nights and traps:
the detector must be exact on all of them (CI, no model), and any engine can be scored on its explanations
against fixed thresholds.

- [Engines, settings and bench commands](docs/engines.md)
