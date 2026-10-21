# Vendored third-party code

## `opendecisions_score.py`

The official OpenDecisions scorer (Intelligence, Calibration, Capability, TVD), copied verbatim.

- Source: https://huggingface.co/datasets/jgeuter/OpenDecisions/blob/ff0b640be2b72c7d06d9d656a34ad6eed79ec7fe/score.py
- Dataset revision: `ff0b640be2b72c7d06d9d656a34ad6eed79ec7fe`
- SHA-256: `478d14ae7a779c72e0dcd6c3225a3462e24faa8d7e8092afb0d6422ff84e4377`
- License: MIT (stated in the file's docstring). The dataset itself is CC-BY-SA-4.0.

Do not edit it. `ts/opendecisions.py` imports `score()` from it so that Capability is computed exactly as published.
To update, download the new revision and update the hash above.
