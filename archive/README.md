# Local archive

This directory separates historical experiments from the maintained GitHub-facing code.

- `original-rl-cpp.zip`: a snapshot of the public coverage-path-planning code
  used as the foundation of the later `Paper_main_pi_unet` RL environment.
- `raw_research/`: the earlier PI-U-Net working material from the original
  local `rl-cpp` folder: notebooks, model weights, datasets, generated images,
  and exploratory files.
- `reference/Paper_main_pi_unet/`: the later RL integration working tree used
  to verify the imported PI-U-Net checkpoint, gas environment, and modified
  policy architecture.
- `generated/`: previous smoke-test agents, logs, and other generated runs.

The actual lineage is PI-U-Net training in the original local work folder,
followed by checkpoint transfer into the separately modified RL project. These
two sources must not be interpreted as one upstream repository state.

Everything below `archive/` except this manifest is ignored by Git because it
is historical, generated, or too large for the installable distribution.
