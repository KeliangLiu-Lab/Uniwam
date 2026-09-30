# UniWAM Code Context

The accompanying UniWAM paper describes three ideas that are represented by
this release:

1. **Mixed-stream modeling.** Navigation and manipulation rows are sampled
   independently, packed along the batch dimension, and routed through separate
   visual/action interfaces while sharing the video/action backbone.
2. **Manipulation-ready grounding.** The manipulation stream uses camera-frame
   end-effector action targets and auxiliary image-plane supervision. The
   six-source parent recipe retains the 20D robot action plus the 6D auxiliary
   channel contract (`manip26`).
3. **Asynchronous execution.** The cloud policy and Piper edge client support
   prefix-conditioned action chunks, including prefix 0 and adaptive prefixes
   in the 6..12 range while the previous chunk is executed.

The 200k run represented here is the real-robot camera-frame parent model. MAP
dataset generation and MAP benchmark construction described in the paper are
not included in this focused release; the relevant shared-stream and
camera-frame interfaces are included so the parent policy can be reproduced
and deployed.

The canonical Python import namespace is `uniwam`. A thin `fastwam` package
forwards the old import path to it, preserving compatibility with existing
Wan/FastWAM checkpoints and Hydra configurations. Deployment uses
`uniwam_piper_cloud`, `uniwam_piper_common`, and `uniwam_piper_robot`, with
the corresponding `fastwam_piper_*` wrappers retained for old launchers.
