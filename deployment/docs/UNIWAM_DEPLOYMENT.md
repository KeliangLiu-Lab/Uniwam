# UniWAM Edge Deployment

Use the UniWAM Piper client with the matching source-specific cloud contract.
Start in dry-run mode, verify camera-frame transforms, IK, gripper ranges, and
latency, then enable physical execution through the robot safety procedure.

The exact prompts are listed in the repository README. The cloud service must
be started with the corresponding `uniwam_source*.yaml` contract and a
checkpoint from the six-source parent run.
