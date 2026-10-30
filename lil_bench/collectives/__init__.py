"""``lil-bench collectives``: which tensor-parallel all-reduce transport is fastest on this machine.

Inference servers on PCIe GPUs reduce activations between tensor-parallel
ranks with FlashInfer/b12x all-reduce kernels or NCCL. Which transport is
fastest depends on the PCIe and CPU fabric: on PCIe switches peer writes
("push") win; when every GPU hangs off its own CPU host bridge, writes to all
peers at once can collapse while reads ("pull") or a ring (NCCL) still run at
full speed. This command measures the fabric and every transport FlashInfer
offers, at every launch configuration its joint tuner considers, and reports
which one wins at which message size, how far the tuner's own choice is from
the fastest, and how NCCL compares. It first checks whether peer reads stall
(milliseconds-long waits that make pull transports slow on average).
"""

SCHEMA = "lil-collectives-result/1"
PROBE_VERSION = 4
# Message rows of a DeepSeek V4.1 TP decode/verification step (rows x hidden BF16 values).
ROWS = (1, 2, 4, 8, 16, 32, 64, 128, 192)
QUICK_ROWS = (1, 4, 16, 64, 128, 192)
HIDDEN = 5120
# FlashInfer transport name -> (description, family, short column name). Older and newer FlashInfer
# revisions name the payload-ready one-shot transports differently; both are listed.
TRANSPORTS = {
    "pull": ("pull 1-shot", "pull", "pull1"),
    "twoshot_pull": ("pull 2-shot", "pull", "pull2"),
    "remote_push_ll": ("push LL", "push", "LL"),
    "remote_push_simple": ("push simple", "push", "simple"),
    "tp2_remote_push": ("push LL (TP2)", "push", "tp2LL"),
    "tp2_remote_push_simple": ("push simple (TP2)", "push", "tp2simp"),
    "remote_push_full": ("push full", "push", "full"),
    "tp4_push_sentinel": ("push sentinel", "push", "sent"),
    "remote_push_rsag": ("push rsag", "push", "rsag"),
    "tp4_push_rsag": ("push rsag", "push", "rsag"),
    "twoshot": ("push 2-shot", "push", "push2"),
    "nccl": ("NCCL", "nccl", "NCCL"),
    "nccl_ring": ("NCCL ring", "nccl", "ring"),
}
# vLLM's NCCL settings for TP on PCIe: peer transfers across host bridges, 16 channels.
NCCL_REFERENCE = {"NCCL_P2P_LEVEL": "SYS", "NCCL_PROTO": "LL,LL128,Simple", "NCCL_MIN_NCHANNELS": "16",
                  "NCCL_MAX_NCHANNELS": "16", "NCCL_BUFFSIZE": "2097152"}


def describe(name: str) -> str:
    return TRANSPORTS.get(name, (name, "?", name))[0]


def family(name: str) -> str:
    return TRANSPORTS.get(name, (name, "?", name))[1]


def short(name: str) -> str:
    return TRANSPORTS.get(name, (name, "?", name))[2]
