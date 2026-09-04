"""Process-global PR2874 probe state.

The GDN core attention runs inside a custom op whose ``forward_batch`` comes from
the op context, not the object the model runner annotated, so per-forward probe
state lives here instead of on ``forward_batch`` attributes. One TP rank per
process; layers of one forward run sequentially on one thread.
"""

active = False
rank = None
layers = {}
first_nonfinite = None


def activate(tp_rank):
    global active, rank, layers, first_nonfinite
    active = True
    rank = tp_rank
    layers = {}
    first_nonfinite = None


def deactivate():
    global active
    active = False
