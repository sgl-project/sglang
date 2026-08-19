"""Pure data-movement kernels: sequence-parallel relayout, varlen pack/scatter, causal padding.

Every kernel here only moves values (plus zero fill), so each is bitwise
identical to the aten chain it replaces.
"""
