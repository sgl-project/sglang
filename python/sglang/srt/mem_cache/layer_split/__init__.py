"""DSA layer-split host storage and coordinated L3 transfers.

GPU owner-broadcast reads remain in dsa_cache_layer_split. The pool assembler
opts HybridCacheController into the staging engine only for collective storage.
"""
