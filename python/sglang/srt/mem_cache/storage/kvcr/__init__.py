# SPDX-License-Identifier: Apache-2.0
"""KVCR (KV Cache Runner) as a direct external linker for UnifiedRadixCache.

Device pools move straight to and from KVCR-owned DRAM through NIXL; there is
no SGLang host pool in this mode. Peer reuse first hydrates the local KVCR tier
and only confirmed, claimed residency is ever reported as a cache hit.
"""
