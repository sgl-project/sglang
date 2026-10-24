"""Experimental full-forward dispatch for vendor monokernel plugins.

General plugins register specs through the sglang.srt.plugins entry point.
The runtime validates configuration, observes weight loading, and routes
supported steps through the selected spec; other steps use the model forward.
"""
