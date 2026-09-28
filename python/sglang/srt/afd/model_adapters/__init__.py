"""Reusable adapter orchestration and concrete model-family bindings.

The transport and graph runtime live one level up. ``base`` holds the
family-neutral decoder contract; sibling modules identify a model family and
bind it to the metadata/backend behavior it needs.
"""
