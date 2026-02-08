# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: LicenseRef-NvidiaProprietary
#
# Use of this software is governed by the terms and conditions of the
# NVIDIA End User License Agreement (EULA), available at:
# https://docs.nvidia.com/cutlass/media/docs/pythonDSL/license.html
#
# Any use, reproduction, disclosure, or distribution of this software
# and related documentation outside the scope permitted by the EULA
# is strictly prohibited.

from typing import Callable, Any, Iterable, Iterator, NamedTuple, Union, get_origin
import dataclasses
import itertools as it
from types import SimpleNamespace

from ..base_dsl.typing import as_numeric, Numeric, Constexpr
from ..base_dsl._mlir_helpers.arith import ArithValue
from ..base_dsl.common import DSLBaseError
from .._mlir import ir

# =============================================================================
# Tree Utils
# =============================================================================


class DSLTreeFlattenError(DSLBaseError):
    """Exception raised when tree flattening fails due to unsupported types."""

    def __init__(self, msg: str, type_str: str):
        super().__init__(msg)
        self.type_str = type_str


def unzip2(pairs: Iterable[tuple[Any, Any]]) -> tuple[list[Any], list[Any]]:
    """Unzip a sequence of pairs into two lists."""
    lst1, lst2 = [], []
    for x1, x2 in pairs:
        lst1.append(x1)
        lst2.append(x2)
    return lst1, lst2


def get_fully_qualified_class_name(x: Any) -> str:
    """
    Get the fully qualified class name of an object.

    Args:
        x: Any object

    Returns:
        str: Fully qualified class name in format 'module.class_name'

    Example:
        >>> get_fully_qualified_class_name([1, 2, 3])
        'builtins.list'
    """
    return f"{x.__class__.__module__}.{x.__class__.__qualname__}"


def is_frozen_dataclass(obj_or_cls: Any) -> bool:
    """
    Check if an object or class is a frozen dataclass.

    Args:
        obj_or_cls: Either a dataclass instance or class

    Returns:
        bool: True if the object/class is a dataclass declared with frozen=True,
              False otherwise

    Example:
        >>> from dataclasses import dataclass
        >>> @dataclass(frozen=True)
        ... class Point:
        ...     x: int
        ...     y: int
        >>> is_frozen_dataclass(Point)
        True
        >>> is_frozen_dataclass(Point(1, 2))
        True
    """
    cls = obj_or_cls if isinstance(obj_or_cls, type) else obj_or_cls.__class__

    return (
        dataclasses.is_dataclass(cls)
        and getattr(cls, "__dataclass_params__", None) is not None
        and cls.__dataclass_params__.frozen
    )


def is_dynamic_expression(x: Any) -> bool:
    """
    Check if an object implements the DynamicExpression protocol.

    Objects implementing this protocol must have both `__extract_mlir_values__`
    and `__new_from_mlir_values__` methods.

    Args:
        x: Any object to check

    Returns:
        bool: True if the object implements the DynamicExpression protocol,
              False otherwise
    """
    return all(
        hasattr(x, attr)
        for attr in ("__extract_mlir_values__", "__new_from_mlir_values__")
    )


def is_constexpr_field(field: dataclasses.Field) -> bool:
    """
    Check if a field is a constexpr field.
    """
    if field.type is Constexpr:
        return True
    elif get_origin(field.type) is Constexpr:
        return True
    return False


# =============================================================================
# PyTreeDef
# =============================================================================

class NodeType(NamedTuple):
    """
    Represents a node in a pytree structure.

    Attributes:
        name: String representation of the node type
        to_iterable: Function to convert node to iterable form
        from_iterable: Function to reconstruct node from iterable form
    """
    name: str
    to_iterable: Callable
    from_iterable: Callable


class PyTreeDef(NamedTuple):
    """
    Represents the structure definition of a pytree.

    Attributes:
        node_type: The type of this node
        node_metadata: SimpleNamespace metadata associated with this node
        child_treedefs: Tuple of child tree definitions
    """
    node_type: NodeType
    node_metadata: SimpleNamespace
    child_treedefs: tuple["PyTreeDef", ...]


@dataclasses.dataclass(frozen=True)
class Leaf:
    """
    Represents a leaf node in a pytree structure.

    Attributes:
        is_numeric: Whether this leaf contains a `Numeric` value
        is_none: Whether this leaf represents None
        node_metadata: SimpleNamespace metadata associated with this leaf
        ir_type_str: String representation of the IR type
    """
    is_numeric: bool = False
    is_none: bool = False
    node_metadata: SimpleNamespace = None
    ir_type_str: str = None


# =============================================================================
# Default to_iterable and from_iterable
# =============================================================================


def extract_dataclass_members(x: Any) -> tuple[list[str], list[Any]]:
    """
    Extract non-method, non-function attributes from a dataclass instance.

    Args:
        x: A dataclass instance

    Returns:
        tuple: (field_names, field_values) lists
    """
    fields = [field.name for field in dataclasses.fields(x)]

    # If the dataclass has extra fields, raise an error
    for k in x.__dict__.keys():
        if k not in fields:
            raise DSLTreeFlattenError(
                f"`{x}` has extra field `{k}`",
                type_str=get_fully_qualified_class_name(x),
            )

    if not fields:
        return [], []

    # record constexpr fields
    members = []
    constexpr_fields = []
    for field in dataclasses.fields(x):
        if is_constexpr_field(field):
            constexpr_fields.append(field.name)
            fields.remove(field.name)
            v = getattr(x, field.name)
            if is_dynamic_expression(v):
                raise DSLTreeFlattenError(
                    f"`{x}` has dynamic expression field `{field.name}` with a Constexpr type annotation `{field.type}`",
                    type_str=get_fully_qualified_class_name(x),
                )
        else:
            members.append(getattr(x, field.name))

    return fields, members, constexpr_fields


def default_dataclass_to_iterable(x: Any) -> tuple[SimpleNamespace, list[Any]]:
    """
    Convert a dataclass instance to iterable form for tree flattening.

    Extracts all non-method, non-function attributes that don't start with '__'
    and returns them along with metadata about the dataclass.

    Args:
        x: A dataclass instance

    Returns:
        tuple: (metadata, members) where metadata contains type info and field names,
               and members is the list of attribute values
    """
    fields, members, constexpr_fields = extract_dataclass_members(x)

    metadata = SimpleNamespace(
        type_str=get_fully_qualified_class_name(x),
        fields=fields,
        constexpr_fields=constexpr_fields,
        original_obj=x,
    )
    return metadata, members


def set_dataclass_attributes(
    instance: Any,
    fields: list[str],
    values: Iterable[Any],
    constexpr_fields: list[str],
) -> Any:
    """
    Set attributes on a dataclass instance.

    Args:
        instance: The dataclass instance
        fields: List of field names
        values: Iterable of field values
        is_frozen: Whether the dataclass is frozen

    Returns:
        The instance with attributes set
    """
    if not fields:
        return instance

    kwargs = dict(zip(fields, values))
    for field in constexpr_fields:
        kwargs[field] = getattr(instance, field)
    return dataclasses.replace(instance, **kwargs)

def default_dataclass_from_iterable(
    metadata: SimpleNamespace, children: Iterable[Any]
) -> Any:
    """
    Reconstruct a dataclass instance from iterable form.

    Handles both regular and frozen dataclasses appropriately.

    Args:
        metadata: Metadata containing type information and field names
        children: Iterable of attribute values to reconstruct the instance

    Returns:
        The reconstructed dataclass instance
    """
    instance = metadata.original_obj

    new_instance = set_dataclass_attributes(
        instance, metadata.fields, children, metadata.constexpr_fields
    )
    metadata.original_obj = new_instance
    return new_instance


def dynamic_expression_to_iterable(x: Any) -> tuple[SimpleNamespace, list[Any]]:
    """
    Convert a dynamic expression to iterable form.

    Uses the object's `__extract_mlir_values__` method to extract MLIR values.

    Args:
        x: A dynamic expression object

    Returns:
        tuple: (metadata, mlir_values) where metadata marks this as a dynamic expression
               and mlir_values are the extracted MLIR values
    """
    return (
        SimpleNamespace(is_dynamic_expression=1, original_obj=x),
        x.__extract_mlir_values__(),
    )


def dynamic_expression_from_iterable(
    metadata: SimpleNamespace, children: Iterable[Any]
) -> Any:
    """
    Reconstruct a dynamic expression from iterable form.

    Uses the object's `__new_from_mlir_values__` method to reconstruct from MLIR values.

    Args:
        metadata: Metadata containing the original object
        children: Iterable of MLIR values to reconstruct from

    Returns:
        The reconstructed dynamic expression object
    """
    return metadata.original_obj.__new_from_mlir_values__(list(children))


def default_dict_to_iterable(x: Any) -> tuple[SimpleNamespace, list[Any]]:
    """
    Convert a dict to iterable form.
    """
    if isinstance(x, SimpleNamespace):
        keys = list(x.__dict__.keys())
        values = list(x.__dict__.values())
    else:
        keys = list(x.keys())
        values = list(x.values())

    return (
        SimpleNamespace(
            type_str=get_fully_qualified_class_name(x), original_obj=x, fields=keys
        ),
        values,
    )


def default_dict_from_iterable(
    metadata: SimpleNamespace, children: Iterable[Any]
) -> Any:
    """
    Reconstruct a dict from iterable form.
    """
    instance = metadata.original_obj
    fields = metadata.fields
    is_simple_namespace = isinstance(instance, SimpleNamespace)

    for k, v in zip(fields, children):
        if is_simple_namespace:
            setattr(instance, k, v)
        else:
            instance[k] = v

    return instance


# =============================================================================
# Register pytree nodes
# =============================================================================

_node_types: dict[type, NodeType] = {}


def register_pytree_node(ty: type, to_iter: Callable, from_iter: Callable) -> NodeType:
    """
    Register a new node type for pytree operations.

    Args:
        ty: The type to register
        to_iter: Function to convert instances of this type to iterable form
        from_iter: Function to reconstruct instances of this type from iterable form

    Returns:
        NodeType: The created NodeType instance
    """
    nt = NodeType(str(ty), to_iter, from_iter)
    _node_types[ty] = nt
    return nt


def register_default_node_types() -> None:
    """Register default node types for pytree operations."""
    default_registrations = [
        (
            tuple,
            lambda t: (SimpleNamespace(length=len(t)), list(t)),
            lambda _, xs: tuple(xs),
        ),
        (
            list,
            lambda l: (SimpleNamespace(length=len(l)), list(l)),
            lambda _, xs: list(xs),
        ),
        (
            dict,
            default_dict_to_iterable,
            default_dict_from_iterable,
        ),
        (
            SimpleNamespace,
            default_dict_to_iterable,
            default_dict_from_iterable,
        ),
    ]

    for ty, to_iter, from_iter in default_registrations:
        register_pytree_node(ty, to_iter, from_iter)


# Initialize default registrations
register_default_node_types()


# =============================================================================
# tree_flatten and tree_unflatten
# =============================================================================

"""
Behavior of tree_flatten and tree_unflatten, for example:

```python
    a = (1, 2, 3)
    b = MyClass(a=1, b =[1,2,3])
```

yields the following tree:

```python
    tree_a = PyTreeDef(type = 'tuple',
                       metadata = {length = 3},
                       children = [
                           Leaf(type = int),
                           Leaf(type = int),
                           Leaf(type = int),
                       ],
                       )
    flattened_a = [1, 2, 3]
    tree_b = PyTreeDef(type = 'MyClass',
                       metadata = {fields = ['a','b']},
                       children = [
                           PyTreeDef(type = `list`,
                                     metadata = {length = 3},
                                     children = [
                                          Leaf(type=`int`),
                                          Leaf(type=`int`),
                                          Leaf(type=`int`),
                                     ],
                           ),
                           Leaf(type=int),
                       ],
                       )
    flattened_b = [1, 1, 2, 3]
```

Passing the flattened values and PyTreeDef to tree_unflatten to reconstruct the original structure.

``` python
    unflattened_a = tree_unflatten(tree_a, flattened_a)
    unflattened_b = tree_unflatten(tree_b, flattened_b)
```

yields the following structure:

``` python
    unflattened_a = (1, 2, 3)
    unflattened_b = MyClass(a=1, b =[1,2,3])
```

unflattened_a should be structurally identical to a, and unflattened_b should be structurally identical to b.

"""


def tree_flatten(x: Any) -> tuple[list[Any], PyTreeDef]:
    """
    Flatten a nested structure into a flat list of values and a tree definition.

    This function recursively traverses nested data structures (trees) and
    flattens them into a linear list of leaf values, while preserving the
    structure information in a PyTreeDef.

    Args:
        x: The nested structure to flatten

    Returns:
        tuple: (flat_values, treedef) where flat_values is a list of leaf values
               and treedef is the tree structure definition

    Raises:
        DSLTreeFlattenError: If the structure contains unsupported types

    Example:
        >>> tree_flatten([1, [2, 3], 4])
        ([1, 2, 3, 4], PyTreeDef(...))
    """
    children_iter, treedef = _tree_flatten(x)
    return list(children_iter), treedef


def get_registered_node_types_or_insert(x: Any) -> NodeType | None:
    """
    Get the registered node type for an object, registering it if necessary.

    This function checks if a type is already registered for pytree operations.
    If not, it automatically registers the type based on its characteristics:
    - Dynamic expressions get registered with dynamic expression handlers
    - Dataclasses get registered with default dataclass handlers

    Args:
        x: The object to get or register a node type for

    Returns:
        NodeType or None: The registered node type, or None if the type
                         cannot be registered
    """
    node_type = _node_types.get(type(x))
    if node_type:
        return node_type
    elif is_dynamic_expression(x):
        # If a class implements DynamicExpression protocol, register it before default dataclass one
        return register_pytree_node(
            type(x), dynamic_expression_to_iterable, dynamic_expression_from_iterable
        )
    elif dataclasses.is_dataclass(x):
        return register_pytree_node(
            type(x), default_dataclass_to_iterable, default_dataclass_from_iterable
        )
    else:
        return None


def create_leaf_for_value(
    x: Any,
    is_numeric: bool = False,
    is_none: bool = False,
    node_metadata: SimpleNamespace = None,
    ir_type_str: str = None,
) -> Leaf:
    """
    Create a Leaf node for a given value.

    Args:
        x: The value to create a leaf for
        is_numeric: Whether this is a numeric value
        is_none: Whether this represents None
        node_metadata: Optional metadata
        ir_type_str: Optional IR type string

    Returns:
        Leaf: The created leaf node
    """
    return Leaf(
        is_numeric=is_numeric,
        is_none=is_none,
        node_metadata=node_metadata,
        ir_type_str=ir_type_str or (str(x.type) if hasattr(x, "type") else None),
    )


def _tree_flatten(x: Any) -> tuple[Iterable[Any], PyTreeDef | Leaf]:
    """
    Internal function to flatten a tree structure.

    This is the core implementation of tree flattening that handles different
    types of objects including None, ArithValue, ir.Value, Numeric types,
    and registered pytree node types.

    Args:
        x: The object to flatten

    Returns:
        tuple: (flattened_values, treedef) where flattened_values is an iterable
               of leaf values and treedef is the tree structure

    Raises:
        DSLTreeFlattenError: If the object type is not supported
    """
    match x:
        case None:
            return [], create_leaf_for_value(x, is_none=True)

        case ArithValue() if is_dynamic_expression(x):
            v = x.__extract_mlir_values__()
            return v, create_leaf_for_value(
                x,
                node_metadata=SimpleNamespace(is_dynamic_expression=1, original_obj=x),
                ir_type_str=str(v[0].type),
            )

        case ArithValue():
            return [x], create_leaf_for_value(x, is_numeric=True)

        case ir.Value():
            return [x], create_leaf_for_value(x)

        case Numeric():
            v = x.__extract_mlir_values__()
            return v, create_leaf_for_value(
                x,
                node_metadata=SimpleNamespace(is_dynamic_expression=1, original_obj=x),
                ir_type_str=str(v[0].type),
            )

        case _:
            node_type = get_registered_node_types_or_insert(x)
            if node_type:
                node_metadata, children = node_type.to_iterable(x)
                children_flat, child_trees = unzip2(map(_tree_flatten, children))
                flattened = it.chain.from_iterable(children_flat)
                return flattened, PyTreeDef(
                    node_type, node_metadata, tuple(child_trees)
                )

            # Try to convert to numeric
            try:
                nval = as_numeric(x).ir_value()
                return [nval], create_leaf_for_value(nval, is_numeric=True)
            except Exception:
                raise DSLTreeFlattenError(
                    "Flatten Error", get_fully_qualified_class_name(x)
                )


def tree_unflatten(treedef: PyTreeDef, xs: list[Any]) -> Any:
    """
    Reconstruct a nested structure from a flat list of values and tree definition.

    This is the inverse operation of tree_flatten. It takes the flattened
    values and the tree structure definition to reconstruct the original
    nested structure.

    Args:
        treedef: The tree structure definition from tree_flatten
        xs: List of flat values to reconstruct from

    Returns:
        The reconstructed nested structure

    Example:
        >>> flat_values, treedef = tree_flatten([1, [2, 3], 4])
        >>> tree_unflatten(treedef, flat_values)
        [1, [2, 3], 4]
    """
    return _tree_unflatten(treedef, iter(xs))


def _tree_unflatten(treedef: PyTreeDef | Leaf, xs: Iterator[Any]) -> Any:
    """
    Internal function to reconstruct a tree structure.

    This is the core implementation of tree unflattening that handles
    different types of tree definitions including Leaf nodes and PyTreeDef nodes.

    Args:
        treedef: The tree structure definition
        xs: Iterator of flat values to reconstruct from

    Returns:
        The reconstructed object
    """
    match treedef:
        case Leaf(is_none=True):
            return None

        case Leaf(
            node_metadata=metadata
        ) if metadata and metadata.is_dynamic_expression:
            return metadata.original_obj.__new_from_mlir_values__([next(xs)])

        case Leaf(is_numeric=True):
            return as_numeric(next(xs))

        case Leaf():
            return next(xs)

        case PyTreeDef():
            children = (_tree_unflatten(t, xs) for t in treedef.child_treedefs)
            return treedef.node_type.from_iterable(treedef.node_metadata, children)


def _check_tree_equal(lhs: Union[PyTreeDef, Leaf], rhs: Union[PyTreeDef, Leaf]) -> bool:
    """
    Check if two tree definitions are structurally equal.

    This is a helper function for check_tree_equal that recursively compares
    tree structures.

    Args:
        lhs: Left tree definition (PyTreeDef or Leaf)
        rhs: Right tree definition (PyTreeDef or Leaf)

    Returns:
        bool: True if the trees are structurally equal, False otherwise
    """
    match (lhs, rhs):
        case (Leaf(), Leaf()):
            return lhs.is_none == rhs.is_none and lhs.ir_type_str == rhs.ir_type_str

        case (PyTreeDef(), PyTreeDef()):
            lhs_metadata = lhs.node_metadata
            rhs_metadata = rhs.node_metadata

            lhs_fields = getattr(lhs_metadata, "fields", [])
            rhs_fields = getattr(rhs_metadata, "fields", [])
            lhs_constexpr_fields = getattr(lhs_metadata, "constexpr_fields", [])
            rhs_constexpr_fields = getattr(rhs_metadata, "constexpr_fields", [])

            return (
                lhs.node_type == rhs.node_type
                and lhs_fields == rhs_fields
                and lhs_constexpr_fields == rhs_constexpr_fields
                and len(lhs.child_treedefs) == len(rhs.child_treedefs)
                and all(map(_check_tree_equal, lhs.child_treedefs, rhs.child_treedefs))
            )

        case _:
            return False


def check_tree_equal(lhs: PyTreeDef, rhs: PyTreeDef) -> int:
    """
    Check if two tree definitions are equal and return the index of first difference.

    This function compares two tree definitions and returns the index of the
    first child that differs, or -1 if they are completely equal.

    Args:
        lhs: Left tree definition
        rhs: Right tree definition

    Returns:
        int: Index of the first differing child, or -1 if trees are equal

    Example:
        >>> treedef1 = tree_flatten([1, [2, 3]])[1]
        >>> treedef2 = tree_flatten([1, [2, 4]])[1]
        >>> check_tree_equal(treedef1, treedef2)
        1  # The second child differs
    """
    assert len(lhs.child_treedefs) == len(rhs.child_treedefs)

    def find_first_difference(
        index_and_pair: tuple[int, tuple[PyTreeDef, PyTreeDef]]
    ) -> int:
        index, (l, r) = index_and_pair
        return index if not _check_tree_equal(l, r) else -1

    differences = map(
        find_first_difference, enumerate(zip(lhs.child_treedefs, rhs.child_treedefs))
    )
    return next((diff for diff in differences if diff != -1), -1)
