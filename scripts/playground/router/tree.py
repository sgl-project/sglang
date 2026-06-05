import time
from collections import defaultdict
from typing import Dict, List


class Node:
    def __init__(self):
        self.children: Dict[str, Node] = dict()
        # We choose to use text because most of the use cases are text-to-text,
        # so we can save the tokenizing overhead.
        self.text: str = ""
        # Maps tenant_id to their last access timestamp
        self.tenant_last_access_time: Dict[str, float] = dict()
        self.parent = None


def shared_prefix_length(s1, s2):
    min_length = min(len(s1), len(s2))
    for i in range(min_length):
        if s1[i] != s2[i]:
            return i
    return min_length


class MultiTenantRadixTree:
    """
    Python Reference of Rust implementation of MultiTenantRadixTree

    MultiTenantRadixTree is the overlap of multiple radix trees by different tenant
    Each node in the tree can be owned by multiple tenants, allowing for efficient storage of common prefixes
    while maintaining tenant isolation.

    Key concepts:
    - Tenant: An entity that owns a subset of the stored strings
    - Each node tracks which tenants have access to it via tenant_last_access_time
    - The tree structure is shared, but queries can be filtered by tenant_id
    """

    def __init__(self):
        self.root = Node()

    def insert(self, s: str, tenant_id: str) -> None:
        """
        Insert string 's' and associate it with the given tenant_id.

        Args:
            s: The string to insert
            tenant_id: The identifier of the tenant who owns this string
        """
        curr = self.root
        curr_idx = 0
        curr.tenant_last_access_time[tenant_id] = time.time()

        while curr_idx < len(s):
            matched_node = None
            if s[curr_idx] in curr.children:
                matched_node = curr.children[s[curr_idx]]

            if matched_node is None:
                # No match => create a new node
                new_node = Node()
                new_node.text = s[curr_idx:]
                new_node.parent = curr

                curr.children[s[curr_idx]] = new_node
                curr_idx = len(s)
                curr = new_node
                curr.tenant_last_access_time[tenant_id] = time.time()
            else:
                shared_len = shared_prefix_length(s[curr_idx:], matched_node.text)

                # 1. If the matched text is shorter than the node text => split the node
                if shared_len < len(matched_node.text):
                    # Split structure: [matched_node] => [new_node] -> [contracted_matched_node]

                    matched_text = matched_node.text[:shared_len]
                    unmatched_text = matched_node.text[shared_len:]

                    new_node = Node()
                    new_node.text = matched_text
                    new_node.children = {unmatched_text[0]: matched_node}
                    new_node.parent = curr
                    new_node.parent.children[matched_text[0]] = new_node
                    new_node.tenant_last_access_time = (
                        matched_node.tenant_last_access_time.copy()
                    )

                    # Contract matched node
                    matched_node.text = unmatched_text
                    matched_node.parent = new_node

                    curr_idx += shared_len
                    curr = new_node
                    curr.tenant_last_access_time[tenant_id] = time.time()
                # 2. If the matched text is longer or equal to the node text => walk down the node
                else:
                    curr_idx += shared_len
                    curr = matched_node
                    curr.tenant_last_access_time[tenant_id] = time.time()

    def prefix_match(self, s: str) -> tuple[str, int]:
        """
        Match string 's' with multiple tenants' trees in one operation.

        Args:
            s: The string to match

        Returns:
            Tuple(str, int): The longest prefix of 's' that matches the tree and the first tenant_id that own the matched prefix
        """
        curr = self.root
        curr_idx = 0

        ret_text = ""
        ret_tenant = None

        while curr_idx < len(s):
            matched_node = None
            if s[curr_idx] in curr.children:
                matched_node = curr.children[s[curr_idx]]

            if matched_node is None:
                break

            shared_len = shared_prefix_length(s[curr_idx:], matched_node.text)
            if shared_len == len(matched_node.text):
                curr_idx += shared_len
                curr = matched_node
            else:
                curr_idx += shared_len
                curr = matched_node
                break

        selected_tenant = list(curr.tenant_last_access_time.keys())[0]

        # traverse back to the root to update last access time for the selected tenant
        while curr != self.root:
            curr.tenant_last_access_time[selected_tenant] = time.time()
            curr = curr.parent

        return s[:curr_idx], selected_tenant

    def evict_tenant_data(self, max_size_per_tenant: Dict[str, int]) -> None:
        """
        Evict data for tenants that have exceeded their storage limits.

        Args:
            max_size_per_tenant: Dictionary mapping tenant_id to their maximum allowed storage size
        """

        def leaf_of(node):
            """
            If the node is a leaf for a tenant, add tenant_id to the return list
            This will return list of tenant ids
            If not a leaf for all tenants, return []
            """
            candidates = dict([(k, True) for k in node.tenant_last_access_time.keys()])

            for n in node.children.values():
                for c in n.tenant_last_access_time.keys():
                    candidates[c] = False

            return [k for k, v in candidates.items() if v]

        # maintain a heap with (time, tenant, node) as the value
        import heapq

        # 1. traverse the tree to
        #   a. add all the leaves into a heap (a node with N tenants will be added N times into the heap)
        #   b. calculate the used size for each tenant
        # do a dfs with stack
        stack = [self.root]
        pq = []
        used_size_per_tenant = defaultdict(int)

        while stack:
            curr = stack.pop()
            for t in curr.tenant_last_access_time.keys():
                used_size_per_tenant[t] += len(curr.text)

            for c in curr.children.values():
                stack.append(c)

            # if the node is a leaf for a tenant, add the tenant to the heap
            tenants = leaf_of(curr)
            for t in tenants:
                heapq.heappush(pq, (curr.tenant_last_access_time[t], t, curr))

        # 2. pop the heap
        #   a. if the tenant's used size is less than the limit, continue
        #   b. if the tenant's used size is greater than the limit, remove the leaf and update the used size, and add its parent to the heap
        while len(pq) > 0:
            time, tenant, node = heapq.heappop(pq)
            if used_size_per_tenant[tenant] <= max_size_per_tenant[tenant]:
                continue

            # remove the leaf
            used_size_per_tenant[tenant] -= len(node.text)
            del node.tenant_last_access_time[tenant]
            # if no children and no tenants, remove the node
            if len(node.children) == 0 and len(node.tenant_last_access_time) == 0:
                del node.parent.children[node.text[0]]

            # add its parent to the heap
            if tenant in leaf_of(node.parent):
                heapq.heappush(
                    pq,
                    (node.parent.tenant_last_access_time[tenant], tenant, node.parent),
                )

    def get_used_size_per_tenant(self) -> Dict[str, int]:
        """
        Calculate the used storage size for each tenant.

        Returns:
            Dict[str, int]: A dictionary mapping tenant_id to their used storage size
        """
        used_size_per_tenant = defaultdict(int)

        stack = [self.root]
        while stack:
            curr = stack.pop()
            for t in curr.tenant_last_access_time.keys():
                used_size_per_tenant[t] += len(curr.text)

            for c in curr.children.values():
                stack.append(c)

        return used_size_per_tenant

    def remove_tenant(self, tenant_id: str) -> None:
        """
        Remove all data associated with a specific tenant from the tree.
        This operation maintains the integrity of the shared tree structure while
        removing only the specified tenant's access information.

        Args:
            tenant_id: The identifier of the tenant whose data should be removed
        """
        # TODO: Implementation needed
        pass

    def pretty_print(self) -> str:
        """
        Returns a string representation of the tree showing the structure, tenant ownership,
        and leaf status for each node.

        Returns:
            str: A formatted string showing the tree hierarchy with tenant information
        """

        def _node_to_str(node: Node, prefix: str = "", is_last: bool = True) -> str:
            # Current node representation
            node_str = prefix
            node_str += "└── " if is_last else "├── "

            # Add node text
            node_str += f"'{node.text}' ["

            # Add tenant information including both timestamp and leaf status
            tenant_info = []
            for tid, ts in node.tenant_last_access_time.items():
                time_str = (
                    time.strftime("%H:%M:%S.", time.localtime(ts))
                    + f"{(ts % 1):0.3f}"[2:]
                )
                tenant_info.append(f"{tid} | {time_str}")

            node_str += ", ".join(tenant_info)
            node_str += "]\n"

            # Handle children
            children = list(node.children.items())
            for i, (char, child) in enumerate(children):
                is_last_child = i == len(children) - 1
                # Adjust prefix for children based on whether this is the last child
                new_prefix = prefix + ("    " if is_last else "│   ")
                node_str += _node_to_str(child, new_prefix, is_last_child)

            return node_str

        if not self.root.children:
            return "Empty tree"

        # Start with root's children since root itself is just an empty node
        result = ""
        children = list(self.root.children.items())
        for i, (char, child) in enumerate(children):
            is_last = i == len(children) - 1
            result += _node_to_str(child, "", is_last)

        return result
