#include <c10/core/DeviceType.h>
#include <c10/core/MemoryFormat.h>
#include <c10/core/ScalarType.h>

#include <cstddef>
#include <cstdlib>
#include <iostream>
#include <stdexcept>
#include <string>

#include "tree_v2.h"
#include "tree_v2_impl.h"

namespace radix_tree_v2 {

void RadixTree::debug_print() const {
  m_impl->debug_print(std::clog);
}

static constexpr auto npos = std::size_t(-1);

void RadixTree::Impl::debug_print(std::ostream& os) const {
  static constexpr auto _check = [](bool condition, auto msg, std::size_t id = npos) {
    if (!condition) {
      std::string suffix = id == npos ? "" : " [id = " + std::to_string(id) + "]";
      throw std::runtime_error(std::string("RadixTree::debug_print failed: ") + msg + suffix);
    }
  };

  static constexpr auto _print_node = [](TreeNode* node, std::size_t depth, std::ostream& os) {
    const auto length = node->length();
    os << node->node_id << " [depth = " << depth << "] [len = " << length << "]";

    // placement status
    if (node->on_both()) {
      os << " [cpu + gpu]";
    } else if (node->on_gpu()) {
      os << " [gpu]";
    } else if (node->on_cpu()) {
      os << " [cpu]";
    } else {
      _check(false, "Node is not on GPU or CPU", node->node_id);
    }

    // IO status
    if (node->is_io_free()) {
      os << " [io = free]";
    } else if (node->is_io_device_to_host()) {
      os << " [io = gpu -> cpu]";
    } else if (node->is_io_host_to_device()) {
      os << " [io = cpu -> gpu]";
    } else {
      _check(false, "Node is in unknown IO state", node->node_id);
    }

    os << " [rc = " << node->ref_count << "]";
    os << " [hit = " << node->hit_count << "]";
  };

  static constexpr auto _print_indices = [](at::Tensor indices, std::ostream& os) {
    if (!indices.defined()) {
      os << "[[N/A]]";
      return indices;
    }
    indices = indices.to(c10::kCPU, c10::kLong, false, false, c10::MemoryFormat::Contiguous);
    const auto length = indices.numel();
    os << "[";
    auto* data_ptr = indices.data_ptr<int64_t>();
    for (const auto i : c10::irange(indices.size(0))) {
      os << data_ptr[i];
      if (i != length - 1) os << ", ";
    }
    os << "]";
    return indices;
  };

  os << "Evictable size: " << evictable_size() << std::endl;
  os << "Protected size: " << protected_size() << std::endl;
  os << "Total size: " << const_cast<Impl*>(this)->total_size() << std::endl;
  std::vector<std::tuple<TreeNode*, TreeNode*, token_slice>> stack;
  auto root = const_cast<TreeNode*>(&m_root);
  os << root->node_id << " [root]" << std::endl;
  for (const auto& [key, child] : *root) {
    stack.push_back({child.get(), root, key});
  }

  std::unordered_map<TreeNode*, std::size_t> depth_map;
  std::string indent_buffer;
  depth_map[root] = 0;
  std::vector<NodeHandle> visited_id;
  std::size_t evictable_size_real = 0;
  while (!stack.empty()) {
    const auto [node, parent, key] = stack.back();
    stack.pop_back();
    visited_id.push_back(node->node_id);
    const auto nid = node->node_id;
    _check(node != nullptr, "Node is null", nid);
    _check(node->on_gpu() || node->on_cpu(), "Node is not on GPU or CPU", nid);
    _check(node->parent() == parent, "Parent is not correct", nid);
    _check(key.size() == page_size && node->diff_key(key, 0) == page_size, "Key is not correct", nid);
    _check(depth_map.count(node) == 0, "Node is visited twice", nid);
    _check(m_node_map.count(nid) == 1, "Node is not in the map", nid);
    _check(m_node_map.at(nid) == node, "Node in the map is not the same as the one in the stack", nid);
    _check(!node->on_gpu() || parent->is_root() || parent->on_gpu(), "Node on GPU must have a GPU/root parent", nid);
    if (!node->is_io_free()) {
      _check(node->ref_count > 0, "Node is in IO state but not protected", nid);
      _check(node->on_both(), "Node in IO state must be on both CPU and GPU", nid);
    }

    if (node->on_gpu() && node->ref_count == 0) {
      evictable_size_real += node->length();
    }

    const auto depth = (depth_map[node] = depth_map[parent] + 1);
    indent_buffer.resize(depth * 2, ' ');
    os << indent_buffer;
    _print_node(node, depth, os);
    os << std::endl;
    for (const auto& [key, child] : *node) {
      stack.push_back({child.get(), node, key});
    }
  }

  _check(evictable_size_real == evictable_size(), "Evictable size is wrong");
  _check(m_node_map.count(root->node_id) == 1, "Root node is not in the map");
  _check(m_node_map.at(root->node_id) == root, "Root node in the map is not correct");

  std::sort(visited_id.begin(), visited_id.end());
  if (visited_id.size() != m_node_map.size() - 1) {
    // Some error in the tree, not all nodes are visited
    std::string id_list;
    id_list += "(visited: ";
    id_list += std::to_string(root->node_id) + " ";
    for (const auto& id : visited_id) {
      id_list += std::to_string(id) + " ";
    }
    id_list += "), (in map: ";
    for (const auto& [id, _] : m_node_map) {
      id_list += std::to_string(id) + " ";
    }
    id_list += ")";
    _check(false, "Not all nodes are visited " + id_list);
  }

  static const auto kSGLANG_RADIX_CPP_DEBUG_LIMIT = [] {
    const char* env = std::getenv("SGLANG_RADIX_CPP_DEBUG_LIMIT");
    const std::size_t default_limit = 16;
    if (env != nullptr) {
      try {
        return static_cast<std::size_t>(std::stoull(env));
      } catch (const std::exception& e) {
        std::cerr << "Invalid SGLANG_RADIX_CPP_DEBUG_LIMIT value: " << env  //
                  << ". Using default value =" << default_limit << std::endl;
      }
    }
    return default_limit;
  }();

  for (const auto nid : visited_id) {
    const auto node = m_node_map.at(nid);
    // print key and indices
    const auto& key = node->_unsafe_tokens();
    if (key.size() > kSGLANG_RADIX_CPP_DEBUG_LIMIT) {
      os << "Node " << nid << ": key is too long (" << key.size() << " tokens), skipping..." << std::endl;
      continue;
    }
    os << "Node " << nid << ": key = [";
    for (const auto& i : c10::irange(key.size())) {
      os << key[i];
      if (i != key.size() - 1) os << ", ";
    }

    _check(key.size() % page_size == 0, "Misaligned key", nid);

    os << "] device_indices = ";
    const auto device_indices = _print_indices(node->device_indices(), os);
    if (device_indices.defined()) {
      std::size_t length = device_indices.numel();
      _check(device_indices.dim() == 1, "Device indices must be 1D tensor", nid);
      _check(length == node->length(), "Wrong device indices size", nid);
    }

    os << " host_indices = ";
    const auto host_indices = _print_indices(node->host_indices(), os);
    if (host_indices.defined()) {
      std::size_t length = host_indices.numel();
      _check(host_indices.dim() == 1, "Host indices must be 1D tensor", nid);
      _check(length == node->length(), "Wrong host indices size", nid);
    }
    os << std::endl;
  }
}

}  // namespace radix_tree_v2
