use std::collections::HashMap;
use std::mem;

#[derive(Clone)]
pub struct Node {
    pub children: HashMap<usize, Node>, // the key is first id of the child because each child must have unique first id
    pub ids: Vec<usize>,
    pub count: usize,
}

pub struct RadixTree {
    pub root: Node,
}

fn common_prefix_len(a: &[usize], b: &[usize]) -> usize {
    let mut i = 0;
    while i < a.len() && i < b.len() && a[i] == b[i] {
        i += 1;
    }
    i
}

impl Default for RadixTree {
    fn default() -> Self {
        Self::new()
    }
}

impl RadixTree {
    pub fn new() -> Self {
        RadixTree {
            root: Node {
                children: HashMap::new(),
                ids: Vec::new(),
                count: 0,
            },
        }
    }

    pub fn insert(&mut self, input_ids: &[usize]) {
        let mut curr = &mut self.root;
        curr.count += 1;

        let mut curr_idx = 0;
        let input_ids_len = input_ids.len();

        while curr_idx < input_ids_len {
            let mut match_info = None;

            for key in curr.children.keys() {
                let prefix_len = common_prefix_len(&input_ids[curr_idx..], &curr.children[key].ids);

                if prefix_len == 0 {
                    continue;
                } else {
                    match_info = Some((*key, prefix_len));
                    break;
                }
            }

            match match_info {
                Some((key, prefix_len)) => {
                    let child = curr.children.get_mut(&key).unwrap();

                    if prefix_len == child.ids.len() {
                        // move curr to child
                        curr = child;
                        curr.count += 1;
                        curr_idx += prefix_len;
                    } else {
                        // split child

                        // [child]->... => [child]->[new child]->...
                        let new_child = Node {
                            // to avoid clone: replace child.children with default value (empty vector) and return the original value
                            children: mem::take(&mut child.children),
                            ids: child.ids[prefix_len..].to_vec(),
                            count: child.count,
                        };

                        child.ids = child.ids[..prefix_len].to_vec();
                        child.children = HashMap::new();
                        child.children.insert(new_child.ids[0], new_child);

                        curr = child;
                        curr.count += 1;
                        curr_idx += prefix_len;
                    }
                }
                None => {
                    // create new child
                    let new_child = Node {
                        children: HashMap::new(),
                        ids: input_ids[curr_idx..].to_vec(),
                        count: 0,
                    };

                    let first_id = new_child.ids[0];
                    curr.children.insert(first_id, new_child);

                    curr = curr.children.get_mut(&first_id).unwrap();
                    curr.count += 1;
                    curr_idx = input_ids_len;
                }
            }
        }
    }

    pub fn prefix_match<'a>(&self, input_ids: &'a [usize]) -> &'a [usize] {
        let mut curr = &self.root;

        let mut curr_idx = 0;
        let input_ids_len = input_ids.len();

        while curr_idx < input_ids_len {
            let mut has_full_match = false;

            for key in curr.children.keys() {
                let prefix_len = common_prefix_len(&input_ids[curr_idx..], &curr.children[key].ids);

                if prefix_len == 0 {
                    continue;
                }

                let child = &curr.children[key];

                if prefix_len == child.ids.len() {
                    // full match
                    curr_idx += prefix_len;
                    curr = child;
                    has_full_match = true;

                    break;
                } else {
                    // partial match
                    curr_idx += prefix_len;
                    break;
                }
            }

            if !has_full_match {
                // if not full match, break
                break;
            }
        }

        &input_ids[..curr_idx]
    }

    pub fn delete(&mut self, input_ids: &[usize]) {
        let mut curr = &mut self.root;
        curr.count -= 1;

        let mut curr_idx = 0;
        let input_ids_len = input_ids.len();

        while curr_idx < input_ids_len {
            // First find the matching child index and prefix length
            let mut child_key = None;
            let mut prefix_len = 0;

            for key in curr.children.keys() {
                let current_prefix_len =
                    common_prefix_len(&input_ids[curr_idx..], &curr.children[key].ids);
                if current_prefix_len == curr.children[key].ids.len() {
                    child_key = Some(*key);
                    prefix_len = current_prefix_len;
                    break;
                }
            }

            match child_key {
                Some(key) => {
                    // Check count first
                    if curr.children[&key].count == 1 {
                        // If count will become 0, remove the child
                        let child = curr.children.get_mut(&key).unwrap();
                        child.count -= 1;
                        curr.children.remove(&key);
                        break;
                    } else {
                        // Otherwise decrement count and continue
                        let child = curr.children.get_mut(&key).unwrap();
                        child.count -= 1;
                        curr = child;
                        curr_idx += prefix_len;
                    }
                }
                None => panic!("No match found for {:?}", input_ids),
            }
        }
    }

    // for debug
    pub fn pretty_print(&self) {
        println!("RadixTree:");
        Self::print_node(&self.root, String::from(""));
    }

    fn print_node(node: &Node, prefix: String) {
        // Print current node info with "count" word
        println!("{}└── {:?} (count: {})", prefix, node.ids, node.count);

        // Print children with proper prefixes
        for (i, child) in node.children.values().enumerate() {
            let is_last = i == node.children.len() - 1;
            let child_prefix = if is_last {
                format!("{}    ", prefix) // Add space for last child
            } else {
                format!("{}│   ", prefix) // Add vertical line for other children
            };
            Self::print_node(child, child_prefix);
        }
    }
}
