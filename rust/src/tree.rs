use std::collections::HashMap;
use std::mem;

#[derive(Debug)]
pub struct Node {
    pub children: HashMap<u32, Node>, // the key is first id of the child because each child must have unique first id
    pub ids: Vec<u32>,
    pub count: u32,
}

#[derive(Debug)]
pub struct RadixTree {
    pub root: Node,
}

fn common_prefix_len(a: &[u32], b: &[u32]) -> usize {
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

    pub fn insert(&mut self, input_ids: &[u32]) {
        let mut curr = &mut self.root;
        curr.count += 1;

        let mut curr_idx = 0;
        let input_ids_len = input_ids.len();

        while curr_idx < input_ids_len {
            let first_id = &input_ids[curr_idx];
            // TODO: changing this get_mut causes error
            if curr.children.contains_key(first_id) {
                let child = curr.children.get_mut(first_id).unwrap();

                let prefix_len = common_prefix_len(&input_ids[curr_idx..], &child.ids);

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
            } else {
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

    pub fn prefix_match<'a>(&self, input_ids: &'a [u32]) -> &'a [u32] {
        let mut curr = &self.root;

        let mut curr_idx = 0;
        let input_ids_len = input_ids.len();

        while curr_idx < input_ids_len {
            match curr.children.get(&input_ids[curr_idx]) {
                Some(child) => {
                    let prefix_len = common_prefix_len(&input_ids[curr_idx..], &child.ids);

                    if prefix_len == child.ids.len() {
                        curr_idx += prefix_len;
                        curr = child;
                    } else {
                        curr_idx += prefix_len;
                        break;
                    }
                }
                None => {
                    break;
                }
            }
        }

        &input_ids[..curr_idx]
    }

    pub fn delete(&mut self, input_ids: &[u32]) {
        let mut curr = &mut self.root;
        curr.count -= 1;

        let mut curr_idx = 0;
        let input_ids_len = input_ids.len();

        while curr_idx < input_ids_len {
            let first_id = &input_ids[curr_idx];

            if curr.children.contains_key(first_id) {
                let child = curr.children.get(first_id).unwrap();

                let prefix_len = common_prefix_len(&input_ids[curr_idx..], &child.ids);

                if prefix_len == child.ids.len() {
                    if child.count == 1 {
                        // If count will become 0, remove the child
                        let child = curr.children.get_mut(first_id).unwrap();
                        child.count -= 1;
                        curr.children.remove(first_id);
                        break;
                    } else {
                        // Otherwise decrement count and continue
                        let child = curr.children.get_mut(first_id).unwrap();

                        child.count -= 1;
                        curr = child;
                        curr_idx += prefix_len;
                    }
                } else {
                    panic!("No match found for {:?}", input_ids);
                }
            } else {
                panic!("No match found for {:?}", input_ids);
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
