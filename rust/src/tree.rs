#[derive(Clone)]
pub struct Node {
    pub children: Vec<Node>,
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

impl RadixTree {
    pub fn new() -> Self {
        RadixTree {
            root: Node {
                children: Vec::new(),
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
            
            let mut has_match = false;

            for i in 0..curr.children.len() {
                let prefix_len = common_prefix_len(&input_ids[curr_idx..], &curr.children[i].ids);
                
                // TODO: moving below `let child =` does not work. Why?
                if prefix_len == 0 {
                    continue;
                }

                let child = &mut curr.children[i];
                
                if prefix_len == child.ids.len() {

                    // move to child
                    curr = child;
                    curr.count += 1;
                    curr_idx += prefix_len;

                    has_match = true;

                    break;
                } else {

                    // split child
                    let new_child = Node {
                        children: child.children.clone(), // move the owndership of child.children
                        ids: child.ids[prefix_len..].to_vec(),
                        count: child.count,
                    };

                    child.ids = child.ids[..prefix_len].to_vec();
                    child.children = vec![new_child];

                    curr = child;
                    curr.count += 1;
                    curr_idx += prefix_len;

                    has_match = true;

                    break;
                }

            }

            if !has_match {
                // create new child
                let new_child = Node {
                    children: Vec::new(),
                    ids: input_ids[curr_idx..].to_vec(),
                    count: 0,
                };

                curr.children.push(new_child);
                
                curr = curr.children.last_mut().unwrap();
                curr.count += 1;
                curr_idx = input_ids_len;
            }
            
        }
    }

    pub fn prefix_match<'a>(&self, input_ids: &'a [usize]) -> &'a [usize] {
        let mut curr = &self.root;

        let mut curr_idx = 0;
        let input_ids_len = input_ids.len();

        while curr_idx < input_ids_len {
            
            let mut has_full_match = false;

            for i in 0..curr.children.len() {
                let prefix_len = common_prefix_len(&input_ids[curr_idx..], &curr.children[i].ids);
                
                if prefix_len == 0 {
                    continue;
                }

                let child = &curr.children[i];
                
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

        return &input_ids[..curr_idx];

    }

    pub fn delete(&mut self, input_ids: &[usize]) {
        let mut curr = &mut self.root;
        curr.count -= 1;

        let mut curr_idx = 0;
        let input_ids_len = input_ids.len();

        while curr_idx < input_ids_len {
            // First find the matching child index and prefix length
            let mut child_idx = None;
            let mut prefix_len = 0;
            
            for i in 0..curr.children.len() {
                let current_prefix_len = common_prefix_len(&input_ids[curr_idx..], &curr.children[i].ids);
                if current_prefix_len == curr.children[i].ids.len() {
                    child_idx = Some(i);
                    prefix_len = current_prefix_len;
                    break;
                }
            }
    
            match child_idx {
                Some(i) => {
                    curr_idx += prefix_len;
                    
                    // Check count first
                    if curr.children[i].count == 1 {
                        // If count will become 0, remove the child
                        curr.children[i].count -= 1;
                        curr.children.remove(i);
                        
                        break;
                    } else {
                        // Otherwise decrement count and continue
                        let child = &mut curr.children[i];
                        child.count -= 1;
                        curr = child;
                    }
                }
                None => panic!("No match found for {:?}", input_ids)
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
        for (i, child) in node.children.iter().enumerate() {
            let is_last = i == node.children.len() - 1;
            let child_prefix = if is_last {
                format!("{}    ", prefix)  // Add space for last child
            } else {
                format!("{}│   ", prefix)  // Add vertical line for other children
            };
            Self::print_node(child, child_prefix);
        }
    }

}

