use std::rc::Rc;
use std::cell::RefCell;
use std::collections::HashMap;
use std::time::{Duration, SystemTime, UNIX_EPOCH};

type NodeRef = Rc<Node>;

#[derive(Debug)]
struct Node {
    children: RefCell<HashMap<char, NodeRef>>,
    text: RefCell<String>, // internal mutability
    tenant_last_access_time: RefCell<HashMap<String, u128>>,
    parent: RefCell<Option<NodeRef>>,
}

#[derive(Debug)]
pub struct Tree {
    root: NodeRef,
}

fn shared_prefix_length(a: &str, b: &str) -> usize {
    let mut i = 0;
    let mut a_iter = a.chars();
    let mut b_iter = b.chars();

    loop {
        match (a_iter.next(), b_iter.next()) {
            (Some(a_char), Some(b_char)) if a_char == b_char => {
                i += 1;
            }
            _ => break,
        }
    }

    return i;
}

fn slice_by_chars(s: &str, start: usize, end: usize) -> String {
    s.chars()
        .skip(start)
        .take(end - start)
        .collect()
}


impl Tree {
    pub fn new() -> Self {
        Tree {
            root: Rc::new(
                Node {
                    children: RefCell::new(HashMap::new()),
                    text: RefCell::new("".to_string()),
                    tenant_last_access_time: RefCell::new(HashMap::new()),
                    parent: RefCell::new(None),
                }
            )
        }
    }

    pub fn insert(&self, text: &str, tenant: &str) {
        let mut curr = Rc::clone(&self.root);
        let mut curr_idx = 0;

        let timestamp_ms = SystemTime::now()
            .duration_since(UNIX_EPOCH)
            .unwrap()
            .as_millis();

        curr.tenant_last_access_time.borrow_mut().insert(tenant.to_string(), timestamp_ms);

        let mut prev = Rc::clone(&self.root);
        let text_count = text.chars().count();

        while curr_idx < text_count {
            let first_char = text.chars().nth(curr_idx).unwrap();

            curr = prev.clone();

            if curr.children.borrow().contains_key(&first_char) {
                // match
                let mut children_mut = curr.children.borrow_mut();
                let matched_node = children_mut.get_mut(&first_char).unwrap().clone();
                let matched_node_text = matched_node.text.borrow().to_string();
                let matched_node_text_count = matched_node_text.chars().count();

                let curr_text = slice_by_chars(text, curr_idx, text_count);

                let shared_len = shared_prefix_length(&curr_text, &matched_node_text);

                if shared_len < matched_node_text_count {
                    // split the matched node
                    // Split structure: [curr] -> [matched_node] =>
                    //                  [curr] -> [new_node] -> [contracted_matched_node]
                    let matched_text = slice_by_chars(&matched_node_text, 0, shared_len);
                    let new_text = slice_by_chars(&matched_node_text, shared_len, matched_node_text_count);

                    let new_node = Rc::new(Node {
                        text: RefCell::new(matched_text),
                        children: RefCell::new(HashMap::new()),
                        parent: RefCell::new(Some(Rc::clone(&curr))),
                        tenant_last_access_time: RefCell::new(matched_node.tenant_last_access_time.borrow().clone()),
                    });

                    let first_new_char = new_text.chars().nth(0).unwrap();
                    new_node.children.borrow_mut().insert(first_new_char, Rc::clone(&matched_node));


                    children_mut.insert(first_char, Rc::clone(&new_node));

                    let mut text_mut = matched_node.text.borrow_mut();
                    *text_mut = new_text;

                    let mut parent_mut = matched_node.parent.borrow_mut();
                    *parent_mut = Some(Rc::clone(&new_node));

                    prev = Rc::clone(&new_node);
                    prev.tenant_last_access_time.borrow_mut().insert(tenant.to_string(), timestamp_ms);
                    curr_idx += shared_len;

                } else {
                    // move to the next node
                    prev = Rc::clone(&matched_node);
                    prev.tenant_last_access_time.borrow_mut().insert(tenant.to_string(), timestamp_ms);
                    curr_idx += matched_node_text_count;
                }

            } else {
                // no match
                let curr_text = slice_by_chars(text, curr_idx, text_count);

                let new_node = Rc::new(Node {
                    children: RefCell::new(HashMap::new()),
                    text: RefCell::new(curr_text),
                    tenant_last_access_time: RefCell::new(HashMap::new()),
                    parent: RefCell::new(Some(Rc::clone(&curr))),
                });

                curr.children.borrow_mut().insert(first_char, Rc::clone(&new_node));

                prev = Rc::clone(&new_node);
                prev.tenant_last_access_time.borrow_mut().insert(tenant.to_string(), timestamp_ms);
                curr_idx = text_count;
            }

        }
    }

    pub fn prefix_match(&self, text: &str) -> (String, String) {
        let mut curr = Rc::clone(&self.root);
        let mut curr_idx = 0;
        let mut prev = Rc::clone(&self.root);

        let text_count = text.chars().count();

        while curr_idx < text_count {
            let first_char = text.chars().nth(curr_idx).unwrap();

            curr = prev.clone();

            let curr_text = slice_by_chars(text, curr_idx, text_count);

            if curr.children.borrow().contains_key(&first_char) {
                let shared_len = shared_prefix_length(&curr_text, &curr.children.borrow().get(&first_char).unwrap().text.borrow());

                let matched_node = curr.children.borrow().get(&first_char).unwrap().clone();
                let matched_node_text_count = matched_node.text.borrow().chars().count();

                if shared_len == matched_node_text_count {
                    curr_idx += shared_len;
                    prev = Rc::clone(&matched_node);
                } else {
                    curr_idx += shared_len;
                    prev = Rc::clone(&matched_node);
                    break
                }
            } else {
                break;
            }
        }

        curr = prev.clone();

        // randomly select the first tenant (key in the map)
        let tenant = curr.tenant_last_access_time.borrow().keys().next().unwrap_or(&String::from("empty")).to_string();

        let ret_text = slice_by_chars(text, 0, curr_idx);
        return (ret_text, tenant);
    }


    fn node_to_string(node: &NodeRef, prefix: &str, is_last: bool) -> String {
        let mut result = String::new();

        // Add prefix and branch character
        result.push_str(prefix);
        result.push_str(if is_last { "└── " } else { "├── " });

        // Add node text
        let node_text = node.text.borrow();
        result.push_str(&format!("'{}' [", node_text));

        // Add tenant information with timestamps
        let mut tenant_info = Vec::new();
        for (tenant_id, timestamp_ms) in node.tenant_last_access_time.borrow().iter() {
            // Convert milliseconds to seconds and remaining milliseconds
            let seconds = (timestamp_ms / 1000) as u64;
            let millis = (timestamp_ms % 1000) as u32;

            // Create SystemTime from Unix timestamp
            let system_time = UNIX_EPOCH + Duration::from_secs(seconds);

            // Format time as HH:MM:SS.mmm
            let datetime = system_time.duration_since(UNIX_EPOCH).unwrap();
            let hours = (datetime.as_secs() % 86400) / 3600;
            let minutes = (datetime.as_secs() % 3600) / 60;
            let seconds = datetime.as_secs() % 60;

            tenant_info.push(format!("{} | {:02}:{:02}:{:02}.{:03}",
                tenant_id, hours, minutes, seconds, millis));
        }

        result.push_str(&tenant_info.join(", "));
        result.push_str("]\n");

        // Process children
        let children: Vec<_> = node.children.borrow().iter()
            .map(|(k, v)| (k.clone(), Rc::clone(v)))
            .collect();
        let child_count = children.len();

        for (i, (_, child)) in children.iter().enumerate() {
            let is_last_child = i == child_count - 1;
            let new_prefix = format!("{}{}", prefix, if is_last { "    " } else { "│   " });

            result.push_str(&Tree::node_to_string(
                child,
                &new_prefix,
                is_last_child
            ));
        }

        result
    }

    pub fn pretty_print(&self) {
        if self.root.children.borrow().is_empty() {
            return;
        }

        let mut result = String::new();
        let children: Vec<_> = self.root.children.borrow().iter()
            .map(|(k, v)| (k.clone(), Rc::clone(v)))
            .collect();
        let child_count = children.len();

        for (i, (_, child)) in children.iter().enumerate() {
            let is_last = i == child_count - 1;
            result.push_str(&Tree::node_to_string(
                child,
                "",
                is_last
            ));
        }

        println!("{result}");
    }
}




// #[cfg(test)]
// mod tests {
//     use super::*;

//     #[test]
//     fn test_cold_start() {
//         let tree = Tree::new();
//         let (matched_text, tenant) = tree.prefix_match("hello");
//         assert_eq!(matched_text, "");
//         assert_eq!(tenant, "empty");
//     }

//     #[test]
//     fn test_exact_match_seq() {
//         let tree = Tree::new();
//         tree.insert("hello", "tenant1");
//         tree.pretty_print();
//         tree.insert("apple", "tenant2");
//         tree.pretty_print();
//         tree.insert("banana", "tenant3");
//         tree.pretty_print();

//         let (matched_text, tenant) = tree.prefix_match("hello");
//         assert_eq!(matched_text, "hello");
//         assert_eq!(tenant, "tenant1");

//         let (matched_text, tenant) = tree.prefix_match("apple");
//         assert_eq!(matched_text, "apple");
//         assert_eq!(tenant, "tenant2");

//         let (matched_text, tenant) = tree.prefix_match("banana");
//         assert_eq!(matched_text, "banana");
//         assert_eq!(tenant, "tenant3");
//     }

//     #[test]
//     fn test_partial_match() {
//         let tree = Tree::new();
//         tree.insert("apple", "tenant0");
//         tree.insert("apabc", "tenant0");
//         tree.insert("acbdeds", "tenant0");

//         let (matched_text, matched_tenant) = tree.prefix_match("apple");
//         assert_eq!(matched_text, "apple");
//         assert_eq!(matched_tenant, "tenant0");

//         let (matched_text, matched_tenant) = tree.prefix_match("apabc");
//         assert_eq!(matched_text, "apabc");
//         assert_eq!(matched_tenant, "tenant0");

//         let (matched_text, matched_tenant) = tree.prefix_match("acbdeds");
//         assert_eq!(matched_text, "acbdeds");
//         assert_eq!(matched_tenant, "tenant0");
//     }

//     #[test]
//     fn test_group_prefix_insert_match() {
//         let prefix = vec![
//             "Clock strikes midnight, I'm still wide awake",
//             "Got dreams bigger than these city lights",
//             "Time waits for no one, gotta make my move",
//             "Started from the bottom, that's no metaphor"
//         ];
//         let suffix = vec![
//             "Got too much to prove, ain't got time to lose",
//             "History in the making, yeah, you can't erase this"
//         ];
//         let tree = Tree::new();

//         for i in 0..prefix.len() {
//             for j in 0..suffix.len() {
//                 let text = format!("{} {}", prefix[i], suffix[j]);
//                 let tenant = format!("tenant{}", i);
//                 tree.insert(&text, &tenant);
//             }
//         }

//         tree.pretty_print();

//         for i in 0..prefix.len() {
//             let (matched_text, matched_tenant) = tree.prefix_match(prefix[i]);
//             let tenant = format!("tenant{}", i);
//             assert_eq!(matched_text, prefix[i]);
//             assert_eq!(matched_tenant, tenant);
//         }
//     }

//     #[test]
//     fn test_utf8_split() {
//         let tree = Tree::new();
//         let test_pairs = vec![
//             ("你好嗎", "tenant1"),
//             ("你好喔", "tenant2"),
//             ("你心情好嗎", "tenant3"),
//         ];

//         // Insert sequentially
//         for (text, tenant) in &test_pairs {
//             tree.insert(text, tenant);
//         }

//         tree.pretty_print();

//         // Test sequentially
//         for (text, tenant) in &test_pairs {
//             let (matched_text, matched_tenant) = tree.prefix_match(text);
//             assert_eq!(matched_text, *text);
//             assert_eq!(matched_tenant, *tenant);
//         }
//     }
// }
