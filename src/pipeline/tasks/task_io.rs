use serde::{Deserialize, Serialize};

#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(untagged)]
pub enum TaskIO<T> {
    Single(T),
    Batch(Vec<T>),
}
