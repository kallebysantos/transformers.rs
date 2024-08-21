use core::f32;
use serde::{Deserialize, Serialize};
use serde_json::{Map, Value};
use std::{fs::File, io::BufReader, path::Path};

use ndarray::{Array1, Axis, Ix3, Zip};
use ort::{inputs, ArrayExtensions, GraphOptimizationLevel, Result, Session};
use tokenizers::Tokenizer;

use crate::argmax;

#[derive(Serialize, Deserialize, Debug, PartialEq, Eq)]
pub enum TokenClassificationAggregationStrategy {
    None,
    Simple,
}

#[derive(Deserialize, Debug)]
#[serde(default)]
pub struct TokenClassificationOptions {
    pub aggregation_strategy: TokenClassificationAggregationStrategy,
    pub ignore_labels: Vec<String>,
}
impl Default for TokenClassificationOptions {
    fn default() -> Self {
        Self {
            aggregation_strategy: TokenClassificationAggregationStrategy::None,
            ignore_labels: vec![String::from("O")],
        }
    }
}

#[derive(Debug, Clone, Serialize)]
pub struct RawClassifiedToken {
    pub entity: String,
    pub score: f32,
    pub index: usize,
    pub word: String,
    pub start: usize,
    pub end: usize,
}

#[derive(Debug, Clone, Serialize)]
pub struct ClassifiedTokenGroup {
    pub group_entity: String,
    pub score: f32,
    pub word: String,
    pub start: usize,
    pub end: usize,
    pub tokens: Vec<RawClassifiedToken>,
}

#[derive(Debug, Clone, Serialize)]
#[serde(untagged)]
pub enum TokenClassificationOutput {
    Raw(Vec<RawClassifiedToken>),
    Grouped(Vec<ClassifiedTokenGroup>),
}

pub fn token_classification(
    input: String,
    options: Option<TokenClassificationOptions>,
) -> Result<TokenClassificationOutput> {
    let options = options.unwrap_or_default();

    let base_path = "/home/kalleby/my-projects/rust/machine-lerning/transformers-rs/temp/models/token_classification/";
    let model_path = Path::new(&base_path).join("model.onnx");
    let config_path = Path::new(&base_path).join("config.json");
    let tokenizer_path = Path::new(&base_path).join("tokenizer.json");

    let config = BufReader::new(File::open(config_path).unwrap());
    let config = serde_json::from_reader::<_, Map<String, Value>>(config).unwrap();

    let model = Session::builder()?
        .with_optimization_level(GraphOptimizationLevel::Level3)?
        .with_intra_threads(4)?
        .commit_from_file(model_path)?;

    let tokenizer = Tokenizer::from_file(tokenizer_path).unwrap();

    let encodings = tokenizer.encode(input.to_owned(), true).unwrap();

    let input_ids = encodings
        .get_ids()
        .iter()
        .map(|v| i64::from(*v))
        .collect::<Vec<_>>();

    let attention_mask = encodings
        .get_attention_mask()
        .iter()
        .map(|v| i64::from(*v))
        .collect::<Vec<_>>();

    let token_type_ids = encodings
        .get_type_ids()
        .iter()
        .map(|v| i64::from(*v))
        .collect::<Vec<_>>();

    let input_ids_array = Array1::from_vec(input_ids.to_owned()).insert_axis(Axis(0));
    let attention_mask_array = Array1::from_vec(attention_mask.to_owned()).insert_axis(Axis(0));
    let token_type_ids_array = Array1::from_vec(token_type_ids.to_owned()).insert_axis(Axis(0));

    let outputs = model.run(inputs! {
        "input_ids" => input_ids_array,
        "token_type_ids" => token_type_ids_array,
        "attention_mask" => attention_mask_array,
    }?)?;

    let outputs = outputs["logits"].try_extract_tensor::<f32>()?;
    let outputs = outputs.into_dimensionality::<Ix3>().unwrap();
    let outputs = outputs.softmax(Axis(2));

    let labels = config.get("id2label").unwrap();
    let vocab = tokenizer.get_added_vocabulary();

    let tokens = encodings.get_tokens();
    let offsets = encodings.get_offsets();

    let outputs = Zip::from(outputs.lanes(Axis(2))).map_collect(argmax);

    let outputs = outputs.map_axis(Axis(1), |row| {
        let classifications = row.iter().enumerate().filter_map(|(idx, predict)| {
            let token = tokens.get(idx).unwrap();

            if vocab.is_special_token(token) {
                return None;
            }

            let entity = labels
                .get(predict.0.to_string())
                .map(|label| label.as_str().unwrap().into())
                .unwrap();

            let (start, end) = offsets.get(idx).unwrap();

            Some(RawClassifiedToken {
                index: idx,
                score: predict.1,
                entity,
                word: token.to_owned(),
                start: *start,
                end: *end,
            })
        });

        match options.aggregation_strategy {
            TokenClassificationAggregationStrategy::None => {
                TokenClassificationOutput::Raw(none_aggregation(classifications))
            }
            TokenClassificationAggregationStrategy::Simple => TokenClassificationOutput::Grouped(
                simple_aggregation(classifications, &input, &options.ignore_labels),
            ),
        }
    });

    Ok(outputs.first().unwrap().to_owned())
}

fn none_aggregation(iter: impl IntoIterator<Item = RawClassifiedToken>) -> Vec<RawClassifiedToken> {
    iter.into_iter().collect()
}

fn simple_aggregation(
    iter: impl IntoIterator<Item = RawClassifiedToken>,
    original_text: &str,
    ignore_labels: &[String],
) -> Vec<ClassifiedTokenGroup> {
    let mut iter = iter.into_iter().peekable();
    let mut result = vec![];

    loop {
        let Some(item) = iter.next() else {
            break;
        };

        let Some((_, label)) = item.entity.split_at_checked(2) else {
            continue;
        };

        if ignore_labels.iter().any(|to_ignore| to_ignore == label) {
            continue;
        }

        let child_label = format!("I-{label}");

        // Peeking take_while()
        let mut group = vec![item.to_owned()];
        loop {
            let Some(child) = iter.next_if(|other| other.entity == child_label) else {
                break;
            };

            group.push(child)
        }

        let start = item.start;
        let end = group.last().map_or(item.end, |last| last.end);
        let word = original_text.get(start..end).unwrap();

        // Apply mean score
        let score = group.iter().map(|i| i.score).sum::<f32>() / group.len() as f32;

        result.push(ClassifiedTokenGroup {
            group_entity: label.to_owned(),
            tokens: group.to_owned(),
            score,
            word: word.to_owned(),
            start,
            end,
        })
    }

    result
}
