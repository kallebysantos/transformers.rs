use serde::{Deserialize, Serialize};
use serde_json::{Map, Value};
use std::{fs::File, io::BufReader, path::Path};

pub use ndarray;
use ndarray::{Array2, Axis};

pub use ort;
use ort::{inputs, ArrayExtensions, GraphOptimizationLevel, Result, Session};

pub use tokenizers;
use tokenizers::{pad_encodings, PaddingParams, Tokenizer};

#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(untagged)]
pub enum SentimentAnalysisInput {
    Single(String),
    Batch(Vec<String>),
}

#[derive(Debug, Clone)]
pub struct SentimentAnalysisOutput {
    pub label: String,
    pub score: f32,
}

type SentimentAnalysisResult = Result<Vec<SentimentAnalysisOutput>>;

pub fn sentiment_analysis(input: SentimentAnalysisInput) -> SentimentAnalysisResult {
    let base_path = "/home/kalleby/my-projects/rust/machine-lerning/transformers-rs/temp/models/sentiment_analysis/";
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

    let input = match input {
        SentimentAnalysisInput::Single(value) => vec![value],
        SentimentAnalysisInput::Batch(values) => values,
    };

    let mut encodings = tokenizer.encode_batch(input.to_owned(), false)?;

    // We use it instead of overriding the Tokenizer
    pad_encodings(encodings.as_mut_slice(), &PaddingParams::default())?;

    let padded_token_length = encodings.get(0).unwrap().len();

    let input_ids = encodings
        .iter()
        .flat_map(|e| e.get_ids().iter().map(|v| i64::from(*v)))
        .collect::<Vec<_>>();

    let attention_mask = encodings
        .iter()
        .flat_map(|e| e.get_ids().iter().map(|v| i64::from(*v)))
        .collect::<Vec<_>>();

    // Sentiment Analysis Shape [-1,-1] = Array2
    let input_tensors = inputs! {
        "input_ids" =>  Array2::from_shape_vec([input.len(), padded_token_length], input_ids).unwrap(),
        "attention_mask" => Array2::from_shape_vec([input.len(), padded_token_length], attention_mask).unwrap()
    }?;

    let outputs = model.run(input_tensors)?;
    let outputs = outputs.get("logits").unwrap().try_extract_tensor::<f32>()?;
    let outputs = outputs.softmax(Axis(1));

    let labels = config.get("id2label").unwrap();

    let mut results = vec![];
    for row in outputs.rows() {
        if let Some((label, score)) = row
            .iter()
            .enumerate()
            .max_by(|a, b| a.1.partial_cmp(b.1).unwrap())
        {
            results.push(SentimentAnalysisOutput {
                label: labels
                    .get(label.to_string())
                    .map(|label| label.as_str().unwrap().into())
                    .unwrap(),
                score: *score,
            })
        }
    }

    Ok(results)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn predict_inputs() {
        let input = SentimentAnalysisInput::Batch(vec![
            String::from("I've been waiting for a HuggingFace course my whole life."),
            String::from("I hate this so much!"),
        ]);
        let result = sentiment_analysis(input.to_owned()).unwrap();

        assert_eq!(2, result.len());
        assert_eq!("POSITIVE", result[0].label);
        assert_eq!("NEGATIVE", result[1].label);

        let input = SentimentAnalysisInput::Single(String::from("I love candies"));
        let result = sentiment_analysis(input.to_owned()).unwrap();

        assert_eq!(1, result.len());
        assert_eq!("POSITIVE", result[0].label);
    }
}
