use serde_json::{Map, Value};
use std::{fs::File, io::BufReader, path::Path};

pub use ndarray;
use ndarray::{Array1, Axis};

pub use ort;
use ort::{inputs, ArrayExtensions, GraphOptimizationLevel, Result, Session};

pub use tokenizers;
use tokenizers::Tokenizer;

pub fn add(left: usize, right: usize) -> usize {
    left + right
}

#[derive(Debug, Clone)]
pub struct SentimentAnalysisOutput {
    pub label: String,
    pub score: f32,
}

type SentimentAnalysisResult = Result<(SentimentAnalysisOutput, SentimentAnalysisOutput)>;

pub fn sentiment_analysis(inputs: String) -> SentimentAnalysisResult {
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
    let encoded = tokenizer.encode(inputs, true).unwrap();

    // TODO: Encode using batch, Accept multiple inputs

    let input_ids = encoded
        .get_ids()
        .iter()
        .map(|v| i64::from(*v))
        .collect::<Vec<_>>();

    let attention_mask = encoded
        .get_attention_mask()
        .iter()
        .map(|v| i64::from(*v))
        .collect::<Vec<_>>();

    // Sentiment Analysis Shape [-1,-1] = Array2
    let input_tensors = inputs! {
        "input_ids" =>  Array1::from_vec(input_ids).insert_axis(Axis(0)),
        "attention_mask" => Array1::from_vec(attention_mask).insert_axis(Axis(0))
    }?;

    let outputs = model.run(input_tensors)?;
    let outputs = outputs.get("logits").unwrap().try_extract_tensor::<f32>()?;
    let outputs = outputs.softmax(Axis(1));

    let labels = config.get("id2label").unwrap();

    let mut results = vec![];

    for row in outputs.rows() {
        let item0 = SentimentAnalysisOutput {
            label: labels.get("0").unwrap().to_string(),
            score: *row.get(0).unwrap(),
        };
        let item1 = SentimentAnalysisOutput {
            label: labels.get("1").unwrap().to_string(),
            score: *row.get(1).unwrap(),
        };

        results.push((item0, item1));
    }

    Ok(results.first().unwrap().to_owned())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn it_works() {
        let result = add(2, 2);
        assert_eq!(result, 4);
    }
}
