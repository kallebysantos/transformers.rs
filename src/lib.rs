use std::{
    fs::{self, File},
    io::BufReader,
    path::Path,
};

use ndarray::{Array1, ArrayView1, Axis, IntoNdProducer, RemoveAxis};
pub use ort;
use ort::{inputs, ArrayExtensions, GraphOptimizationLevel, Result, Session};

use serde_json::{Map, Value};
pub use tokenizers;
use tokenizers::Tokenizer;

pub fn add(left: usize, right: usize) -> usize {
    left + right
}

#[derive(Debug)]
pub struct SentimentAnalysisOutput {
    pub label: String,
    pub score: f32,
}

pub fn sentiment_analysis(inputs: Vec<String>) -> Result<Vec<Vec<SentimentAnalysisOutput>>> {
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

    println!("{:?}", model.inputs);
    println!("{:?}", model.outputs);

    let tokenizer = Tokenizer::from_file(tokenizer_path).unwrap();
    let encoded = tokenizer.encode(inputs, true).unwrap();

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

    // println!("encoded: {:?}", encoded);
    // println!("input_ids: {:?}", input_ids);
    // println!("attention_mask: {:?}", attention_mask);

    // Sentiment Analysis Shape [-1,-1] = Array2
    let input_tensors = inputs! {
        "input_ids" =>  Array1::from_vec(input_ids).insert_axis(Axis(0)),
        "attention_mask" => Array1::from_vec(attention_mask).insert_axis(Axis(0))
    }?;

    let outputs = model.run(input_tensors)?;
    let outputs = outputs.get("logits").unwrap().try_extract_tensor::<f32>()?;
    let outputs = outputs.softmax(Axis(1));

    let labels = config.get("id2label").unwrap();
    // println!("lables: {:?}", labels);
    // println!("lables: {:?}", labels.get("0"));

    let outputs = outputs
        .rows()
        .into_producer()
        .into_iter()
        .map(|row| {
            row.into_iter()
                .enumerate()
                // .inspect(|tu| println!("{:?}", tu))
                .map(|(i, val)| SentimentAnalysisOutput {
                    label: labels
                        .get(i.to_string())
                        .unwrap()
                        .as_str()
                        .unwrap()
                        .to_owned(),
                    //label: "TESTE".to_owned(),
                    score: *val,
                })
                .collect::<Vec<_>>()
        })
        .collect::<Vec<_>>();

    println!("outputs: {:?}", outputs);
    // let outputs = outputs["last_hidden_state"];
    Ok(outputs)
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
