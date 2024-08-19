use serde::Deserialize;
use std::path::Path;

pub use ndarray;
use ndarray::{Array2, Ix3};

pub use ort;
use ort::{inputs, GraphOptimizationLevel, Result, Session};

pub use tokenizers;
use tokenizers::{pad_encodings, PaddingParams, Tokenizer};

#[derive(Deserialize, Debug)]
#[serde(untagged)]
pub enum FeatureExtractionInput {
    Single(String),
    Batch(Vec<String>),
}

#[derive(Deserialize, Debug)]
#[serde(default)]
pub struct FeatureExtractionOptions {
    pub mean_pool: bool,
    pub normalize: bool,
}
impl Default for FeatureExtractionOptions {
    fn default() -> Self {
        Self {
            mean_pool: true,
            normalize: true,
        }
    }
}

type FeatureExtractionResult = Result<Vec<Vec<f32>>>;

pub fn feature_extraction(
    input: FeatureExtractionInput,
    _options: Option<FeatureExtractionOptions>,
) -> FeatureExtractionResult {
    let base_path = "/home/kalleby/my-projects/rust/machine-lerning/transformers-rs/temp/models/feature_extraction/";
    let model_path = Path::new(&base_path).join("model.onnx");
    let tokenizer_path = Path::new(&base_path).join("tokenizer.json");

    let model = Session::builder()?
        .with_optimization_level(GraphOptimizationLevel::Level3)?
        .with_intra_threads(4)?
        .commit_from_file(model_path)?;

    let tokenizer = Tokenizer::from_file(tokenizer_path).unwrap();

    let input = match input {
        FeatureExtractionInput::Single(value) => vec![value],
        FeatureExtractionInput::Batch(values) => values,
    }; // Encode our input strings. `encode_batch` will pad each input to be the same length.

    let mut encodings = tokenizer.encode_batch(input.to_owned(), true).unwrap();

    // We use it instead of overriding the Tokenizer
    pad_encodings(encodings.as_mut_slice(), &PaddingParams::default()).unwrap();

    let padded_token_length = encodings.get(0).unwrap().len();
    let input_shape = [input.len(), padded_token_length];

    println!("S: {:?}\n", model);
    println!("E: {:?}\n", encodings);

    let input_ids = encodings
        .iter()
        .flat_map(|e| e.get_ids().iter().map(|v| i64::from(*v)))
        .collect::<Vec<_>>();

    let attention_mask = encodings
        .iter()
        .flat_map(|e| e.get_attention_mask().iter().map(|v| i64::from(*v)))
        .collect::<Vec<_>>();

    let token_type_ids = encodings
        .iter()
        .flat_map(|e| e.get_type_ids().iter().map(|v| i64::from(*v)))
        .collect::<Vec<_>>();

    let input_ids_array = Array2::from_shape_vec(input_shape, input_ids.to_owned()).unwrap();
    let attention_mask_array = Array2::from_shape_vec(input_shape, attention_mask).unwrap();
    let token_type_ids_array = Array2::from_shape_vec(input_shape, token_type_ids).unwrap();

    let outputs = model.run(inputs! {
        "input_ids" => input_ids_array,
        "token_type_ids" => token_type_ids_array,
        "attention_mask" => attention_mask_array.to_owned(),
    }?)?;

    let embeddings = outputs["last_hidden_state"].try_extract_tensor::<f32>()?;
    let embeddings = embeddings.into_dimensionality::<Ix3>().unwrap();

    println!("{:?}", embeddings);

    /*
        let options = options.unwrap_or(&FeatureExtractionPipelineInputOptions {
            mean_pool: true,
            normalize: true,
        });

        let result = if options.mean_pool {
            mean_pool(embeddings, attention_mask_array.view().insert_axis(Axis(2)))
        } else {
            embeddings.into_owned().remove_axis(Axis(0))
        };

        let result = if options.normalize {
            let (normalized, _) = normalize(result, NormalizeAxis::Row);
            normalized
        } else {
            result
        };
    */

    Ok(vec![])
}

#[cfg(test)]
mod tests {}
