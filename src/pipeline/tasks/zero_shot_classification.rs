use serde::{Deserialize, Serialize};
use std::{iter, path::Path};

use ndarray::{Array2, Axis};
use ort::{inputs, ArrayExtensions, GraphOptimizationLevel, Result, Session};
use tokenizers::{pad_encodings, PaddingParams, Tokenizer};

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ZeroShotClassificationInput(pub String, pub Vec<String>);

#[derive(Debug, Clone)]
pub struct ZeroShotClassificationOutput {
    pub label: String,
    pub score: f32,
}

#[derive(Deserialize, Debug)]
#[serde(default)]
pub struct ZeroShotClassificationOptions {
    pub multi_label: bool,
    pub hypothesis_template: String,
}
impl Default for ZeroShotClassificationOptions {
    fn default() -> Self {
        Self {
            hypothesis_template: String::from("This example is {}."),
            multi_label: false,
        }
    }
}

type ZeroShotClassificationResult = Result<Vec<ZeroShotClassificationOutput>>;

pub fn zero_shot_classification(
    input: ZeroShotClassificationInput,
    options: Option<ZeroShotClassificationOptions>,
) -> ZeroShotClassificationResult {
    let ZeroShotClassificationInput(input, labels) = input;
    let options = options.unwrap_or_default();

    let base_path = "/home/kalleby/my-projects/rust/machine-lerning/transformers-rs/temp/models/zero_shot_classification/";
    let model_path = Path::new(&base_path).join("model.onnx");
    let tokenizer_path = Path::new(&base_path).join("tokenizer.json");

    let model = Session::builder()?
        .with_optimization_level(GraphOptimizationLevel::Level3)?
        .with_intra_threads(4)?
        .commit_from_file(model_path)?;

    let tokenizer = Tokenizer::from_file(tokenizer_path).unwrap();

    // Pose sequence as NLI premise and label as a hypothesis
    let input_values = iter::repeat(input)
        .zip(
            labels
                .iter()
                .map(|label| options.hypothesis_template.replace("{}", label)),
        )
        .collect::<Vec<_>>();

    let mut encodings = tokenizer.encode_batch(input_values, true)?;
    pad_encodings(encodings.as_mut_slice(), &PaddingParams::default()).unwrap();

    let padded_token_length = encodings.get(0).unwrap().len();
    let input_shape = [labels.len(), padded_token_length];

    let input_ids = encodings
        .iter()
        .flat_map(|e| e.get_ids().iter().map(|v| i64::from(*v)))
        .collect::<Vec<_>>();

    let attention_mask = encodings
        .iter()
        .flat_map(|e| e.get_attention_mask().iter().map(|v| i64::from(*v)))
        .collect::<Vec<_>>();

    let input_ids_array = Array2::from_shape_vec(input_shape, input_ids.to_owned()).unwrap();
    let attention_mask_array = Array2::from_shape_vec(input_shape, attention_mask).unwrap();

    let input_tensors = inputs! {
        "input_ids" =>  input_ids_array ,
        "attention_mask" => attention_mask_array
    }?;

    let outputs = model.run(input_tensors)?;
    let outputs = outputs.get("logits").unwrap().try_extract_tensor::<f32>()?;
    let mut outputs = outputs.into_owned();

    // We throw away "neutral" (dim 1) and take the probability of
    // "entailment" (2) as the probability of the label being true
    outputs.remove_index(Axis(1), 1);

    let predicts = if options.multi_label {
        // Softmax over the Entailment vs. Contradiction dim for each label independently
        outputs.softmax(Axis(1)).index_axis(Axis(1), 1).into_owned()
    } else {
        // Softmax the "entailment" logits over all candidate labels
        outputs.index_axis(Axis(1), 1).softmax(Axis(0))
    };

    let mut results = predicts
        .iter()
        .enumerate()
        .map(|(idx, score)| ZeroShotClassificationOutput {
            label: labels.get(idx).unwrap().to_owned(),
            score: score.to_owned().into(),
        })
        .collect::<Vec<_>>();

    // Consider using `BinaryHeap<_>` while `collect` instead.
    results.sort_by(|item, other| item.score.total_cmp(&other.score).reverse());

    Ok(results)
}

#[cfg(test)]
mod tests {}
