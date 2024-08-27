use serde::{Deserialize, Serialize};
use serde_json::{Map, Value};
use std::{
    fs::File,
    io::{stdout, BufReader, Stdout, Write},
    path::Path,
    usize,
};

pub use ndarray;
use ndarray::{array, concatenate, s, Array1, Array2, Axis, Ix3, Zip};

pub use ort;
use ort::{inputs, ArrayExtensions, GraphOptimizationLevel, Result, Session};

pub use tokenizers;
use tokenizers::{pad_encodings, PaddingParams, Tokenizer};

use crate::argmax;

// Seq2Seq Models: Encoder -> Decoder Architecture
pub fn seq_to_seq(input: String) -> Result<()> {
    println!("INPUT: {input}");
    let base_path =
        "/home/kalleby/my-projects/rust/machine-lerning/transformers-rs/temp/models/seq_to_seq/t5";

    let encoder_path = Path::new(&base_path).join("encoder.onnx");
    let decoder_path = Path::new(&base_path).join("decoder.onnx");

    let config_path = Path::new(&base_path).join("config.json");
    let tokenizer_path = Path::new(&base_path).join("tokenizer.json");

    let config = BufReader::new(File::open(config_path).unwrap());
    // let config = serde_json::from_reader::<_, Map<String, Value>>(config).unwrap();

    let encoder_model = Session::builder()?
        .with_optimization_level(GraphOptimizationLevel::Level3)?
        .with_intra_threads(4)?
        .commit_from_file(encoder_path)?;

    // println!("Encoder: {encoder_model:?}");

    let decoder_model = Session::builder()?
        .with_optimization_level(GraphOptimizationLevel::Level3)?
        .with_intra_threads(4)?
        .commit_from_file(decoder_path)?;

    // println!("\nDecoder: {decoder_model:?}");

    let tokenizer = Tokenizer::from_file(tokenizer_path).unwrap();
    // println!("\nTokenizer: {tokenizer:?}");

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

    let input_ids_array = Array1::from_vec(input_ids.to_owned()).insert_axis(Axis(0));
    let attention_mask_array = Array1::from_vec(attention_mask.to_owned()).insert_axis(Axis(0));

    let encoder_outputs = encoder_model.run(inputs! {
        "input_ids" => input_ids_array,
        "attention_mask" => attention_mask_array.to_owned(),

    }?)?;

    // println!("\nEncoder Outputs: {encoder_outputs:?}");

    let encoder_outputs = encoder_outputs["last_hidden_state"].try_extract_tensor::<f32>()?;
    let encoder_outputs = encoder_outputs.into_dimensionality::<Ix3>().unwrap();

    // Populate with START_TOKEN
    // Get from generation_config.json -> "decoder_start_token"
    let mut decoder_input_ids = Array1::from_vec(vec![i64::from(0)]).insert_axis(Axis(0));
    let mut generated_tokens = vec![];

    // MAX TOKENS: 10
    for i in 0..10 {
        // println!("\nINPUTS: {decoder_input_ids:?}");
        let decoder_outputs = decoder_model.run(inputs! {
            "input_ids" => decoder_input_ids.view(),
            // "encoder_attention_mask" => attention_mask_array.to_owned(),
            "encoder_hidden_states" => encoder_outputs,
            "use_cache_branch" => Array1::from_vec(vec![false])
        }?)?;

        let decoder_logits = decoder_outputs["logits"].try_extract_tensor::<f32>()?;
        let decoder_logits = decoder_logits.into_dimensionality::<Ix3>().unwrap();
        let decoder_logits = decoder_logits.softmax(Axis(2));
        // println!("Decoder SOFTMAX: {decoder_logits:?}");

        let next_token_id = Zip::from(decoder_logits.lanes(Axis(2))).map_collect(argmax);
        //println!("Decoder ARGMAX: {next_token_id:?}");
        let next_token_id = next_token_id.last().unwrap().0;

        generated_tokens.push(next_token_id);
        decoder_input_ids
            .push(Axis(1), Array1::from_vec(vec![next_token_id as i64]).view())
            .unwrap();

        println!(
            "[{i}] Generated token: {:?}",
            tokenizer.decode(&[next_token_id as _], false)
        );

        // Get from generation_config.json -> "eos_token_id"
        if next_token_id == 1 {
            println!("EOS");
            break;
        }
    }

    /*
    let past_key_values = decoder_outputs.iter().find_map(|(key, value)| {
        if key.contains("present") {
            Some(value.try_extract_tensor::<f32>().unwrap())
        } else {
            None
        }
    });

    println!("\nLogits: {decoder_logits:?}",);
    println!("\npast_key_values: {past_key_values:?}",);

    let decoder_outputs = decoder_model.run(inputs! {
        "input_ids" => decoder_input_ids.view(),
        "encoder_attention_mask" => attention_mask_array.to_owned(),
        "encoder_hidden_states" => encoder_outputs,
        "use_cache_branch" => Array1::from_vec(vec![true]),
        "past_key_values" => past_key_values.unwrap()
    }?)?;

    let decoder_logits = decoder_outputs
        .into_keys()
        .pop_first()
        .unwrap()
        .1
        .try_extract_tensor::<f32>()?;

    //println!("\nDecoder Outputs: {decoder_outputs:?}");
        let decoder_outputs = decoder_outputs.into_dimensionality::<Ix3>().unwrap();
        // println!("2 Decoder Outputs: {decoder_outputs:?}");
        // Collect and sort logits
        let probabilities = &mut decoder_outputs
            .slice(s![0, 0, -1])
            .insert_axis(Axis(0))
            .to_owned()
            .iter()
            .cloned()
            .enumerate()
            .collect::<Vec<_>>();
        probabilities
            .sort_unstable_by(|a, b| b.1.partial_cmp(&a.1).unwrap_or(std::cmp::Ordering::Less));
        // let decoder_outputs = decoder_outputs.softmax(Axis(2));
        // println!("probabilities: Decoder Outputs: {probabilities:?}");
        //
        // Sample using top-k sampling
        let token = probabilities[0].0;
        decoder_input_ids
            .push(Axis(1), Array1::from_vec(vec![token as _]).view())
            .unwrap();

        let token_str = tokenizer.decode(&[token as _], true).unwrap();
        // println!("token: {}", token_str);
        // std::io::stdout().flush().unwrap();
    // */

    Ok(())
}
