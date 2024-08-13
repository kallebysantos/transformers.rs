use transformers_rs::{sentiment_analysis, SentimentAnalysisInput};

fn main() {
    let input = SentimentAnalysisInput::Batch(vec![
        String::from("I've been waiting for a HuggingFace course my whole life."),
        String::from("I hate this so much!"),
    ]);
    let result = sentiment_analysis(input.to_owned());
    println!("\n1 - TEXT: {:?}\nR: {:?}", input, result);

    let input = SentimentAnalysisInput::Single(String::from("I love candies"));
    let result = sentiment_analysis(input.to_owned());
    println!("\n2 - TEXT: {:?}\nR: {:?}", input, result);
}
