use transformers_rs::tasks::{
    feature_extraction, sentiment_analysis, token_classification, zero_shot_classification,
    FeatureExtractionInput, SentimentAnalysisInput, TaskIO, TokenClassificationAggregationStrategy,
    TokenClassificationOptions, ZeroShotClassificationInput,
};

fn main() {
    /*
    let input = SentimentAnalysisInput::Batch(vec![
        String::from("I've been waiting for a HuggingFace course my whole life."),
        String::from("I hate this so much!"),
    ]);
    let result = sentiment_analysis(input.to_owned());
    println!("\n1 - TEXT: {:?}\nR: {:?}", input, result);

    let input = SentimentAnalysisInput::Single(String::from("I love candies"));
    let result = sentiment_analysis(input.to_owned());
    println!("\n2 - TEXT: {:?}\nR: {:?}", input, result);
    */

    /*
    let result = feature_extraction(
        FeatureExtractionInput::Batch(vec![
            String::from("I've been waiting for a HuggingFace course my whole life."),
            String::from("I hate this so much!"),
            String::from("Hello World"),
        ]),
        None,
    );
    println!("R: {:?}", result);

    let result = feature_extraction(FeatureExtractionInput::Single(String::from("Hello World")));
    println!("R: {:?}", result);
    */

    /*
        let input = ZeroShotClassificationInput(
            String::from("I have a problem with my iphone that needs to be resolved asap!!"),
            vec![
                String::from("urgent"),
                String::from("not urgent"),
                String::from("phone"),
                String::from("tablet"),
                String::from("computer"),
            ],
        );

        let result = zero_shot_classification(
            input.to_owned(),
            Some(ZeroShotClassificationOptions {
                multi_label: true,
                ..Default::default()
            }),
        );

        println!("Input: {:#?}", input);
        println!("Result: {:#?}", result);
    */

    let result = token_classification(
        String::from(
            "My name is Sylvain and I work at Hugging Face in Brooklyn.",
            //"Alice and Bob",
        ),
        Some(TokenClassificationOptions {
            aggregation_strategy: TokenClassificationAggregationStrategy::Simple,
            //ignore_labels: vec![String::from("O"), String::from("PER")], //..Default::default()
            ..Default::default()
        }),
    );

    println!("\n\nResult: {:#?}", result);
}
