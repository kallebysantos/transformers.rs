use ndarray::{Array2, ArrayView1, ArrayView3, Axis};

pub fn mean_pool(
    last_hidden_states: ArrayView3<f32>,
    attention_mask: ArrayView3<i64>,
) -> Array2<f32> {
    let masked_hidden_states = last_hidden_states.into_owned() * &attention_mask.mapv(|x| x as f32);
    let sum_hidden_states = masked_hidden_states.sum_axis(Axis(1));
    let sum_attention_mask = attention_mask.mapv(|x| x as f32).sum_axis(Axis(1));

    sum_hidden_states / sum_attention_mask
}

pub fn argmax(lane: ArrayView1<f32>) -> (usize, f32) {
    lane.iter()
        .enumerate()
        .fold((usize::MIN, -f32::INFINITY), |maxima, curr| {
            match maxima.1.total_cmp(curr.1) {
                std::cmp::Ordering::Greater => maxima,
                _ => (curr.0, *curr.1),
            }
        })
}
