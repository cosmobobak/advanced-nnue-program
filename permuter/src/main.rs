// #![warn(clippy::all, clippy::pedantic, clippy::nursery)]

use std::{cmp::Reverse, time::Instant};

use anyhow::{Context, ensure};
use rand::{Rng, rngs::ThreadRng};
use rayon::iter::{IntoParallelIterator, ParallelIterator};

#[derive(Debug, Clone, serde::Deserialize)]
#[serde(transparent)]
struct Matrix {
    data: Vec<Vec<u32>>,
}

fn cost_function(order: &[u16], matrix: &Matrix) -> u64 {
    const BLOCK_SIZE: usize = 4;

    (BLOCK_SIZE..order.len() - BLOCK_SIZE)
        .into_par_iter()
        .map(|i| {
            let matrix_row = &matrix.data[order[i] as usize][..];
            let cost_a = order[i + BLOCK_SIZE..]
                .iter()
                .map(|&j| matrix_row[j as usize] as u64)
                .sum::<u64>();
            let cost_b = order[..i - BLOCK_SIZE]
                .iter()
                .map(|&j| matrix_row[j as usize] as u64)
                .sum::<u64>();
            cost_a + cost_b
        })
        .sum()
}

fn cost_difference_for_swap(order: &[u16], matrix: &Matrix, i: usize, j: usize) -> i64 {
    const BLOCK_SIZE: usize = 4;
    let n = order.len();
    let mut diff = 0i64;

    // Only recalculate for positions affected by the swap
    for pos in 0..n {
        if pos < BLOCK_SIZE || pos >= n - BLOCK_SIZE {
            continue;
        }

        let elem_at_pos = order[pos] as usize;

        // Check if this position's cost is affected by the swap
        let affects_i = (i >= pos + BLOCK_SIZE || i < pos - BLOCK_SIZE) && pos != i;
        let affects_j = (j >= pos + BLOCK_SIZE || j < pos - BLOCK_SIZE) && pos != j;

        if affects_i {
            let old_cost = matrix.data[elem_at_pos][order[i] as usize] as i64;
            let new_cost = matrix.data[elem_at_pos][order[j] as usize] as i64;
            diff += new_cost - old_cost;
        }

        if affects_j {
            let old_cost = matrix.data[elem_at_pos][order[j] as usize] as i64;
            let new_cost = matrix.data[elem_at_pos][order[i] as usize] as i64;
            diff += new_cost - old_cost;
        }

        // If pos is i or j, recalculate its contribution entirely
        if pos == i || pos == j {
            // Calculate old contribution
            let old_elem = order[pos] as usize;
            let new_elem = order[if pos == i { j } else { i }] as usize;

            for (k, o_k) in order.iter().enumerate() {
                if k >= pos + BLOCK_SIZE || k < pos.saturating_sub(BLOCK_SIZE) {
                    diff -= matrix.data[old_elem][*o_k as usize] as i64;
                    diff += matrix.data[new_elem][*o_k as usize] as i64;
                }
            }
        }
    }

    diff
}

fn greedy_sort(order: &mut [u16], matrix: &Matrix) {
    let mut changed = true;
    let mut cost = cost_function(order, matrix);
    let mut improvements = 0;
    let mut iters = 0u64;
    let start = Instant::now();
    while changed {
        changed = false;
        for i in 0..order.len() - 1 {
            for j in i + 1..order.len() {
                iters += 1;
                let delta = cost_difference_for_swap(order, matrix, i, j);
                if delta >= 0 {
                    continue; // Skip if swap does not improve cost
                }
                order.swap(i, j);
                cost = cost.wrapping_add_signed(delta);
                changed = true;
                if improvements % 1000 == 0 {
                    println!("Improvement #{}: cost = {}", improvements, cost);
                }
                improvements += 1;
            }
        }
    }
    println!(
        "Greedy sort completed in {:.2}s with {} improvements and {} iterations. Final cost: {}",
        start.elapsed().as_secs_f64(),
        improvements,
        iters,
        cost
    );
    println!("Final order: {:?}", order);
}

fn swap_two(rng: &mut ThreadRng, order: &mut [u16]) -> (usize, usize) {
    let len = order.len();
    let i = rng.random_range(0..len);
    let mut j = rng.random_range(0..len);
    while j == i {
        j = rng.random_range(0..len);
    }
    order.swap(i, j);
    (i, j)
}

fn undo_swap(order: &mut [u16], (i, j): (usize, usize)) {
    order.swap(i, j);
}

fn reverse_segment(rng: &mut ThreadRng, order: &mut [u16]) -> (usize, usize) {
    let len = order.len();
    let mut start = rng.random_range(0..len);
    let mut end = rng.random_range(0..len);
    while end == start {
        end = rng.random_range(0..len);
    }
    if start > end {
        (start, end) = (end, start);
    }
    order[start..=end].reverse();
    (start, end)
}

fn undo_reverse_segment(order: &mut [u16], (start, end): (usize, usize)) {
    order[start..=end].reverse();
}

fn swap_local(rng: &mut ThreadRng, order: &mut [u16]) -> (usize, usize) {
    let len = order.len();
    let i = rng.random_range(0..len - 1);
    let j = i + 1;
    order.swap(i, j);
    (i, j)
}

fn simulated_annealing(
    order: &mut [u16],
    matrix: &Matrix,
    initial_temp: f64,
    cooling_rate: f64,
    min_temp: f64,
) {
    let mut rng = rand::rng();
    let mut current_cost = cost_function(order, matrix);
    let mut best_cost = current_cost;
    let mut best_order = order.to_vec();
    let mut temp = initial_temp;
    let mut iterations = 0;
    let mut improvements = 0;

    let mutators = [swap_local, swap_two, reverse_segment];
    let reversers = [undo_swap, undo_swap, undo_reverse_segment];

    while temp > min_temp {
        let operation = rng.random_range(0..mutators.len());
        let u = mutators[operation](&mut rng, order);
        let new_cost = cost_function(order, matrix);
        let improvement = current_cost as f64 - new_cost as f64;

        if improvement > 0.0 || (improvement / temp).exp() > rng.random::<f64>() {
            // Accept the new order
            current_cost = new_cost;
            if current_cost < best_cost {
                best_cost = current_cost;
                best_order.copy_from_slice(order);
                improvements += 1;
                println!("New best cost: {}", best_cost);
            }
        } else {
            // Undo the operation
            reversers[operation](order, u);
        }

        temp *= cooling_rate;
        iterations += 1;
    }
    order.copy_from_slice(&best_order);
    println!(
        "Final cost: {}, iterations: {}, improvements: {}, time: {:.2}s",
        best_cost,
        iterations,
        improvements,
        Instant::now().elapsed().as_secs_f64()
    );
    println!("Best order: {:?}", best_order);
}

fn main() -> anyhow::Result<()> {
    let matrix =
        std::fs::read_to_string("correlations.json").context("Failed to read correlations.json")?;
    let matrix =
        serde_json::from_str::<Matrix>(&matrix).context("Failed to parse correlations.json")?;

    ensure!(
        matrix.data.iter().all(|row| row.len() == matrix.data.len()),
        "Matrix must be square"
    );

    let diag = matrix
        .data
        .iter()
        .enumerate()
        .map(|(i, row)| row[i])
        .collect::<Vec<_>>();

    let default_order = (0..diag.len() as u16).collect::<Vec<_>>();

    let mut sorted_indices = default_order.clone();
    sorted_indices.sort_unstable_by_key(|&i| Reverse(diag[i as usize]));

    println!(
        "Cost of default order: {}",
        cost_function(&default_order, &matrix)
    );
    println!(
        "Cost of sorted order: {}",
        cost_function(&sorted_indices, &matrix)
    );

    // let mut greedy_order = default_order.clone();
    // greedy_sort(&mut greedy_order, &matrix);

    // println!(
    //     "Cost of greedy sorted order: {}",
    //     cost_function(&greedy_order, &matrix)
    // );

    let mut annealed_order = sorted_indices.clone();
    simulated_annealing(
        &mut annealed_order,
        &matrix,
        1000.0,  // Initial temperature
        0.99999, // Cooling rate
        1e-6,    // Minimum temperature
    );

    println!(
        "Cost of simulated annealing order: {}",
        cost_function(&annealed_order, &matrix)
    );

    greedy_sort(&mut annealed_order, &matrix);

    println!(
        "Cost after greedy sort: {}",
        cost_function(&annealed_order, &matrix)
    );

    // Save the final order to a file
    std::fs::write("final_order.json", serde_json::to_string(&annealed_order)?)
        .context("Failed to write final_order.json")?;

    Ok(())
}
