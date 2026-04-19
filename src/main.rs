use std::fs::File;
use std::io::{BufRead, BufReader};
use russcip::prelude::*;
use russcip::variable::*;

fn solve_grouping(ids: Vec<String>, scores: Vec<f64>) {
    let n = ids.len();
    if n % 3 != 0 {
        println!("Error: Input count ({}) must be a multiple of 3.", n);
        return;
    }

    let num_groups = n / 3;
    let total_score: f64 = scores.iter().sum();
    let target_sum = total_score / num_groups as f64;

    // 1. Initialize Model
    let mut model = Model::new()
        .hide_output()
        .include_default_plugins()
        .create_prob("picjle")
        .set_obj_sense(ObjSense::Minimize);

    // 2. Decision Variables: x[i][j] (Binary)
    // x[i][j] = 1 if item i is in group j
    let mut x = Vec::new();
    for i in 0..n {
        let mut row = Vec::new();
        for j in 0..num_groups {
            let variable = model.add_var(
                0.0, 
                1.0, 
                0.0, 
                &format!("x_{}_{}", i, j), 
                VarType::Binary
            );
            row.push(variable);
        }
        x.push(row);
    }

    // 3. Deviation Variables: d[j] (Continuous)
    // Objective coefficient is 1.0 (minimizing sum of deviations)
    let big_m = 1e20;
    let mut d = Vec::new();
    for j in 0..num_groups {
        let variable = model.add_var(
            0.0, 
            big_m, 
            1.0, 
            &format!("d_{}", j), 
            VarType::Continuous
        );
        d.push(variable);
    }

    // 4. Constraint: Each item must be in exactly one group
    for i in 0..n {
        let vars: Vec<&Variable> = (0..num_groups).map(|j| &x[i][j]).collect();
        let coeffs = vec![1.0; num_groups];
        model.add_cons(vars, &coeffs, 1.0, 1.0, &format!("item_once_{}", i));
    }

    // 5. Constraint: Each group must have exactly 3 items
    for j in 0..num_groups {
        let vars: Vec<&Variable> = (0..n).map(|i| &x[i][j]).collect();
        let coeffs = vec![1.0; n];
        model.add_cons(vars, &coeffs, 3.0, 3.0, &format!("group_size_{}", j));
    }

    // 6. Absolute Deviation Constraints (Linearizing Variance)
    for j in 0..num_groups {
        let mut vars = (0..n).map(|i| &x[i][j]).collect::<Vec<&Variable>>();
        vars.push(&d[j]);

        // d[j] - Sum(scores * x) >= -target_sum
        let mut coeffs_up = scores.iter().map(|s| -*s).collect::<Vec<f64>>();
        coeffs_up.push(1.0);
        model.add_cons(vars.clone(), &coeffs_up, -target_sum, big_m, &format!("dev_up_{}", j));

        // d[j] + Sum(scores * x) >= target_sum
        let mut coeffs_low = scores.clone();
        coeffs_low.push(1.0);
        model.add_cons(vars, &coeffs_low, target_sum, big_m, &format!("dev_low_{}", j));
    }

    // 7. Solve
    let solved_model = model.solve();
    let status = solved_model.status();

    if status == Status::Optimal {
        let sol = solved_model.best_sol().expect("No solution found");
        
        println!("\nTarget Average Sum: {:.2}", target_sum);
        for j in 0..num_groups {
            let mut members = Vec::new();
            let mut current_sum = 0.0;
            for i in 0..n {
                if sol.val(&x[i][j]) > 0.5 {
                    members.push(&ids[i]);
                    current_sum += scores[i];
                }
            }
            println!("Group {}: {:?} | Sum: {:.2} (Dev: {:.2})", 
                j + 1, members, current_sum, (current_sum - target_sum).abs());
        }
    } else {
        println!("Solver Status: {:?}", status);
    }
}

fn main() {
    // let ids = vec!["A", "B", "C", "D", "E", "F"]
        // .into_iter().map(String::from).collect();
    // let scores = vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0];

    let file = File::open("data99").expect("Unable to open file");
    let reader = BufReader::new(file);
    let mut ids = Vec::new();
    let mut scores = Vec::new();
    for line in reader.lines() {
        let line = line.expect("Unable to read line");
        let parts: Vec<&str> = line.split(',').collect();
        ids.push(parts[0].to_string());
        scores.push(parts[1].parse::<f64>().expect("Invalid score"));
    }
    solve_grouping(ids, scores);
}