// Copyright 2025 Google LLC
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//      http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.

// Entry point for the fluid simulation.

use instant::Instant;
use js_sys;
use rayon::prelude::*;
use std::cmp;
use std::collections::HashSet;
use std::f32::consts::PI;
use std::ops::{Add, AddAssign, Neg, Sub, SubAssign};
use wasm_bindgen::prelude::*;

const CELL_SIZE: f32 = 0.25;
const EPSILON: f32 = 1e-12;
const NUM_ITERATIONS: usize = 3;
const NUM_PARTICLES: usize = 5000;
const RELAXATION: f32 = 1e-4;
const DEFAULT_BETA: f32 = 1.005;
const VISCOSITY_STIFFNESS_SV: f32 = 0.5;
const REST_DENSITY: f32 = 1e9;

extern crate web_sys;

// A macro to provide `println!(..)`-style syntax for `console.log` logging.
#[allow(unused_macros)]
macro_rules! log {
    ( $( $t:tt )* ) => {
        web_sys::console::log_1(&format!( $( $t )* ).into());
    }
}

#[derive(Clone, Debug)]
pub struct ViscosityConstraint {
    p1_idx: usize,
    p2_idx: usize,
    d_ij: f32, // Reference distance (rest length)
}

pub struct ParticlesData {
    pub positions: Vec<Vec3>,
    pub velocities: Vec<Vec3>,
    pub predictions: Vec<Vec3>,
    pub lambdas: Vec<f32>,
    pub delta_ps: Vec<Vec3>,
    pub betas: Vec<f32>,
    count: usize,
}

impl ParticlesData {
    pub fn new(num_particles: usize) -> Self {
        Self {
            positions: vec![Vec3::default(); num_particles],
            velocities: vec![Vec3::default(); num_particles],
            predictions: vec![Vec3::default(); num_particles],
            lambdas: vec![0.0; num_particles],
            delta_ps: vec![Vec3::default(); num_particles],
            betas: vec![DEFAULT_BETA; num_particles],
            count: num_particles,
        }
    }

    pub fn len(&self) -> usize {
        self.count
    }
}

#[wasm_bindgen]
#[derive(Clone, Copy, Debug, Default, PartialEq)]
pub struct Vec3 {
    x: f32,
    y: f32,
    z: f32,
}

#[derive(Clone, Copy, Debug, Default)]
#[wasm_bindgen]
pub struct Quaternion {
    pub x: f32,
    pub y: f32,
    pub z: f32,
    pub w: f32,
}

#[wasm_bindgen]
pub struct UniformGrid {
    cells: Vec<Vec<usize>>,
    cell_size: f32,
    num_x: usize,
    num_y: usize,
    num_z: usize,
    bounding_box_min: Vec3,
    bounding_box_max: Vec3,
}

#[wasm_bindgen]
pub struct Simulation {
    num_particles: usize,
    particles: ParticlesData,
    gravity: f32,
    last_update_time: Option<Instant>,
    grid: UniformGrid,
    rest_density: f32,
    pub xsph_viscosity: f32,
    pipe: Pipe,

    viscosity_constraints: Vec<ViscosityConstraint>,
    viscosity_stiffness_sv: f32,
    max_constraint_distance_h: f32,
}

#[derive(Clone, Copy, Debug)]
#[wasm_bindgen]
pub struct Pipe {
    pub tip: Vec3,
    pub end: Vec3,
    pub radius: f32,
    pub orientation: Quaternion,
}

#[wasm_bindgen]
impl Vec3 {
    pub fn new(x: f32, y: f32, z: f32) -> Vec3 {
        Vec3 {
            x,
            y,
            z
        }
    }
    pub fn norm(&self) -> f32 {
        (self.x * self.x + self.y * self.y + self.z * self.z).sqrt()
    }

    pub fn norm_squared(&self) -> f32 {
        self.x * self.x + self.y * self.y + self.z * self.z
    }

    pub fn cross(&self, other: Vec3) -> Vec3 {
        Vec3 {
            x: self.y * other.z - self.z * other.y,
            y: self.z * other.x - self.x * other.z,
            z: self.x * other.y - self.y * other.x
        }
    }

    pub fn scalar_mul(&self, scalar: f32) -> Vec3 {
        Vec3 {
            x: self.x * scalar,
            y: self.y * scalar,
            z: self.z * scalar,
        }
    }

    fn is_neighbor(&self, other_particle: &Vec3, h: f32) -> bool {
        let dist_sq = (*self - *other_particle).norm_squared();
        dist_sq < h * h
    }

    pub fn handle_wall_collision(&mut self, bounding_box_min: &Vec3, bounding_box_max: &Vec3) {
        self.x = self.x.clamp(bounding_box_min.x, bounding_box_max.x);
        self.y = self.y.clamp(bounding_box_min.y, bounding_box_max.y);
        self.z = self.z.clamp(bounding_box_min.z, bounding_box_max.z);
    }

    pub fn handle_pipe_collision(&mut self, pipe: &Pipe) {
        let line_vec = pipe.end - pipe.tip;
        let point_vec = *self - pipe.tip;
        let t = (line_vec.x * point_vec.x + line_vec.y * point_vec.y + line_vec.z * point_vec.z) / line_vec.norm_squared();
        let t_clamped = t.clamp(0.0, 1.0);

        let closest_point_on_line = pipe.tip + line_vec.scalar_mul(t_clamped);

        let dist_vec = *self - closest_point_on_line;
        let distance = dist_vec.norm();

        if distance < pipe.radius && distance > EPSILON {
            let penetration = pipe.radius - distance;
            let correction = dist_vec.scalar_mul(penetration / distance);
            *self += correction;
        }
    }
}

impl Add for Vec3 {
    type Output = Self;

    fn add(self, other: Self) -> Self {
        Vec3 {
            x: self.x + other.x,
            y: self.y + other.y,
            z: self.z + other.z,
        }
    }
}

impl AddAssign for Vec3 {
    fn add_assign(&mut self, other: Self) {
        *self = Self {
            x: self.x + other.x,
            y: self.y + other.y,
            z: self.z + other.z,
        };
    }
}

impl Sub for Vec3 {
    type Output = Self;

    fn sub(self, other: Self) -> Self {
        Vec3 {
            x: self.x - other.x,
            y: self.y - other.y,
            z: self.z - other.z,
        }
    }
}

impl SubAssign for Vec3 {
    fn sub_assign(&mut self, other: Self) {
        *self = Self {
            x: self.x - other.x,
            y: self.y - other.y,
            z: self.z - other.z,
        };
    }
}

impl Neg for Vec3 {
    type Output = Self;

    fn neg(self) -> Self {
        Vec3 {
            x: -self.x,
            y: -self.y,
            z: -self.z,
        }
    }
}

impl std::iter::Sum for Vec3 {
    fn sum<I>(iter: I) -> Self
    where
        I: Iterator<Item = Self>
    {
        iter.fold( Vec3::default(), |acc, item| Vec3 {
            x: acc.x + item.x,
            y: acc.y + item.y,
            z: acc.z + item.z,
        })
    }
}

#[wasm_bindgen]
impl UniformGrid {
    pub fn new(cell_size: f32, bounding_box_min: Vec3,
                bounding_box_max: Vec3) -> UniformGrid {
        let num_x = bounding_box_max.x - bounding_box_min.x;
        let num_y = bounding_box_max.y - bounding_box_min.y;
        let num_z = bounding_box_max.z - bounding_box_min.z;

        let num_x = (num_x / cell_size).ceil() as usize;
        let num_y = (num_y / cell_size).ceil() as usize;
        let num_z = (num_z / cell_size).ceil() as usize;

        let num_cells = num_x * num_y * num_z;
        let cells = vec![Vec::new(); num_cells];

        UniformGrid {
            cells,
            cell_size,
            num_x,
            num_y,
            num_z,
            bounding_box_min,
            bounding_box_max,
        }
    }

    fn get_cell_index(&self, ix: usize, iy: usize, iz: usize) -> usize {
        ix + iy * self.num_x + iz * self.num_x * self.num_y
    }

    pub fn insert_particle(&mut self, position: &Vec3, index: usize) {
      let ix = ((position.x - self.bounding_box_min.x) / self.cell_size)
                        .floor() as usize;
      let iy = ((position.y - self.bounding_box_min.y) / self.cell_size)
                        .floor() as usize;
      let iz = ((position.z - self.bounding_box_min.z) / self.cell_size)
                        .floor() as usize;

      let ix = cmp::min(cmp::max(ix, 0), self.num_x - 1);
      let iy = cmp::min(cmp::max(iy, 0), self.num_y - 1);
      let iz = cmp::min(cmp::max(iz, 0), self.num_z - 1);

      let cell_index = self.get_cell_index(ix, iy, iz);
      self.cells[cell_index].push(index);
    }

    pub fn clear(&mut self) {
        for cell in &mut self.cells {
            cell.clear();
        }
    }

    pub fn find_neighbors(&self, position: Vec3) -> Vec<usize> {
        let mut neighbors = Vec::new();
        let ix = ((position.x - self.bounding_box_min.x) / self.cell_size)
                            .floor() as usize;
        let iy = ((position.y - self.bounding_box_min.y) / self.cell_size)
                            .floor() as usize;
        let iz = ((position.z - self.bounding_box_min.z) / self.cell_size)
                            .floor() as usize;

        let ix = cmp::min(cmp::max(ix, 0), self.num_x - 1);
        let iy = cmp::min(cmp::max(iy, 0), self.num_y - 1);
        let iz = cmp::min(cmp::max(iz, 0), self.num_z - 1);

        for z in (iz.saturating_sub(1))..=(cmp::min(iz + 1, self.num_z - 1)) {
            for y in (iy.saturating_sub(1))..=(cmp::min(iy + 1, self.num_y - 1)) {
                for x in (ix.saturating_sub(1))..=(cmp::min(ix + 1, self.num_x - 1)) {
                    let cell_index = self.get_cell_index(x, y, z);
                    neighbors.extend(&self.cells[cell_index]);
                }
            }
        }
        neighbors
    }
}

#[wasm_bindgen]
impl Simulation {
    // Helper function to calculate the Poly6 kernel value at r=0 (self-contribution)
    // W(0, h) = 315.0 / (64.0 * PI * h^3)
    fn kernel_poly6_zero(h: f32) -> f32 {
        315.0 / (64.0 * PI * h.powi(3))
    }

    #[wasm_bindgen(constructor)]
    pub fn new(bounding_box_dim: f32) -> Simulation {
        let bounding_box_min = Vec3::new(
            -bounding_box_dim / 2.0,
            0.0,
            -bounding_box_dim / 2.0
        );
        let bounding_box_max = Vec3::new(
            bounding_box_dim / 2.0,
            bounding_box_dim,
            bounding_box_dim / 2.0
        );
        let cell_size = CELL_SIZE;
        let grid = UniformGrid::new(cell_size, bounding_box_min,
                                                 bounding_box_max);

        let pipe = Pipe {
            tip: Vec3::default(),
            end: Vec3::default(),
            radius: 0.5,
            orientation: Quaternion {
                x: 0.0, y: 0.0, z: 0.0, w: 1.0
            }
        };

        let calculated_rest_density = Simulation::kernel_poly6_zero(cell_size) * 1.2;

        let mut simulation = Simulation {
            num_particles: NUM_PARTICLES,
            particles: ParticlesData::new(NUM_PARTICLES),
            gravity: -9.81,
            last_update_time: None,
            grid,
            rest_density: REST_DENSITY,
            xsph_viscosity: 0.01,
            pipe,
            viscosity_constraints: Vec::new(),
            viscosity_stiffness_sv: VISCOSITY_STIFFNESS_SV,
            max_constraint_distance_h: 2.0 * cell_size, // H = 2h
        };
        simulation.reset_particles();

        simulation
    }

    pub fn update_pipe_transform(
        &mut self,
        tip_x: f32, tip_y: f32, tip_z: f32,
        end_x: f32, end_y: f32, end_z: f32,
        q_x: f32, q_y: f32, q_z: f32, q_w: f32,
    ) {
        self.pipe.tip = Vec3::new(tip_x, tip_y, tip_z);
        self.pipe.end = Vec3::new(end_x, end_y, end_z);
        self.pipe.orientation = Quaternion { x: q_x, y: q_y, z: q_z, w: q_w };
    }

    fn kernel_poly6(&self, r: Vec3, h: f32) -> f32 {
        let r_norm = r.norm();
        if r_norm >= 0.0 && r_norm <= h {
            let factor = 315.0 / (64.0 * PI * h.powi(9));
            return factor * (h.powi(2) - r_norm.powi(2)).powi(3);
        }
        0.0
    }

    fn kernel_spiky_grad(&self, r: Vec3, h: f32) -> Vec3 {
        let r_norm = r.norm();
        if r_norm > EPSILON && r_norm <= h {
            let r_unit = r.scalar_mul(1.0 / r_norm);
            let factor = -45.0 / (PI * h.powi(6));
            let scalar = factor * (h - r_norm).powi(2);
            // Flip the sign because we are actually taking the gradient with
            // respect to the neighbor particle p_i, even though we defined the
            // distance to be r := p_i - p_j.
            return -r_unit.scalar_mul(scalar);
        }
        Vec3::new(0.0, 0.0, 0.0)
    }

    fn calculate_density(
        &self,
        particle_i_index: usize,
        all_neighbors: &Vec<Vec<usize>>,
    ) -> f32 {
        let particle_i = self.particles.predictions[particle_i_index];
        let neighbors_i = &all_neighbors[particle_i_index];
        let cutoff = self.grid.cell_size;

        neighbors_i
            .iter()
            .map(|&neighbor_j_index| {
                let neighbor_j = self.particles.predictions[neighbor_j_index];
                let r = particle_i - neighbor_j;
                self.kernel_poly6(r, cutoff)
            })
            .sum()
    }

    // Computes the gradient of the constraint function for particle i, C_i,
    // with respect to particle k, i.e., \nabla_p_k C_i. Potentially accesses
    // and sums over all the neighbors of particle i.
    fn calculate_grad_constraint(
        &self,
        particle_i_index: usize,
        particle_k_index: usize,
        all_neighbors: &Vec<Vec<usize>>,
    ) -> Vec3 {
        let cutoff = self.grid.cell_size;
        let particle_i = self.particles.predictions[particle_i_index];

        // Case 1: grad(C_i) w.r.t. itself, so a movement affects all neighbors
        if particle_i_index == particle_k_index {
            let neighbors_of_i = &all_neighbors[particle_i_index];
            let sum_vec: Vec3 = neighbors_of_i
                .iter()
                .map(|&j_index| {
                    let dist_ij = particle_i - self.particles.predictions[j_index];
                    self.kernel_spiky_grad(dist_ij, cutoff)
                })
                .sum();
            sum_vec.scalar_mul(1.0 / self.rest_density)

        // Case 2: grad(C_i) w.r.t. another k; if k is not a neighbor, then
        // the gradient must be 0.
        } else {
            let particle_k = self.particles.predictions[particle_k_index];
            if particle_i.is_neighbor(&particle_k, cutoff) {
                let dist_ik = particle_i - particle_k;
                -self
                    .kernel_spiky_grad(dist_ik, cutoff)
                    .scalar_mul(1.0 / self.rest_density)
            } else {
                Vec3::default()
            }
        }
    }

    fn apply_xsph_viscosity(&mut self, c: f32, all_neighbors: &Vec<Vec<usize>>) {
        let cutoff = self.grid.cell_size;

        let velocity_corrections: Vec<Vec3> = (0..self.num_particles)
                .into_par_iter()
                .map(|i| {
                    all_neighbors[i]
                        .iter()
                        .map(|&j| {
                            let v_ij = self.particles.velocities[j] - self.particles.velocities[i];
                            let r_ij = self.particles.positions[i] - self.particles.positions[j];
                            let kernel_w = self.kernel_poly6(r_ij, cutoff);
                            v_ij.scalar_mul(kernel_w)
                        })
                        .sum::<Vec3>()
                })
                .collect();

        // Apply the corrections
        for i in 0..self.num_particles {
            self.particles.velocities[i] += velocity_corrections[i]
                .scalar_mul(c);
        }
    }

    fn manage_viscosity_constraints(&mut self, all_neighbors: &Vec<Vec<usize>>) {
        let h = self.grid.cell_size;
        let h_delete = self.max_constraint_distance_h;

        let predictions = &self.particles.predictions;
        let betas = &self.particles.betas;

        // 1. Delete and Modify existing constraints (Alg 1, lines 1-7)
        // Use retain_mut for efficient in-place modification and removal.
        self.viscosity_constraints.retain_mut(|constraint| {
            let p1 = predictions[constraint.p1_idx];
            let p2 = predictions[constraint.p2_idx];
            let current_dist = (p1 - p2).norm();

            if current_dist > h_delete {
                return false;
            }

            let alpha = 0.01;
            if constraint.d_ij - current_dist < alpha * constraint.d_ij {
                let beta_i = betas[constraint.p1_idx];
                let beta_j = betas[constraint.p2_idx];

                let avg_beta_increment = (beta_i + beta_j) * 0.5;
                constraint.d_ij *= 1.0 + avg_beta_increment;

                if constraint.d_ij > h_delete {
                    constraint.d_ij = h_delete;
                }
            }
            true
        });

        // 2. Generate new constraints (Alg 1, lines 8-11)
        // Track existing pairs to avoid duplication.
        let mut existing_pairs: HashSet<(usize, usize)> = self.viscosity_constraints
            .iter()
            .map(|c| {
                if c.p1_idx < c.p2_idx { (c.p1_idx, c.p2_idx) } else { (c.p2_idx, c.p1_idx) }
            })
            .collect();

        for i in 0..self.num_particles {
            for &j in &all_neighbors[i] {
                if i < j {
                    let pair = (i, j);
                    if !existing_pairs.contains(&pair) {
                        let dist = (predictions[i] - predictions[j]).norm();

                        if dist < h && dist > EPSILON {
                            self.viscosity_constraints.push(ViscosityConstraint {
                                p1_idx: i,
                                p2_idx: j,
                                d_ij: dist,
                            });
                            existing_pairs.insert(pair);
                        }
                    }
                }
            }
        }
    }

    fn calculate_viscosity_corrections(&self) -> Vec<Vec3> {
        let s_v = self.viscosity_stiffness_sv;
        let mass_ratio = 0.5;
        let predictions = &self.particles.predictions;

        self.viscosity_constraints
            .par_iter()
            .fold(
                || vec![Vec3::default(); self.num_particles],
                |mut acc, constraint| {
                    let i = constraint.p1_idx;
                    let j = constraint.p2_idx;
                    let d_ij = constraint.d_ij;

                    let p_ij = predictions[i] - predictions[j];
                    let current_dist = p_ij.norm();

                    if current_dist > d_ij && current_dist > EPSILON {
                        let violation = current_dist - d_ij;
                        let direction = p_ij.scalar_mul(1.0 / current_dist);

                        let correction_magnitude = s_v * mass_ratio * violation;
                        let correction = direction.scalar_mul(correction_magnitude);

                        acc[i] -= correction;
                        acc[j] += correction;
                    }
                    acc
                },
            )
            .reduce(
                || vec![Vec3::default(); self.num_particles],
                |mut acc1, acc2| {
                    for k in 0..self.num_particles {
                        acc1[k] += acc2[k];
                    }
                    acc1
                },
            )
    }

    pub fn update(&mut self) {
        let dt = 1.0 / 60.0;

        let now = instant::Instant::now();
        self.last_update_time = Some(now);

        self.grid.clear();
        for i in 0..self.num_particles {
            self.grid.insert_particle(&self.particles.positions[i], i);
        }
        let initial_neighbors: Vec<Vec<usize>> = (0..self.num_particles)
            .into_par_iter()
            .map(|i| self.grid.find_neighbors(self.particles.positions[i]))
            .collect();

        // 1. Apply XSPH viscosity (Alg 2, line 5)
        self.apply_xsph_viscosity(self.xsph_viscosity, &initial_neighbors);


        // 2. Apply forces and predict positions (Alg 2, lines 6-7)
        for i in 0..self.num_particles {
            self.particles.velocities[i].y += self.gravity * dt;
            let mut prediction = self.particles.positions[i].clone();
            prediction +=  self.particles.velocities[i].scalar_mul(dt);
            self.particles.predictions[i] = prediction;
        }

        // 3. Find neighbors based on predictions (p_i) (Alg 2, line 10)
        self.grid.clear();
        for i in 0..self.num_particles {
            self.grid.insert_particle(&self.particles.predictions[i], i);
        }

        // Cache neighbors for the solver loop.
        let all_neighbors: Vec<Vec<usize>> = (0..self.num_particles)
            .into_par_iter()
            .map(|i| self.grid.find_neighbors(self.particles.predictions[i]))
            .collect();

        // 4. Control constraints (Alg 2, line 11)
        self.manage_viscosity_constraints(&all_neighbors);

        for _ in 0..NUM_ITERATIONS {
            // 5a. Calculate Lambdas (Density Constraint)
            let new_lambdas: Vec<f32> = (0..self.num_particles)
                .into_par_iter()
                .map(|i| {
                    let density_i = self.calculate_density(i, &all_neighbors);
                    let constraint_i = (density_i / self.rest_density) - 1.0;

                    let sum_grad_constraints: f32 = all_neighbors[i]
                        .iter()
                        .map(|&k| {
                            self.calculate_grad_constraint(i, k, &all_neighbors)
                                .norm_squared()
                        })
                        .sum();

                    if sum_grad_constraints < EPSILON && constraint_i.abs() < EPSILON {
                        0.0
                    } else {
                        -constraint_i / (sum_grad_constraints + RELAXATION)
                    }
                })
                .collect();
            self.particles.lambdas = new_lambdas;

            // 5b. Calculate Corrections (Δp_dens and Δp_visc) (Alg 2, line 18)

            // i. Calculate Density Corrections (Δp_dens)
            let k = 0.1;
            let h = self.grid.cell_size;
            let delta_q_mag = 0.2 * h;
            let delta_q_vec = Vec3::new(delta_q_mag, 0.0, 0.0);
            let n = 4;
            let s_corr_denom = self.kernel_poly6(delta_q_vec, h);

            let density_delta_ps: Vec<Vec3> = (0..self.num_particles)
                .into_par_iter()
                .map(|i| {
                     all_neighbors[i]
                        .iter()
                        .map(|&j| {
                            let cutoff = self.grid.cell_size;
                            let dist_ij = self.particles.predictions[i] - self.particles.predictions[j];
                            let s_corr_numer = self.kernel_poly6(dist_ij, h);

                            let s_corr = if s_corr_denom > EPSILON {
                                -k * (s_corr_numer / s_corr_denom).powi(n)
                            } else {
                                0.0
                            };

                            let scalar = self.particles.lambdas[i] + self.particles.lambdas[j] + s_corr;
                            self.kernel_spiky_grad(dist_ij, cutoff)
                                .scalar_mul(scalar)
                        })
                        .sum::<Vec3>()
                        .scalar_mul(1.0 / self.rest_density)
                })
                .collect();

            // ii. Calculate Viscosity Corrections (Δp_visc)
            let viscosity_delta_ps = self.calculate_viscosity_corrections();

            // 5c. Update predictions and handle collisions (Alg 2, line 20)
            // p_i <- p_i + Δp_dens + Δp_visc
            self.particles.predictions.iter_mut().enumerate().for_each(|(i, prediction_i)| {
                let total_delta_p = density_delta_ps[i] + viscosity_delta_ps[i];

                // Store total delta_p (optional, useful for debugging)
                self.particles.delta_ps[i] = total_delta_p;
                *prediction_i += total_delta_p;

                // Handle collisions after combining all corrections
                prediction_i.handle_pipe_collision(&self.pipe);
                prediction_i.handle_wall_collision(&self.grid.bounding_box_min,
                                                   &self.grid.bounding_box_max);
            });
        }

        // 6. Update velocities (Alg 2, line 22)
        for i in 0..self.num_particles {
            if dt > EPSILON {
                self.particles.velocities[i] = (self.particles.predictions[i] - self.particles.positions[i])
                                            .scalar_mul(1.0 / dt);
            }
        }

        // 7. Final position update (Alg 2, line 23)
        for i in 0..self.num_particles {
            self.particles.positions[i] = self.particles.predictions[i];
        }
    }

    // Used by frontend to load particle positions into the scene.
    pub fn get_particle_positions(&self) -> Vec<f32> {
        let mut positions = Vec::with_capacity(self.particles.len() * 3);
        for particle in &self.particles.positions {
            positions.push(particle.x);
            positions.push(particle.y);
            positions.push(particle.z);
        }
        positions
    }

    // Used by frontend as a manual "reset" button.
    pub fn reset_particles(&mut self) {
        // Clear constraints and reset particle data
        self.viscosity_constraints.clear();
        for i in 0..self.num_particles {
            self.particles.betas[i] = DEFAULT_BETA;
            self.particles.velocities[i] = Vec3::default();
        }

        // Initialize in a structured grid.
        // Spacing should be related to the kernel radius for stable packing.
        let spacing = self.grid.cell_size * 1.0;
        let particles_per_dim = (self.num_particles as f32).cbrt().ceil() as usize;
        let y_offset = 5.0; // Start higher up

        // Calculate offsets to center the block horizontally
        let block_width = particles_per_dim as f32 * spacing;
        let center_offset = -block_width / 2.0;

        for i in 0..self.num_particles {
            let ix = i % particles_per_dim;
            let iy = (i / particles_per_dim) % particles_per_dim;
            let iz = i / (particles_per_dim * particles_per_dim);

            let x = (ix as f32 * spacing) + center_offset;
            let y = (iy as f32 * spacing) + y_offset;
            let z = (iz as f32 * spacing) + center_offset;

            // Add a tiny jitter to prevent perfect alignment artifacts
            let jitter_scale = spacing * 0.05;
            let jitter = Vec3::new(
                (js_sys::Math::random() as f32 - 0.5) * jitter_scale,
                (js_sys::Math::random() as f32 - 0.5) * jitter_scale,
                (js_sys::Math::random() as f32 - 0.5) * jitter_scale,
            );

            self.particles.positions[i] = Vec3::new(x, y, z) + jitter;
        }

        self.particles.predictions = self.particles.positions.clone();
    }
}
