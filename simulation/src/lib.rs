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
use std::f32::consts::PI;
use std::ops::{Add, AddAssign, Neg, Sub};
use wasm_bindgen::prelude::*;

const CELL_SIZE: f32 = 0.25;
const EPSILON: f32 = 1e-12;
const NUM_ITERATIONS: usize = 3;
const NUM_PARTICLES: usize = 5000;
const RELAXATION: f32 = 1e-4;

extern crate web_sys;

// A macro to provide `println!(..)`-style syntax for `console.log` logging.
#[allow(unused_macros)]
macro_rules! log {
    ( $( $t:tt )* ) => {
        web_sys::console::log_1(&format!( $( $t )* ).into());
    }
}

#[wasm_bindgen]
#[derive(Clone, Copy, Debug, Default, PartialEq)]
pub struct Vec3 {
    x: f32,
    y: f32,
    z: f32,
}

#[wasm_bindgen]
#[derive(Clone, Copy, Debug, Default, PartialEq)]
pub struct Particle {
    position: Vec3,
    velocity: Vec3,
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
    particles: Vec<Particle>,
    predictions: Vec<Particle>,
    bounding_box_dim: f32,
    gravity: f32,
    last_update_time: Option<Instant>,
    grid: UniformGrid,
    rest_density: f32,
    pub stiffness: f32,
    #[allow(dead_code)]
    viscosity: f32,
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

    pub fn scalar_mul(&self, scalar: f32) -> Vec3 {
        Vec3 {
            x: self.x * scalar,
            y: self.y * scalar,
            z: self.z * scalar,
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

impl Particle {
    fn handle_collision(&mut self, bounding_box_min: &Vec3,
                            bounding_box_max: &Vec3) {
        if self.position.z < bounding_box_min.z {
            self.position.z = bounding_box_min.z;
        } else if self.position.z > bounding_box_max.z {
            self.position.z = bounding_box_max.z;
        }

        if self.position.x < bounding_box_min.x {
            self.position.x = bounding_box_min.x;
        } else if self.position.x > bounding_box_max.x {
            self.position.x = bounding_box_max.x;
        }

        if self.position.y < bounding_box_min.y {
            self.position.y = bounding_box_min.y;
        } else if self.position.y > bounding_box_max.y {
            self.position.y = bounding_box_max.y;
        }
    }

    fn is_neighbor(&self, other_particle: &Particle, h: f32) -> bool {
        let dist_sq = (self.position - other_particle.position).norm_squared();
        dist_sq < h * h
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

    pub fn insert_particle(&mut self, particle: &Particle, index: usize) {
      let ix = ((particle.position.x - self.bounding_box_min.x) / self.cell_size)
                        .floor() as usize;
      let iy = ((particle.position.y - self.bounding_box_min.y) / self.cell_size)
                        .floor() as usize;
      let iz = ((particle.position.z - self.bounding_box_min.z) / self.cell_size)
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

    pub fn find_neighbors(&self, particle: &Particle) -> Vec<usize> {
        let mut neighbors = Vec::new();
        let ix = ((particle.position.x - self.bounding_box_min.x) / self.cell_size)
                            .floor() as usize;
        let iy = ((particle.position.y - self.bounding_box_min.y) / self.cell_size)
                            .floor() as usize;
        let iz = ((particle.position.z - self.bounding_box_min.z) / self.cell_size)
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
    #[wasm_bindgen(constructor)]
    pub fn new(bounding_box_dim: f32) -> Simulation {
        let particles = vec![Particle::default() ; NUM_PARTICLES];
        let predictions = vec![Particle::default() ; NUM_PARTICLES];

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

        let mut simulation = Simulation {
            num_particles: NUM_PARTICLES,
            particles,
            predictions,
            bounding_box_dim,
            gravity: -9.81,
            last_update_time: None,
            grid,
            rest_density: 1e9,
            stiffness: 0.5,
            viscosity: 0.1,
        };
        simulation.reset_particles();

        simulation
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
        let particle_i = &self.predictions[particle_i_index];
        let neighbors_i = &all_neighbors[particle_i_index];
        let cutoff = self.grid.cell_size * 0.1;

        neighbors_i
            .iter()
            .map(|&neighbor_j_index| {
                let neighbor_j = &self.predictions[neighbor_j_index];
                let r = particle_i.position - neighbor_j.position;
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
        let particle_i = &self.predictions[particle_i_index];

        // Case 1: grad(C_i) w.r.t. itself, so a movement affects all neighbors
        if particle_i_index == particle_k_index {
            let neighbors_of_i = &all_neighbors[particle_i_index];
            let sum_vec: Vec3 = neighbors_of_i
                .iter()
                .map(|&j_index| {
                    let dist_ij = particle_i.position - self.predictions[j_index].position;
                    self.kernel_spiky_grad(dist_ij, cutoff)
                })
                .sum();
            sum_vec.scalar_mul(1.0 / self.rest_density)

        // Case 2: grad(C_i) w.r.t. another k; if k is not a neighbor, then
        // the gradient must be 0.
        } else {
            let particle_k = &self.predictions[particle_k_index];
            if particle_i.is_neighbor(particle_k, cutoff) {
                let dist_ik = particle_i.position - particle_k.position;
                -self
                    .kernel_spiky_grad(dist_ik, cutoff)
                    .scalar_mul(1.0 / self.rest_density)
            } else {
                Vec3::default()
            }
        }
    }

    pub fn update(&mut self) {
        let now = instant::Instant::now();
        let mut dt = match self.last_update_time {
            Some(last_time) => now.duration_since(last_time)
                                           .as_secs_f32(),
            None => 1.0 / 60.0,
        };
        dt = dt.min(1.0 / 30.0);
        self.last_update_time = Some(now);

        // 1. Apply forces and populate predictions
        for i in 0..self.num_particles {
            self.particles[i].velocity.y += self.gravity * dt;
            let mut prediction = self.particles[i].clone();
            prediction.position += prediction.velocity.scalar_mul(dt);
            self.predictions[i] = prediction;
        }

        // 2. Populate grid with predictions
        self.grid.clear();
        for i in 0..self.num_particles {
            self.grid.insert_particle(&self.predictions[i], i);
        }

        for _ in 0..NUM_ITERATIONS {
            // 3. Solver iteration |> calculate Laplace coefficients

            // Cache neighbors at the beginning of the iterations loop.
            let mut all_neighbors: Vec<Vec<usize>> = vec![Vec::new(); self.num_particles];
            for i in 0..self.num_particles {
                all_neighbors[i] = self.grid.find_neighbors(&self.predictions[i]);
            }

            let mut lambdas = vec![0.0; self.num_particles];
            lambdas.par_iter_mut().enumerate().for_each(|(i, lambda)| {
                let density_i = self.calculate_density(i, &all_neighbors);
                let constraint_i = (density_i / self.rest_density) - 1.0;
                let sum_grad_constraints: f32 = all_neighbors[i]
                    .iter()
                    .map(|&k| {
                       self.calculate_grad_constraint(i, k, &all_neighbors)
                            .norm_squared()
                    })
                    .sum();
                *lambda = -constraint_i / (sum_grad_constraints + RELAXATION);

            });

            // 4. Solver iteration |> calculate corrections and handle collisions
            let mut delta_ps = vec![Vec3::default(); self.num_particles];
            delta_ps.par_iter_mut().enumerate().for_each(|(i, delta_p)| {
                let prediction_i = self.predictions[i];
                let neighbors_i = &all_neighbors[i];

                *delta_p = neighbors_i
                    .iter()
                    .map(|&j| {
                        let prediction_j = self.predictions[j];
                        let cutoff = self.grid.cell_size;
                        let dist_ij = prediction_i.position - prediction_j.position;
                        let scalar = lambdas[i] + lambdas[j];
                        self.kernel_spiky_grad(dist_ij, cutoff)
                            .scalar_mul(scalar)
                    })
                    .sum::<Vec3>()
                    .scalar_mul(1.0 / self.rest_density);
            });

            self.predictions.par_iter_mut().for_each(|prediction_i| {
                prediction_i.handle_collision(&self.grid.bounding_box_min,
                                              &self.grid.bounding_box_max);
            });

            // 5. Solver iteration |> update predictions
            self.predictions.iter_mut().enumerate().for_each(|(i, prediction_i)| {
                prediction_i.position += delta_ps[i];
            });
        }

        // 6. Update true velocities and positions
        self.particles.iter_mut().enumerate().for_each(|(i, particle_i)| {
            particle_i.velocity = (self.predictions[i].position - particle_i.position)
                                            .scalar_mul(1.0 / dt);
            particle_i.position = self.predictions[i].position;
        });
    }

    // Used by frontend to load particle positions into the scene.
    pub fn get_particle_positions(&self) -> Vec<f32> {
        let mut positions = Vec::with_capacity(self.particles.len() * 3);
        for particle in &self.particles {
            positions.push(particle.position.x);
            positions.push(particle.position.y);
            positions.push(particle.position.z);
        }
        positions
    }

    // Used by frontend as a manual "reset" button.
    pub fn reset_particles(&mut self) {
        let y_offset = 2.0;

        for particle in &mut self.particles {
            let x = (js_sys::Math::random() as f32 - 0.5)
                            * self.bounding_box_dim / 2.0;
            let y = (js_sys::Math::random() as f32 - 0.5)
                            * self.bounding_box_dim / 2.0 + y_offset;
            let z = (js_sys::Math::random() as f32 - 0.5)
                            * self.bounding_box_dim / 2.0;
            particle.position = Vec3::new(x, y, z);
            particle.velocity = Vec3::default();
        }

        self.predictions = self.particles.clone();
    }
}
