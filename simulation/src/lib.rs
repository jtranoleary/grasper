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
use std::cmp;
use std::f32::consts::PI;
use wasm_bindgen::prelude::*;

#[wasm_bindgen]
#[derive(Clone, Copy)]
pub struct Vec3 {
    x: f32,
    y: f32,
    z: f32,
}

#[wasm_bindgen]
#[derive(Clone, Copy)]
pub struct Particle {
    x: f32,
    y: f32,
    z: f32,
    px: f32,
    py: f32,
    pz: f32,
    vx: f32,
    vy: f32,
    vz: f32,
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
    particles: Vec<Particle>,
    floor_y: f32,
    gravity: f32,
    last_update_time: Option<Instant>,
    grid: UniformGrid,
    rest_density: f32,
    stiffness: f32,
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

    pub fn insert_particle(&mut self, particle_index: usize, position: Vec3) {
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
      self.cells[cell_index].push(particle_index);
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

        for z in (cmp::min(iz - 1, 0))..=(cmp::min(iz + 1, self.num_z - 1)) {
            for y in (cmp::min(iy - 1, 0))..=(cmp::min(iy + 1, self.num_y - 1)) {
                for x in (cmp::min(ix - 1, 0))..=(cmp::min(ix + 1, self.num_x - 1)) {
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
    pub fn new() -> Simulation {
        let num_particles = 2000;
        let bounding_box_dim = 5.0;
        let y_offset = 2.0;
        let floor_y = 0.0;
        let mut particles = Vec::with_capacity(num_particles);

        let bounding_box_min = Vec3::new(
            -bounding_box_dim / 2.0,
            floor_y,
            -bounding_box_dim / 2.0
        );
        let bounding_box_max = Vec3::new(
            bounding_box_dim / 2.0,
            bounding_box_dim + floor_y,
            bounding_box_dim / 2.0
        );

        for _ in 0..num_particles {
            let x = (js_sys::Math::random() as f32 - 0.5)
                            * bounding_box_dim;
            let y = (js_sys::Math::random() as f32 - 0.5)
                            * bounding_box_dim + y_offset;
            let z = (js_sys::Math::random() as f32 - 0.5)
                            * bounding_box_dim;
            particles.push(Particle {
                x,
                y,
                z,
                px: x,
                py: y,
                pz: z,
                vx: 0.0,
                vy: 0.0,
                vz: 0.0,
            });
        }

        let cell_size = 0.5;
        let grid = UniformGrid::new(cell_size, bounding_box_min,
                                                 bounding_box_max);

        Simulation {
            particles,
            floor_y: 0.0,
            gravity: -9.81,
            last_update_time: None,
            grid,
            rest_density: 1000.0,
            stiffness: 1.0,
            viscosity: 0.1,
        }
    }

    fn kernel_poly6(&self, r_squared: f32, d: f32) -> f32 {
        let d_squared = d * d;
        if r_squared >= 0.0 && r_squared <= d_squared {
            let factor = 315.0 / (64.0 * PI * d.powi(9));
            return factor * (d_squared - r_squared).powi(3);
        }
        0.0
    }

    fn grad_kernel_poly6(&self, r: f32, d: f32) -> f32 {
        let d_squared = d * d;
        if r >= 0.0 && r <= d {
          let factor = -945.0 / (32.0 * PI * d.powi(9));
          return factor * (d_squared - r * r).powi(2) * r;
        }
        0.0
    }

    fn calculate_density(&self, particle_index: usize) -> f32 {
        let particle = &self.particles[particle_index];
        let position = Vec3::new(particle.x, particle.y, particle.z);
        let neighbors = self.grid.find_neighbors(position);
        let mut density = 0.0;
        let cutoff = self.grid.cell_size;

        for &neighbor_index in &neighbors {
            let neighbor = &self.particles[neighbor_index];
            let dx = neighbor.x - particle.x;
            let dy = neighbor.y - particle.y;
            let dz = neighbor.z - particle.z;
            let r_squared = dx * dx + dy * dy + dz * dz;
            density += self.kernel_poly6(r_squared, cutoff);
        }
        density
    }

    pub fn update(&mut self) {
        let now = instant::Instant::now();
        let dt = match self.last_update_time {
            Some(last_time) => now.duration_since(last_time)
                                           .as_secs_f32(),
            None => 1.0 / 60.0,
        };
        self.last_update_time = Some(now);

        self.grid.clear();

        // 1. Apply external forces and predict positions.
        for particle in &mut self.particles {
            particle.vy += self.gravity * dt;

            particle.px = particle.x;
            particle.py = particle.y;
            particle.pz = particle.z;

            particle.x += particle.vx * dt;
            particle.y += particle.vy * dt;
            particle.z += particle.vz * dt;
        }

        // 2. Re-insert particles into the grid (using predicted positions).
        for (index, particle) in self.particles.iter().enumerate() {
            self.grid.insert_particle(index,
                Vec3::new(particle.x, particle.y, particle.z));
        }

        // 3. Constraint projection
        let num_iterations = 4;
        for _ in 0..num_iterations {
            for i in 0..self.particles.len() {
                let density_i = self.calculate_density(i);
                let pressure_i = self.stiffness * (density_i -
                                    self.rest_density);

                let particle_i = &self.particles[i];
                let position_i = Vec3::new(
                    particle_i.x,
                    particle_i.y,
                    particle_i.z
                );
                let neighbors = self.grid.find_neighbors(position_i);

                let mut dx = 0.0;
                let mut dy = 0.0;
                let mut dz = 0.0;

                for &j in &neighbors {
                    if i == j { continue; }
                    let particle_j = &self.particles[j];

                    let nx = particle_j.x - particle_i.x;
                    let ny = particle_j.y - particle_i.y;
                    let nz = particle_j.z - particle_i.z;
                    let r = (nx * nx + ny * ny + nz * nz).sqrt();
                    let h = self.grid.cell_size;

                    if r < h && r > 1e-12 {
                        let grad_c_i = self.grad_kernel_poly6(r, h)
                                        * nx / r / self.rest_density;
                        let grad_c_j = self.grad_kernel_poly6(r, h)
                                        * ny / r / self.rest_density;
                        let grad_c_k = self.grad_kernel_poly6(r, h)
                                        * nz / r / self.rest_density;

                        let correction = pressure_i / self.rest_density;
                        dx -= grad_c_i * correction;
                        dy -= grad_c_j * correction;
                        dz -= grad_c_k * correction;
                    }
                }

                let particle_i = &mut self.particles[i];
                particle_i.x += dx;
                particle_i.y += dy;
                particle_i.z += dz;
            }
        }

        // 4. Update previous position and handle collisions (floor).
        for particle in &mut self.particles {
            particle.vx = (particle.x - particle.px) / dt;
            particle.vy = (particle.y - particle.py) / dt;
            particle.vz = (particle.z - particle.pz) / dt;

            if particle.y < self.floor_y {
                particle.y = self.floor_y;
                particle.py = 0.0;
            }
        }
    }

    pub fn get_particle_positions(&self) -> Vec<f32> {
        let mut positions = Vec::with_capacity(self.particles.len() * 3);
        for particle in &self.particles {
            positions.push(particle.x);
            positions.push(particle.y);
            positions.push(particle.z);
        }
        positions
    }

    pub fn reset_particles(&mut self) {
        let bounding_box_dim = 5.0;
        let y_offset = 2.0;

        for particle in &mut self.particles {
            let x = (js_sys::Math::random() as f32 - 0.5)
                            * bounding_box_dim;
            let y = (js_sys::Math::random() as f32 - 0.5)
                            * bounding_box_dim + y_offset;
            let z = (js_sys::Math::random() as f32 - 0.5)
                            * bounding_box_dim;
            particle.x = x;
            particle.y = y;
            particle.z = z;
            particle.px = x;
            particle.py = y;
            particle.pz = z;
            particle.vx = 0.0;
            particle.vy = 0.0;
            particle.vz = 0.0;
        }
    }
}