import numpy as np
import wgpu
from wgpu.utils.device import get_default_device
from PIL import Image
import json
import struct
from pathlib import Path
import time
from numba import jit, prange

# Rich imports
from rich.console import Console
from rich.progress import Progress, SpinnerColumn, BarColumn, TextColumn, TimeRemainingColumn, TimeElapsedColumn

console = Console()

# Constants
WG_SIZE_XY = 8
WG_SIZE_SEEDS = 256
OUTPUT_RESOLUTION = 512  # Can be adjusted
GRID_SIZE = 128  # Source image will be resized to this

class CellBody:
    """Physics body for each particle"""
    def __init__(self, srcx, srcy, dstx, dsty, dst_force=0.14):
        self.srcx = srcx
        self.srcy = srcy
        self.dstx = dstx
        self.dsty = dsty
        self.velx = 0.0
        self.vely = 0.0
        self.accx = 0.0
        self.accy = 0.0
        self.dst_force = dst_force
        self.age = 0
        self.stroke_id = 0

    def update(self, pos, idx):
        """Apply physics integration"""
        self.velx += self.accx
        self.vely += self.accy
        
        self.accx = 0.0
        self.accy = 0.0
        
        self.velx *= 0.97  # Damping
        self.vely *= 0.97
        
        # Clamp velocity
        MAX_VEL = 6.0
        self.velx = np.clip(self.velx, -MAX_VEL, MAX_VEL)
        self.vely = np.clip(self.vely, -MAX_VEL, MAX_VEL)
        
        pos[idx, 0] += self.velx
        pos[idx, 1] += self.vely
        
        self.age += 1

    def apply_dst_force(self, pos, idx, sidelen):
        """Pull toward destination"""
        elapsed = self.age / 60.0
        factor = min((elapsed * self.dst_force) ** 3, 10.0) if self.dst_force > 0 else 0.1
        
        dx = self.dstx - pos[idx, 0]
        dy = self.dsty - pos[idx, 1]
        dist = np.sqrt(dx*dx + dy*dy)
        
        self.accx += (dx * dist * factor) / sidelen
        self.accy += (dy * dist * factor) / sidelen

    def apply_neighbour_force(self, pos, idx, other_pos, pixel_size):
        """Repulsion from neighbors"""
        PERSONAL_SPACE = 0.95
        
        dx = other_pos[0] - pos[idx, 0]
        dy = other_pos[1] - pos[idx, 1]
        dist = np.sqrt(dx*dx + dy*dy)
        personal_space = pixel_size * PERSONAL_SPACE
        
        if dist > 0 and dist < personal_space:
            weight = (1.0 / dist) * (personal_space - dist) / personal_space
            self.accx -= dx * weight
            self.accy -= dy * weight
            return weight
        elif dist < 1e-6:
            self.accx += (np.random.random() - 0.5) * 0.1
            self.accy += (np.random.random() - 0.5) * 0.1
            return 1.0
        return 0.0

    def apply_wall_force(self, pos, idx, sidelen, pixel_size):
        """Keep particles within bounds"""
        personal_space = pixel_size * 0.95 * 0.5
        
        if pos[idx, 0] < personal_space:
            self.accx += (personal_space - pos[idx, 0]) / personal_space
        elif pos[idx, 0] > sidelen - personal_space:
            self.accx -= (pos[idx, 0] - (sidelen - personal_space)) / personal_space
            
        if pos[idx, 1] < personal_space:
            self.accy += (personal_space - pos[idx, 1]) / personal_space
        elif pos[idx, 1] > sidelen - personal_space:
            self.accy -= (pos[idx, 1] - (sidelen - personal_space)) / personal_space


@jit(nopython=True, parallel=False, fastmath=True) 
def build_spatial_grid(positions, pixel_size, grid_size):
    """Build efficient spatial grid for neighbor queries"""
    n_particles = positions.shape[0]
    
    # Calculate grid cell for each particle
    grid_indices = np.empty((n_particles, 2), dtype=np.int32)
    for i in range(n_particles):
        grid_indices[i, 0] = min(max(int(positions[i, 0] / pixel_size), 0), grid_size - 1)
        grid_indices[i, 1] = min(max(int(positions[i, 1] / pixel_size), 0), grid_size - 1)
    
    # Count particles per cell
    cell_counts = np.zeros(grid_size * grid_size, dtype=np.int32)
    for i in range(n_particles):
        cell_id = grid_indices[i, 1] * grid_size + grid_indices[i, 0]
        cell_counts[cell_id] += 1
    
    # Calculate start indices for each cell
    cell_starts = np.zeros(grid_size * grid_size + 1, dtype=np.int32)
    for i in range(grid_size * grid_size):
        cell_starts[i + 1] = cell_starts[i] + cell_counts[i]
    
    # Fill grid with particle indices
    grid_particles = np.zeros(n_particles, dtype=np.int32)
    cell_offsets = np.zeros(grid_size * grid_size, dtype=np.int32)
    
    for i in range(n_particles):
        cell_id = grid_indices[i, 1] * grid_size + grid_indices[i, 0]
        offset = cell_starts[cell_id] + cell_offsets[cell_id]
        grid_particles[offset] = i
        cell_offsets[cell_id] += 1
    
    return grid_indices, cell_starts, grid_particles


@jit(nopython=True, parallel=True, fastmath=True)
def update_physics_optimized(positions, velocities, accelerations, destinations, ages, 
                             output_res, pixel_size, grid_size, dst_forces,
                             grid_indices, cell_starts, grid_particles):
    """Optimized physics with proper spatial grid"""
    n_particles = positions.shape[0]
    
    ALIGNMENT_FACTOR = 0.7
    MAX_VEL = 6.0
    DAMPING = 0.97
    PERSONAL_SPACE = pixel_size * 0.95
    personal_space_half = PERSONAL_SPACE * 0.5
    personal_space_sq = PERSONAL_SPACE * PERSONAL_SPACE
    
    # Reset accelerations
    accelerations[:] = 0.0
    
    # Apply forces for each particle
    for i in prange(n_particles):
        pos_x = positions[i, 0]
        pos_y = positions[i, 1]
        vel_x = velocities[i, 0]
        vel_y = velocities[i, 1]
        
        # Wall forces
        if pos_x < personal_space_half:
            accelerations[i, 0] += (personal_space_half - pos_x) / personal_space_half
        elif pos_x > output_res - personal_space_half:
            accelerations[i, 0] -= (pos_x - (output_res - personal_space_half)) / personal_space_half
            
        if pos_y < personal_space_half:
            accelerations[i, 1] += (personal_space_half - pos_y) / personal_space_half
        elif pos_y > output_res - personal_space_half:
            accelerations[i, 1] -= (pos_y - (output_res - personal_space_half)) / personal_space_half
        
        # Destination force
        elapsed = ages[i] / 60.0
        factor = 0.1 if dst_forces[i] <= 0 else min((elapsed * dst_forces[i]) ** 3, 10.0)
        
        dx = destinations[i, 0] - pos_x
        dy = destinations[i, 1] - pos_y
        dist = np.sqrt(dx*dx + dy*dy)
        
        accelerations[i, 0] += (dx * dist * factor) / output_res
        accelerations[i, 1] += (dy * dist * factor) / output_res
        
        # Neighbor interactions using spatial grid
        col = grid_indices[i, 0]
        row = grid_indices[i, 1]
        
        avg_velx = 0.0
        avg_vely = 0.0
        count = 0.0
        
        # Check 3x3 neighborhood
        for nrow in range(max(0, row - 1), min(grid_size, row + 2)):
            for ncol in range(max(0, col - 1), min(grid_size, col + 2)):
                cell_id = nrow * grid_size + ncol
                start = cell_starts[cell_id]
                end = cell_starts[cell_id + 1]
                
                for idx in range(start, end):
                    j = grid_particles[idx]
                    if i == j:
                        continue
                    
                    other_pos_x = positions[j, 0]
                    other_pos_y = positions[j, 1]
                    dx = other_pos_x - pos_x
                    dy = other_pos_y - pos_y
                    dist_sq = dx*dx + dy*dy
                    
                    if dist_sq < 1e-12:
                        accelerations[i, 0] += (np.random.random() - 0.5) * 0.1
                        accelerations[i, 1] += (np.random.random() - 0.5) * 0.1
                        weight = 1.0
                    elif dist_sq < personal_space_sq:
                        dist = np.sqrt(dist_sq)
                        weight = (1.0 / dist) * (PERSONAL_SPACE - dist) / PERSONAL_SPACE
                        accelerations[i, 0] -= dx * weight
                        accelerations[i, 1] -= dy * weight
                    else:
                        continue
                    
                    # Accumulate velocity alignment
                    avg_velx += velocities[j, 0] * weight
                    avg_vely += velocities[j, 1] * weight
                    count += weight
        
        # Apply velocity alignment
        if count > 0:
            avg_velx /= count
            avg_vely /= count
            accelerations[i, 0] += (avg_velx - vel_x) * ALIGNMENT_FACTOR
            accelerations[i, 1] += (avg_vely - vel_y) * ALIGNMENT_FACTOR
    
    # Integration
    for i in prange(n_particles):
        velocities[i, 0] += accelerations[i, 0]
        velocities[i, 1] += accelerations[i, 1]
        
        velocities[i, 0] *= DAMPING
        velocities[i, 1] *= DAMPING
        
        velocities[i, 0] = min(max(velocities[i, 0], -MAX_VEL), MAX_VEL)
        velocities[i, 1] = min(max(velocities[i, 1], -MAX_VEL), MAX_VEL)
        
        positions[i, 0] += velocities[i, 0]
        positions[i, 1] += velocities[i, 1]
        
        ages[i] += 1


class VoronoiSimulation:
    """Main simulation class"""
    
    def __init__(self, source_image_path, assignments_path, output_res=OUTPUT_RESOLUTION, grid_size=GRID_SIZE):
        self.output_res = output_res
        self.grid_size = grid_size
        self.pixel_size = output_res / grid_size
        
        # Load and process image
        img = Image.open(source_image_path).convert('RGB')
        img = img.resize((grid_size, grid_size), Image.Resampling.LANCZOS)
        img_array = np.array(img, dtype=np.float32) / 255.0
        
        # Load assignments
        with open(assignments_path, 'r') as f:
            assignments = json.load(f)
        
        # Initialize particles
        n_particles = grid_size * grid_size
        self.positions = np.zeros((n_particles, 2), dtype=np.float32)
        self.velocities = np.zeros((n_particles, 2), dtype=np.float32)
        self.accelerations = np.zeros((n_particles, 2), dtype=np.float32)
        self.destinations = np.zeros((n_particles, 2), dtype=np.float32)
        self.ages = np.zeros(n_particles, dtype=np.int32)
        self.dst_forces = np.full(n_particles, 0.14, dtype=np.float32)
        self.colors = np.zeros((n_particles, 4), dtype=np.float32)
        self.cells = []
        
        # Create cells from image
        for y in range(grid_size):
            for x in range(grid_size):
                idx = y * grid_size + x
                color = img_array[y, x]
                self.colors[idx] = [color[0], color[1], color[2], 1.0]
                self.positions[idx] = [(x + 0.5) * self.pixel_size, (y + 0.5) * self.pixel_size]
                self.cells.append(CellBody(
                    (x + 0.5) * self.pixel_size,
                    (y + 0.5) * self.pixel_size,
                    0, 0, 0.14  # dst will be set by assignments
                ))
        
        # Apply assignments
        self.set_assignments(assignments)
        
        # Initialize WGPU
        self.device = get_default_device()
        self._init_gpu_resources()
    
    def set_assignments(self, assignments):
        """Set destination positions from assignments"""
        for dst_idx, src_idx in enumerate(assignments):
            dst_x = dst_idx % self.grid_size
            dst_y = dst_idx // self.grid_size
            
            self.destinations[src_idx, 0] = (dst_x + 0.5) * self.pixel_size
            self.destinations[src_idx, 1] = (dst_y + 0.5) * self.pixel_size
            self.ages[src_idx] = 0
            
            # Also update legacy CellBody for compatibility
            self.cells[src_idx].dstx = self.destinations[src_idx, 0]
            self.cells[src_idx].dsty = self.destinations[src_idx, 1]
            self.cells[src_idx].age = 0
    
    def _init_gpu_resources(self):
        """Initialize GPU buffers, textures, and pipelines"""
        n_particles = len(self.cells)
        
        # Buffers
        self.seed_buffer = self.device.create_buffer_with_data(
            data=self.positions.tobytes(),
            usage=wgpu.BufferUsage.STORAGE | wgpu.BufferUsage.COPY_DST
        )
        
        self.color_buffer = self.device.create_buffer_with_data(
            data=self.colors.tobytes(),
            usage=wgpu.BufferUsage.STORAGE
        )
        
        params_common = struct.pack('IIII', self.output_res, self.output_res, n_particles, 0)
        self.params_common_buffer = self.device.create_buffer_with_data(
            data=params_common,
            usage=wgpu.BufferUsage.UNIFORM
        )
        
        params_jfa = struct.pack('IIII', self.output_res, self.output_res, 1, 0)
        self.params_jfa_buffer = self.device.create_buffer_with_data(
            data=params_jfa,
            usage=wgpu.BufferUsage.UNIFORM | wgpu.BufferUsage.COPY_DST
        )
        
        # Textures
        self.ids_a = self.device.create_texture(
            size=(self.output_res, self.output_res, 1),
            usage=wgpu.TextureUsage.STORAGE_BINDING | wgpu.TextureUsage.TEXTURE_BINDING,
            format=wgpu.TextureFormat.r32uint
        )
        self.ids_b = self.device.create_texture(
            size=(self.output_res, self.output_res, 1),
            usage=wgpu.TextureUsage.STORAGE_BINDING | wgpu.TextureUsage.TEXTURE_BINDING,
            format=wgpu.TextureFormat.r32uint
        )
        self.color_texture = self.device.create_texture(
            size=(self.output_res, self.output_res, 1),
            usage=wgpu.TextureUsage.STORAGE_BINDING | wgpu.TextureUsage.COPY_SRC,
            format=wgpu.TextureFormat.rgba8unorm
        )
        
        # Create pipelines
        self._create_pipelines()
    
    def _create_pipelines(self):
        """Create compute pipelines"""
        
        # Clear shader
        clear_shader = """
@group(0) @binding(0) var dst_ids: texture_storage_2d<r32uint, write>;

@compute @workgroup_size(8,8,1)
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
    let size = textureDimensions(dst_ids);
    if (gid.x >= size.x || gid.y >= size.y) { return; }
    textureStore(dst_ids, vec2<i32>(gid.xy), vec4<u32>(0xFFFFFFFFu, 0u, 0u, 0u));
}
"""
        
        # Seed splat shader
        seed_shader = """
struct Seeds { pos: array<vec2<f32>> };
@group(0) @binding(0) var<storage, read> seeds: Seeds;

struct ParamsCommon { width: u32, height: u32, n_seeds: u32, _pad: u32 };
@group(0) @binding(1) var<uniform> params: ParamsCommon;

@group(0) @binding(2) var dst_ids: texture_storage_2d<r32uint, write>;

@compute @workgroup_size(256,1,1)
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
    let i = gid.x;
    if (i >= params.n_seeds) { return; }
    let p = seeds.pos[i];
    let x = i32(round(p.x));
    let y = i32(round(p.y));
    textureStore(dst_ids, vec2<i32>(x,y), vec4<u32>(i, 0u, 0u, 0u));
}
"""
        
        # JFA shader
        jfa_shader = """
struct Seeds { pos: array<vec2<f32>> };
@group(0) @binding(0) var<storage, read> seeds: Seeds;

@group(0) @binding(1) var src_ids: texture_2d<u32>;
@group(0) @binding(2) var dst_ids: texture_storage_2d<r32uint, write>;

struct JfaParams { width: u32, height: u32, step: u32, _pad: u32 };
@group(0) @binding(3) var<uniform> params: JfaParams;

fn dist2(a: vec2<f32>, b: vec2<f32>) -> f32 { let d = a - b; return dot(d,d); }

@compute @workgroup_size(8,8,1)
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
    if (gid.x >= params.width || gid.y >= params.height) { return; }
    
    let p = vec2<f32>(f32(gid.x), f32(gid.y));
    var best_id: u32 = textureLoad(src_ids, vec2<i32>(gid.xy), 0).r;
    var best_d2: f32 = 3.4e38;
    if (best_id != 0xFFFFFFFFu) { best_d2 = dist2(p, seeds.pos[best_id]); }
    
    let s = i32(params.step);
    let offs = array<vec2<i32>,8>(
        vec2<i32>( s, 0), vec2<i32>(-s, 0), vec2<i32>( 0, s), vec2<i32>( 0,-s),
        vec2<i32>( s, s), vec2<i32>( s,-s), vec2<i32>(-s, s), vec2<i32>(-s,-s)
    );
    
    for (var i = 0; i < 8; i = i + 1) {
        let q = vec2<i32>(gid.xy) + offs[i];
        if (q.x < 0 || q.y < 0 || q.x >= i32(params.width) || q.y >= i32(params.height)) {
            continue;
        }
        
        let cand = textureLoad(src_ids, q, 0).r;
        if (cand != 0xFFFFFFFFu) {
            let d2 = dist2(p, seeds.pos[cand]);
            if (d2 < best_d2) { best_d2 = d2; best_id = cand; }
        }
    }
    
    textureStore(dst_ids, vec2<i32>(gid.xy), vec4<u32>(best_id, 0u, 0u, 0u));
}
"""
        
        # Shade shader
        shade_shader = """
@group(0) @binding(0) var ids: texture_2d<u32>;
@group(0) @binding(1) var out_color: texture_storage_2d<rgba8unorm, write>;

struct Seeds { pos: array<vec2<f32>> };
@group(0) @binding(2) var<storage, read> seeds: Seeds;

struct Colors { rgba: array<vec4<f32>> };
@group(0) @binding(3) var<storage, read> colors: Colors;

@compute @workgroup_size(8,8,1)
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
    let size = textureDimensions(out_color);
    if (gid.x >= size.x || gid.y >= size.y) { return; }
    let id = textureLoad(ids, vec2<i32>(gid.xy), 0).r;
    
    var rgba: vec4<f32>;
    if (id == 0xFFFFFFFFu) {
        rgba = vec4<f32>(0.0, 0.0, 0.0, 1.0);
    } else {
        rgba = colors.rgba[id];
    }
    textureStore(out_color, vec2<i32>(gid.xy), rgba);
}
"""
        
        # Create shader modules
        clear_module = self.device.create_shader_module(code=clear_shader)
        seed_module = self.device.create_shader_module(code=seed_shader)
        jfa_module = self.device.create_shader_module(code=jfa_shader)
        shade_module = self.device.create_shader_module(code=shade_shader)
        
        # Create bind group layouts and pipelines
        # Clear pipeline
        clear_bgl = self.device.create_bind_group_layout(entries=[
            {"binding": 0, "visibility": wgpu.ShaderStage.COMPUTE,
             "storage_texture": {"access": wgpu.StorageTextureAccess.write_only,
                                "format": wgpu.TextureFormat.r32uint}}
        ])
        self.clear_pipeline = self.device.create_compute_pipeline(
            layout=self.device.create_pipeline_layout(bind_group_layouts=[clear_bgl]),
            compute={"module": clear_module, "entry_point": "main"}
        )
        
        # Seed pipeline
        seed_bgl = self.device.create_bind_group_layout(entries=[
            {"binding": 0, "visibility": wgpu.ShaderStage.COMPUTE,
             "buffer": {"type": wgpu.BufferBindingType.read_only_storage}},
            {"binding": 1, "visibility": wgpu.ShaderStage.COMPUTE,
             "buffer": {"type": wgpu.BufferBindingType.uniform}},
            {"binding": 2, "visibility": wgpu.ShaderStage.COMPUTE,
             "storage_texture": {"access": wgpu.StorageTextureAccess.write_only,
                                "format": wgpu.TextureFormat.r32uint}}
        ])
        self.seed_pipeline = self.device.create_compute_pipeline(
            layout=self.device.create_pipeline_layout(bind_group_layouts=[seed_bgl]),
            compute={"module": seed_module, "entry_point": "main"}
        )
        
        # JFA pipeline
        jfa_bgl = self.device.create_bind_group_layout(entries=[
            {"binding": 0, "visibility": wgpu.ShaderStage.COMPUTE,
             "buffer": {"type": wgpu.BufferBindingType.read_only_storage}},
            {"binding": 1, "visibility": wgpu.ShaderStage.COMPUTE,
             "texture": {"sample_type": wgpu.TextureSampleType.uint}},
            {"binding": 2, "visibility": wgpu.ShaderStage.COMPUTE,
             "storage_texture": {"access": wgpu.StorageTextureAccess.write_only,
                                "format": wgpu.TextureFormat.r32uint}},
            {"binding": 3, "visibility": wgpu.ShaderStage.COMPUTE,
             "buffer": {"type": wgpu.BufferBindingType.uniform}}
        ])
        self.jfa_pipeline = self.device.create_compute_pipeline(
            layout=self.device.create_pipeline_layout(bind_group_layouts=[jfa_bgl]),
            compute={"module": jfa_module, "entry_point": "main"}
        )
        
        # Shade pipeline
        shade_bgl = self.device.create_bind_group_layout(entries=[
            {"binding": 0, "visibility": wgpu.ShaderStage.COMPUTE,
             "texture": {"sample_type": wgpu.TextureSampleType.uint}},
            {"binding": 1, "visibility": wgpu.ShaderStage.COMPUTE,
             "storage_texture": {"access": wgpu.StorageTextureAccess.write_only,
                                "format": wgpu.TextureFormat.rgba8unorm}},
            {"binding": 2, "visibility": wgpu.ShaderStage.COMPUTE,
             "buffer": {"type": wgpu.BufferBindingType.read_only_storage}},
            {"binding": 3, "visibility": wgpu.ShaderStage.COMPUTE,
             "buffer": {"type": wgpu.BufferBindingType.read_only_storage}}
        ])
        self.shade_pipeline = self.device.create_compute_pipeline(
            layout=self.device.create_pipeline_layout(bind_group_layouts=[shade_bgl]),
            compute={"module": shade_module, "entry_point": "main"}
        )
        
        # Create bind groups
        self.clear_bg_a = self.device.create_bind_group(
            layout=clear_bgl,
            entries=[{"binding": 0, "resource": self.ids_a.create_view()}]
        )
        self.clear_bg_b = self.device.create_bind_group(
            layout=clear_bgl,
            entries=[{"binding": 0, "resource": self.ids_b.create_view()}]
        )
        
        self.seed_bg = self.device.create_bind_group(
            layout=seed_bgl,
            entries=[
                {"binding": 0, "resource": {"buffer": self.seed_buffer, "size": self.seed_buffer.size}},
                {"binding": 1, "resource": {"buffer": self.params_common_buffer, "size": self.params_common_buffer.size}},
                {"binding": 2, "resource": self.ids_a.create_view()}
            ]
        )
        
        self.jfa_bg_a_to_b = self.device.create_bind_group(
            layout=jfa_bgl,
            entries=[
                {"binding": 0, "resource": {"buffer": self.seed_buffer, "size": self.seed_buffer.size}},
                {"binding": 1, "resource": self.ids_a.create_view()},
                {"binding": 2, "resource": self.ids_b.create_view()},
                {"binding": 3, "resource": {"buffer": self.params_jfa_buffer, "size": self.params_jfa_buffer.size}}
            ]
        )
        
        self.jfa_bg_b_to_a = self.device.create_bind_group(
            layout=jfa_bgl,
            entries=[
                {"binding": 0, "resource": {"buffer": self.seed_buffer, "size": self.seed_buffer.size}},
                {"binding": 1, "resource": self.ids_b.create_view()},
                {"binding": 2, "resource": self.ids_a.create_view()},
                {"binding": 3, "resource": {"buffer": self.params_jfa_buffer, "size": self.params_jfa_buffer.size}}
            ]
        )
        
        self.shade_bg = self.device.create_bind_group(
            layout=shade_bgl,
            entries=[
                {"binding": 0, "resource": self.ids_a.create_view()},
                {"binding": 1, "resource": self.color_texture.create_view()},
                {"binding": 2, "resource": {"buffer": self.seed_buffer, "size": self.seed_buffer.size}},
                {"binding": 3, "resource": {"buffer": self.color_buffer, "size": self.color_buffer.size}}
            ]
        )
        
        self.shade_bg_b = self.device.create_bind_group(
            layout=shade_bgl,
            entries=[
                {"binding": 0, "resource": self.ids_b.create_view()},
                {"binding": 1, "resource": self.color_texture.create_view()},
                {"binding": 2, "resource": {"buffer": self.seed_buffer, "size": self.seed_buffer.size}},
                {"binding": 3, "resource": {"buffer": self.color_buffer, "size": self.color_buffer.size}}
            ]
        )
    
    def update_physics(self):
        """Run one step of physics simulation"""
        # Build spatial grid 
        grid_indices, cell_starts, grid_particles = build_spatial_grid(
            self.positions, self.pixel_size, self.grid_size
        )
        
        # Run physics with proper grid
        update_physics_optimized(
            self.positions, self.velocities, self.accelerations,
            self.destinations, self.ages, self.output_res,
            self.pixel_size, self.grid_size, self.dst_forces,
            grid_indices, cell_starts, grid_particles
        )

    def render_frame(self):
        """Render current frame using GPU"""
        # Update seed buffer
        self.device.queue.write_buffer(self.seed_buffer, 0, self.positions.tobytes())
        
        encoder = self.device.create_command_encoder()
        
        # Clear textures
        cpass = encoder.begin_compute_pass()
        cpass.set_pipeline(self.clear_pipeline)
        cpass.set_bind_group(0, self.clear_bg_a)
        cpass.dispatch_workgroups((self.output_res + 7) // 8, (self.output_res + 7) // 8, 1)
        cpass.end()
        
        cpass = encoder.begin_compute_pass()
        cpass.set_pipeline(self.clear_pipeline)
        cpass.set_bind_group(0, self.clear_bg_b)
        cpass.dispatch_workgroups((self.output_res + 7) // 8, (self.output_res + 7) // 8, 1)
        cpass.end()
        
        # Seed splat
        cpass = encoder.begin_compute_pass()
        cpass.set_pipeline(self.seed_pipeline)
        cpass.set_bind_group(0, self.seed_bg)
        cpass.dispatch_workgroups((len(self.cells) + 255) // 256, 1, 1)
        cpass.end()
        
        # JFA passes
        max_dim = max(self.output_res, self.output_res)
        step = 1
        while step < max_dim:
            step <<= 1
        step >>= 1
        
        flip = False
        while step >= 1:
            params_data = struct.pack('IIII', self.output_res, self.output_res, step, 0)
            self.device.queue.write_buffer(self.params_jfa_buffer, 0, params_data)
            
            cpass = encoder.begin_compute_pass()
            cpass.set_pipeline(self.jfa_pipeline)
            cpass.set_bind_group(0, self.jfa_bg_b_to_a if flip else self.jfa_bg_a_to_b)
            cpass.dispatch_workgroups((self.output_res + 7) // 8, (self.output_res + 7) // 8, 1)
            cpass.end()
            
            flip = not flip
            step >>= 1
        
        # Shade
        cpass = encoder.begin_compute_pass()
        cpass.set_pipeline(self.shade_pipeline)
        cpass.set_bind_group(0, self.shade_bg_b if flip else self.shade_bg)
        cpass.dispatch_workgroups((self.output_res + 7) // 8, (self.output_res + 7) // 8, 1)
        cpass.end()
        
        self.device.queue.submit([encoder.finish()])
    
    def get_frame_image(self):
        """Read back rendered frame as PIL Image"""
        # Create staging buffer
        bytes_per_row = self.output_res * 4
        aligned_bytes_per_row = (bytes_per_row + 255) & ~255  # Align to 256
        buffer_size = aligned_bytes_per_row * self.output_res
        
        staging_buffer = self.device.create_buffer(
            size=buffer_size,
            usage=wgpu.BufferUsage.COPY_DST | wgpu.BufferUsage.MAP_READ
        )
        
        encoder = self.device.create_command_encoder()
        encoder.copy_texture_to_buffer(
            {"texture": self.color_texture},
            {"buffer": staging_buffer, "bytes_per_row": aligned_bytes_per_row},
            (self.output_res, self.output_res, 1)
        )
        self.device.queue.submit([encoder.finish()])
        
        # Map and read data
        staging_buffer.map_sync(mode=wgpu.MapMode.READ)
        data = staging_buffer.read_mapped()
        staging_buffer.unmap()
        
        # Remove padding
        rgba = np.frombuffer(data, dtype=np.uint8).reshape(self.output_res, aligned_bytes_per_row)
        rgba = rgba[:, :self.output_res * 4].reshape(self.output_res, self.output_res, 4)
        
        return Image.fromarray(rgba)
    
    def animate(self, total_frames=140, gif_fps=8, output_path='animation.gif'):
        """Generate animation and save as GIF"""
        frames = []
        
        # Physics runs at 60fps internally
        physics_fps = 60
        capture_interval = physics_fps // gif_fps  # Capture every Nth physics frame
        total_physics_frames = total_frames * capture_interval  # Total physics steps needed
        
        # Timing accumulators
        physics_times = []
        render_times = []
        
        # Warmup: Run one iteration to trigger JIT compilation
        with console.status("[#FF8C00]Warming up (JIT compilation)...", spinner="dots"):
            self.update_physics()
            
        
        # Reset simulation state after warmup
        self.velocities[:] = 0.0
        self.accelerations[:] = 0.0
        self.ages[:] = 0

        with Progress(
            SpinnerColumn(),
            TextColumn("[progress.description]{task.description}"),
            BarColumn(),
            TextColumn("[progress.percentage]{task.percentage:>3.0f}%"),
            TextColumn("[#FF8C00]{task.fields[captured]}[/#FF8C00] frames captured"),
            TimeElapsedColumn(),
            TimeRemainingColumn(),
            console=console,
            transient=True
        ) as progress:
            task = progress.add_task(
                "[#FF8C00]Generating animation...",
                total=total_physics_frames,
                captured=0
            )
            
            for frame in range(total_physics_frames):
                # Physics update (always runs at 60fps)
                physics_start = time.perf_counter()
                self.update_physics()
                physics_times.append(time.perf_counter() - physics_start)
                
                
                # Render and capture frame at gif_fps rate
                if frame % capture_interval == 0:
                    render_start = time.perf_counter()
                    self.render_frame()
                    render_times.append(time.perf_counter() - render_start)
                    img = self.get_frame_image()
                    frames.append(img.convert('RGB'))
                    progress.update(task, advance=1, captured=len(frames))
                else:
                    progress.update(task, advance=1)
        
        # Save GIF
        if frames:
            with console.status("[#FF8C00]Saving GIF...", spinner="dots"):
                frames[0].save(
                    output_path,
                    save_all=True,
                    append_images=frames[1:],
                    duration=1000 // gif_fps,  
                    loop=0
                )

