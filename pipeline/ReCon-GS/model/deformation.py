import math
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Tuple

import torch
from absl import logging
from jaxtyping import Float
from torch import Tensor, nn
import time

from ..utils import average_quaternions, batched_knn, batched_keops_knn, uniform_grid_sampling_optimized
    
class Grid(nn.Module):
    def __init__(
        self,
        size: Tuple[int, int, int],
        xyz_min: Float[Tensor, "3"],
        xyz_max: Float[Tensor, "3"],
    ):
        super().__init__()

        size = size if torch.is_tensor(size) else torch.tensor(size)
        base = torch.tensor([size[1] * size[2], size[2], 1])

        self.register_buffer("size", size)
        self.register_buffer("base", base)
        self.register_buffer("xyz_min", xyz_min)
        self.register_buffer("xyz_max", xyz_max)

        logging.info(f"Set grid min: {xyz_min.tolist()}, max: {xyz_max.tolist()}")

    def normalize(self, xyz: Float[Tensor, "n 3"], clamp: bool = True):
        xyz = (xyz - self.xyz_min) / (self.xyz_max - self.xyz_min)
        if clamp:
            return xyz.clamp(0, 1)
        return xyz

    def hash(self, xyz: Float[Tensor, "n 3"]) -> torch.Tensor:
        xyz_normed = self.normalize(xyz)
        index = ((xyz_normed * self.size - 0.5).clamp(0).int() * self.base).sum(dim=-1)
        return index
        
class Adaptive_Grid(nn.Module):
    def __init__(
        self,
        vertex_num: int,
        xyz: Float[Tensor, "n 3"]
    ):
        super().__init__()
        self.vertex_num = vertex_num
        t0 = time.time()
        self.vertex_idx = uniform_grid_sampling_optimized(xyz, self.vertex_num)
        t1 = time.time()
        self.vertex = xyz[self.vertex_idx]
        t2 = time.time()
        self.p2v = batched_keops_knn(xyz, self.vertex, 1)
        t3 = time.time()
        logging.info(f"Setup Adaptive_grids: {self.vertex_num}, total_time for fps: {t1-t0}, total time for knn: {t3-t2}")

    def reset_p2v(self, xyz: Float[Tensor, "n 3"]):
        t2 = time.time()
        self.p2v = batched_keops_knn(xyz, self.vertex, 1)
        t3 = time.time()
        logging.info(f"Reset p2v, Adaptive_grids: {self.vertex_num}, total time for knn: {t3-t2}")


@dataclass
class DeformConfig:
    quantile: float = 0.05
    lr: float = 0.0005
    num_stage1_steps: int = 100
    max_gs_per_grid: int = 5
    num_grid_levels: int = 3
    grid_level_ratio: int = 2
    momentum: Optional[float] = 0.6
    densify_interval: int = 40
    densify_grad_threshold: float = 1.5e-4
    opacity_threshold: float = 0.01
    grid_reset_interval: int = 4


class Deformation(nn.Module):
    config: DeformConfig
    grids: List[Grid]
    optimizer: Optional[torch.optim.Optimizer] = None

    def __init__(self, config: DeformConfig):
        super().__init__()
        self.config = config
        self.rotation_activation = torch.nn.functional.normalize
        self.grids = []
        self.grids_num = 0
        
    @property
    def get_grid_num(self) -> int:
        return self.grids_num
    
    @torch.no_grad()
    def create_grids(self, xyz: Float[Tensor, "n 3"]):
        q, max_gs_per_grid = self.config.quantile, self.config.max_gs_per_grid
        n = math.ceil((xyz.shape[0] / max_gs_per_grid))
        grids, level = [], self.config.num_grid_levels
        num_grids = 0
        while n > 0 and level > 0:
            num_grids += (n)
            grids.append(Adaptive_Grid(n, xyz).to(xyz.device))
            n, level = n // (self.config.grid_level_ratio), level - 1
        return grids, num_grids
    
    def reset_grid(self, xyz: Float[Tensor, "n 3"]):
        old_grids = self.grids
        old_delta = self.delta
        self.grids, num_grids = self.create_grids(xyz)
        new_delta = torch.zeros(num_grids, 7, device=xyz.device)
        new_delta[:, 3] = 1
        offset = 0
        
        for level_idx in range(self.config.num_grid_levels):
            og = old_grids[level_idx]
            ng = self.grids[level_idx]
            dist, ng2og = batched_knn(ng.vertex, og.vertex,k=3)
            ng_corr_old_delta = old_delta[ng2og + offset]
            ng_delta3 = ng_corr_old_delta[:,:,:3].mean(dim=1)
            ng_delta4 = average_quaternions(ng_corr_old_delta[:, :, 3:7])
            new_delta[offset:offset + ng.vertex_num] = torch.cat([ng_delta3, ng_delta4], dim=1)
            offset+=ng.vertex_num
            
        self.delta.data.copy_(new_delta)
        
        index, offset = [], 0
        for g in self.grids:
            index.append(g.p2v + offset)
            offset += g.vertex_num
            
        index = torch.stack(index).squeeze(dim=-1)
        self.register_buffer("index", index)
        if self.config.momentum is not None:
            self.delta.data *= self.config.momentum
            
        count = index.flatten().unique().numel()
        self.grids_num = offset
        logging.info(f"Setup deformation, grids: {offset}, occupied grids: {count}")

        self.reset_optimizer()

    def reset(self, xyz: Float[Tensor, "n 3"]):
        
        for level_idx in range(self.config.num_grid_levels):
            self.grids[level_idx].reset_p2v(xyz)
            
        index, offset = [], 0
        for g in self.grids:
            index.append(g.p2v + offset)
            offset += g.vertex_num
            
        index = torch.stack(index).squeeze(dim=-1)
        self.register_buffer("index", index)
        if self.config.momentum is not None:
            self.delta.data *= self.config.momentum
            
        count = index.flatten().unique().numel()
        self.grids_num = offset
        logging.info(f"Setup deformation, grids: {offset}, occupied grids: {count}")
    

    def setup(self, xyz: Float[Tensor, "n 3"], reset_grid: bool = False):
        if not self.grids or reset_grid:
            self.grids, num_grids = self.create_grids(xyz)
            delta = torch.zeros(num_grids, 7, device=xyz.device)
            delta[:, 3] = 1  # 0 0 0 1 0 0 0 -> x y z q1 q2 q3 q4
            self.register_parameter("delta", nn.Parameter(delta))

        index, offset = [], 0
        for g in self.grids:
            index.append(g.p2v + offset)
            offset += g.vertex_num
        index = torch.stack(index).squeeze(dim=-1)
        self.register_buffer("index", index)
        if self.config.momentum is not None:
            self.delta.data *= self.config.momentum

        count = index.flatten().unique().numel()
        self.grids_num = offset
        logging.info(f"Setup deformation, grids: {offset}, occupied grids: {count}")
        
        self.reset_optimizer()

    def reg_loss(self):
        identity = torch.zeros_like(self.delta[:1, :])
        identity[:, 3] = 1
        return (self.delta - identity).abs().mean()

    def reset_optimizer(self):
        self.optimizer = torch.optim.Adam(
            self.parameters(), lr=self.config.lr, eps=1e-15
        )

    def capture(self) -> Dict[str, Any]:
        if not self.grids:
            return {}
        return dict(delta=self.delta, index=self.index)  # TODO grids state

    def restore(self, ckpt: Dict[str, Any]):
        if "delta" in ckpt:
            logging.info("Restore deformation from checkpoint")
            self.register_buffer("index", ckpt["index"])
            self.register_parameter("delta", ckpt["delta"])
            self.reset_optimizer()

    def forward(self, xyz: Float[Tensor, "n 3"], normalized: bool = False):
        delta = self.delta[self.index].sum(dim=0)
        delta_xyz = delta[:, :3].contiguous()
        delta_rot = delta[:, 3:].contiguous()
        return delta_xyz, self.rotation_activation(delta_rot)