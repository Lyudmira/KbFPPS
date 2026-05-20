from __future__ import annotations

import warnings

import numpy as np

from .bernstein import BernsteinBranchAndBound
from .kruppa import build_kruppa_objective_coefficients
from .types import CertifiedPrincipalPointResult, FundamentalMatrixObservation, SearchBox


class CertifiedPrincipalPointSolver:
    def __init__(
        self,
        *,
        fx: float,
        fy: float,
        image_size: tuple[int, int],
        search_box: SearchBox | None = None,
        search_fraction: float = 0.15,
        rank_threshold: float = 1e-8,
        support_discriminant_threshold: float = 1e-8,
        min_box_size_px: float = 0.1,
        objective_tolerance: float = 1e-9,
        max_nodes: int = 20000,
        expansion_factor: float = 1.5,
        max_expansions: int = 2,
    ) -> None:
        self.fx = float(fx)
        self.fy = float(fy)
        if not np.isfinite(self.fx) or not np.isfinite(self.fy) or self.fx == 0.0 or self.fy == 0.0:
            raise ValueError("Known focal lengths must be finite and non-zero.")
        self.image_size = image_size
        self.search_box = search_box
        self.search_fraction = float(search_fraction)
        self.rank_threshold = float(rank_threshold)
        self.support_discriminant_threshold = float(support_discriminant_threshold)
        self.min_box_size_px = float(min_box_size_px)
        self.objective_tolerance = float(objective_tolerance)
        self.max_nodes = int(max_nodes)
        self.expansion_factor = float(expansion_factor)
        self.max_expansions = int(max_expansions)

    def solve(self, observations: list[FundamentalMatrixObservation]) -> CertifiedPrincipalPointResult:
        coeffs, diagnostics = build_kruppa_objective_coefficients(
            observations,
            fx=self.fx,
            fy=self.fy,
            rank_threshold=self.rank_threshold,
            support_discriminant_threshold=self.support_discriminant_threshold,
        )
        valid_pairs = sum(1 for diagnostics_i in diagnostics if diagnostics_i.contributes_objective)
        angle_pairs = sum(1 for diagnostics_i in diagnostics if diagnostics_i.angle_constraint_active)
        if valid_pairs == 0:
            raise ValueError("No pairwise constraints remain after degeneracy filtering.")
        if valid_pairs < 2 and angle_pairs == 0:
            warnings.warn("F3 failure mode: fewer than 2 F-only pairwise constraints remain after filtering.")

        solver = BernsteinBranchAndBound(
            min_box_size_px=self.min_box_size_px,
            objective_tolerance=self.objective_tolerance,
            max_nodes=self.max_nodes,
        )
        current_box = self.search_box or SearchBox.from_image_size(
            self.image_size,
            fraction=self.search_fraction,
        )
        used_expanded_box = False
        certificate = solver.solve(coeffs, current_box)
        boundary_hit = current_box.on_boundary(
            certificate.best_point[0],
            certificate.best_point[1],
            self.min_box_size_px,
        )
        expansion_count = 0
        while boundary_hit and expansion_count < self.max_expansions:
            used_expanded_box = True
            current_box = current_box.expand(self.expansion_factor)
            certificate = solver.solve(coeffs, current_box)
            boundary_hit = current_box.on_boundary(
                certificate.best_point[0],
                certificate.best_point[1],
                self.min_box_size_px,
            )
            expansion_count += 1

        return CertifiedPrincipalPointResult(
            cx=float(certificate.best_point[0]),
            cy=float(certificate.best_point[1]),
            objective_value=float(certificate.best_value),
            certificate_gap=float(certificate.global_upper_bound - certificate.global_lower_bound),
            certified=bool(certificate.certified),
            search_box=current_box,
            used_expanded_box=used_expanded_box,
            boundary_hit=boundary_hit,
            valid_pairs=valid_pairs,
            diagnostics=diagnostics,
            certificate=certificate,
        )
