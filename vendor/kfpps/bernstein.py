from __future__ import annotations

from dataclasses import dataclass
import heapq

import numpy as np
from scipy import optimize

from .polynomial import (
    bernstein_coefficients_on_box,
    derivative_tensors,
    evaluate_power_polynomial,
)
from .types import BernsteinCertificate, BernsteinNodeSummary, SearchBox


@dataclass(order=True, slots=True)
class _QueuedNode:
    lower_bound: float
    depth: int
    box: SearchBox


class BernsteinBranchAndBound:
    def __init__(
        self,
        *,
        min_box_size_px: float = 0.1,
        objective_tolerance: float = 1e-9,
        max_nodes: int = 20000,
        max_recorded_nodes: int = 64,
    ) -> None:
        self.min_box_size_px = float(min_box_size_px)
        self.objective_tolerance = float(objective_tolerance)
        self.max_nodes = int(max_nodes)
        self.max_recorded_nodes = int(max_recorded_nodes)

    def solve(self, coeffs: np.ndarray, search_box: SearchBox) -> BernsteinCertificate:
        coeffs = np.asarray(coeffs, dtype=np.float64)
        grad_x_coeffs, grad_y_coeffs = derivative_tensors(coeffs)
        best_point, best_value = self._initial_upper_bound(coeffs, search_box)
        best_point, best_value = self._refine_point(
            coeffs,
            grad_x_coeffs,
            grad_y_coeffs,
            search_box,
            best_point,
            best_value,
        )
        initial_lower = self._lower_bound(coeffs, search_box)
        queue: list[_QueuedNode] = [_QueuedNode(initial_lower, 0, search_box)]
        heapq.heapify(queue)
        explored_nodes = 0
        pruned_nodes = 0
        leaf_nodes = 0
        recorded_nodes: list[BernsteinNodeSummary] = []

        while queue and explored_nodes < self.max_nodes:
            global_lower = queue[0].lower_bound
            if best_value - global_lower <= self.objective_tolerance:
                break

            node = heapq.heappop(queue)
            explored_nodes += 1
            sampled_upper = self._sample_upper_bound(coeffs, node.box)
            if sampled_upper < best_value:
                best_point = self._best_sample_point(coeffs, node.box)
                best_point, best_value = self._refine_point(
                    coeffs,
                    grad_x_coeffs,
                    grad_y_coeffs,
                    node.box,
                    best_point,
                    sampled_upper,
                )

            if node.lower_bound >= best_value - self.objective_tolerance:
                pruned_nodes += 1
                self._record_node(recorded_nodes, node, sampled_upper)
                continue

            if max(node.box.width, node.box.height) <= self.min_box_size_px:
                leaf_nodes += 1
                self._record_node(recorded_nodes, node, sampled_upper)
                continue

            left_box, right_box = node.box.split_longest()
            left_lower = self._lower_bound(coeffs, left_box)
            right_lower = self._lower_bound(coeffs, right_box)
            heapq.heappush(queue, _QueuedNode(left_lower, node.depth + 1, left_box))
            heapq.heappush(queue, _QueuedNode(right_lower, node.depth + 1, right_box))

        if queue:
            global_lower = min(node.lower_bound for node in queue)
        else:
            global_lower = best_value
        certified = (best_value - global_lower) <= self.objective_tolerance
        return BernsteinCertificate(
            best_point=np.asarray(best_point, dtype=np.float64),
            best_value=float(best_value),
            global_lower_bound=float(global_lower),
            global_upper_bound=float(best_value),
            certified=bool(certified),
            explored_nodes=explored_nodes,
            pruned_nodes=pruned_nodes,
            leaf_nodes=leaf_nodes,
            node_summaries=recorded_nodes,
        )

    def _initial_upper_bound(self, coeffs: np.ndarray, box: SearchBox) -> tuple[np.ndarray, float]:
        center = box.center
        best_point = center.copy()
        best_value = evaluate_power_polynomial(coeffs, center[0], center[1])

        def objective(sample: np.ndarray) -> float:
            return evaluate_power_polynomial(coeffs, sample[0], sample[1])

        try:
            result = optimize.shgo(
                objective,
                bounds=box.as_bounds(),
                iters=2,
                sampling_method="sobol",
            )
            if np.isfinite(result.fun):
                best_value = float(result.fun)
                best_point = np.asarray(result.x, dtype=np.float64)
        except Exception:
            pass
        return best_point, float(best_value)

    def _best_sample_point(self, coeffs: np.ndarray, box: SearchBox) -> np.ndarray:
        samples = box.sample_points()
        values = [evaluate_power_polynomial(coeffs, sample[0], sample[1]) for sample in samples]
        return np.asarray(samples[int(np.argmin(values))], dtype=np.float64)

    def _sample_upper_bound(self, coeffs: np.ndarray, box: SearchBox) -> float:
        return float(
            min(
                evaluate_power_polynomial(coeffs, sample[0], sample[1])
                for sample in box.sample_points()
            )
        )

    def _refine_point(
        self,
        coeffs: np.ndarray,
        grad_x_coeffs: np.ndarray,
        grad_y_coeffs: np.ndarray,
        box: SearchBox,
        initial_point: np.ndarray,
        initial_value: float,
    ) -> tuple[np.ndarray, float]:
        point = np.asarray(initial_point, dtype=np.float64)
        bounds = box.as_bounds()

        def objective(sample: np.ndarray) -> float:
            return evaluate_power_polynomial(coeffs, sample[0], sample[1])

        def gradient(sample: np.ndarray) -> np.ndarray:
            return np.array(
                [
                    evaluate_power_polynomial(grad_x_coeffs, sample[0], sample[1]),
                    evaluate_power_polynomial(grad_y_coeffs, sample[0], sample[1]),
                ],
                dtype=np.float64,
            )

        try:
            result = optimize.minimize(
                objective,
                x0=point,
                jac=gradient,
                bounds=bounds,
                method="L-BFGS-B",
            )
        except Exception:
            return point, float(initial_value)

        if not np.isfinite(result.fun):
            return point, float(initial_value)
        if float(result.fun) < float(initial_value):
            return np.asarray(result.x, dtype=np.float64), float(result.fun)
        return point, float(initial_value)

    def _lower_bound(self, coeffs: np.ndarray, box: SearchBox) -> float:
        bernstein = bernstein_coefficients_on_box(coeffs, box)
        return float(np.min(bernstein))

    def _record_node(
        self,
        recorded_nodes: list[BernsteinNodeSummary],
        node: _QueuedNode,
        sampled_upper: float,
    ) -> None:
        if len(recorded_nodes) >= self.max_recorded_nodes:
            return
        recorded_nodes.append(
            BernsteinNodeSummary(
                box=node.box,
                depth=node.depth,
                lower_bound=float(node.lower_bound),
                sampled_upper_bound=float(sampled_upper),
            )
        )
