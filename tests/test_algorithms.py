"""Edge cases for search, transforms, estimation and closed-loop control."""

from pathlib import Path
import sys
import unittest
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "Simulation Labs"))
from planning import grid_search, DStarLite, segment_free, adjacent
from estimation import rotation, icp, between, compose, wrap, ekf_step
from control import jacobian_arm, forward_arm, mpc_inputs, regulated_track


class PlanningTests(unittest.TestCase):
    def test_all_door_states_and_moved_start(self):
        grid = np.zeros((8, 12), bool)
        grid[:, 6] = True
        a, b = (2, 6), (6, 6)
        grid[a] = grid[b] = False
        planner = DStarLite(grid, (2, 0), (2, 11))
        for start, changes in [
            ((2, 0), []),
            ((2, 2), [(a, True)]),
            ((3, 3), [(b, True)]),
            ((3, 3), [(a, False)]),
            ((5, 8), [(b, False)]),
        ]:
            planner.move_start(start)
            for pos, value in changes:
                planner.change_cell(pos, value)
            planner.repair()
            p = planner.path()
            reference, _ = grid_search(planner.grid, start, planner.goal)
            self.assertEqual(len(p), len(reference))
            if p:
                self.assertEqual(p[0], start)
                self.assertEqual(p[-1], planner.goal)
                self.assertTrue(
                    all(v in adjacent(u, grid.shape) for u, v in zip(p, p[1:]))
                )
                self.assertTrue(all(not planner.grid[v] for v in p))

    def test_tangent_and_degenerate_segments(self):
        circles = [(np.array([0.0, 0.0]), 1.0)]
        self.assertFalse(
            segment_free(np.array([-2.0, 1.0]), np.array([2.0, 1.0]), circles)
        )
        self.assertTrue(
            segment_free(np.array([2.0, 1.0]), np.array([2.0, 1.0]), circles)
        )
        self.assertFalse(
            segment_free(np.array([0.0, 0.0]), np.array([0.0, 0.0]), circles)
        )

    def test_start_equals_goal(self):
        grid = np.zeros((3, 3), bool)
        p = (1, 1)
        solver = DStarLite(grid, p, p)
        solver.repair()
        self.assertEqual(solver.path(), [p])
        self.assertEqual(grid_search(grid, p, p)[0], [p])


class EstimationTests(unittest.TestCase):
    def test_pose_inverse_and_angle_wrap(self):
        a = np.array([2.0, -1.0, 3.11])
        b = np.array([1.5, 0.2, -3.12])
        np.testing.assert_allclose(compose(a, between(a, b)), b)
        self.assertLess(abs(wrap(2 * np.pi + 0.01) - 0.01), 1e-12)

    def test_registration_known_pose(self):
        rng = np.random.default_rng(89)
        p = rng.normal(size=(80, 2))
        q = p @ rotation(0.03) + [0.02, 0.03]
        fit, R, t, e = icp(p, q)
        np.testing.assert_allclose(fit, q, atol=1e-8)
        self.assertAlmostEqual(np.linalg.det(R), 1)
        self.assertLess(e[-1], 1e-8)

    def test_covariance_without_observations(self):
        x = np.zeros(3)
        P = np.eye(3) * 0.1
        Q = np.eye(2) * 0.02
        R = np.eye(2) * 0.01
        estimate, cov = ekf_step(x, P, [1.0, 0.0], [], [], 0.1, Q, R)
        np.testing.assert_allclose(estimate, [0.1, 0, 0])
        np.testing.assert_allclose(cov, cov.T)
        self.assertGreater(np.linalg.eigvalsh(cov).min(), 0)


class ControlTests(unittest.TestCase):
    def test_cart_at_goal_and_at_limit(self):
        np.testing.assert_allclose(
            mpc_inputs([2.0, 0.0], np.array([2.0, 0.0])), 0.0, atol=1e-12
        )
        u = mpc_inputs([0.0, 0.0], np.array([8.0, 0.0]), limit=0.4)
        self.assertLessEqual(np.max(np.abs(u)), 0.4 + 1e-9)
        self.assertGreater(u[0], 0.35)

    def test_arm_jacobian(self):
        q = np.array([0.1, 0.7])
        lengths = np.array([0.8, 0.4])
        h = 1e-6
        num = np.column_stack(
            [
                (
                    forward_arm(q + h * v, lengths)[-1]
                    - forward_arm(q - h * v, lengths)[-1]
                )
                / (2 * h)
                for v in np.eye(2)
            ]
        )
        np.testing.assert_allclose(jacobian_arm(q, lengths), num, atol=1e-8)

    def test_forward_projection_holds_position(self):
        x = np.linspace(0.0, 1.0, 101)
        path = np.c_[x, np.zeros_like(x)]
        track, speed, _, _ = regulated_track(
            path, [(np.array([0.12, 0.04]), 0.045)], max_steps=5
        )
        np.testing.assert_allclose(speed, 0.0)
        np.testing.assert_allclose(track, np.broadcast_to(track[0], track.shape))


if __name__ == "__main__":
    unittest.main()
