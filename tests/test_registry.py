from __future__ import annotations

import unittest

from sim_benchmarks.registry import benchmark_manifests, dataset_catalog, model_manifests, reproduction_targets


class RegistryTests(unittest.TestCase):
    def test_first_wave_benchmarks_are_pinned(self) -> None:
        manifests = benchmark_manifests()
        self.assertEqual(
            {item["id"] for item in manifests},
            {"colosseum_v2", "vla_arena", "domino", "ebench"},
        )
        for item in manifests:
            self.assertEqual(len(item["revision"]), 40)
            self.assertEqual(item["status"], "planned")

    def test_dataset_catalog_is_limited_to_benchmark_evaluation_data(self) -> None:
        catalog = dataset_catalog()
        self.assertEqual({"train_id", "eval_id", "eval_ood"}, set(catalog["split_policy"]))
        self.assertTrue(catalog["benchmark_native"])
        self.assertNotIn("external_pretraining", catalog)
        self.assertTrue(all("evaluation_role" in item for item in catalog["benchmark_native"]))
        self.assertTrue(all("use" not in item for item in catalog["benchmark_native"]))

    def test_xiaomi_robotics_1_model_source_is_pinned(self) -> None:
        manifests = model_manifests()
        self.assertEqual([item["id"] for item in manifests], ["xiaomi_robotics_1"])
        xr1 = manifests[0]
        self.assertEqual(len(xr1["revision"]), 40)
        self.assertEqual(xr1["kind"], "model_baseline")
        self.assertFalse(xr1["transport"]["reuse_upstream_transport"])
        self.assertEqual(
            {checkpoint["benchmark"] for checkpoint in xr1["checkpoints"]},
            {"vlabench", "robocasa", "robocasa365"},
        )
        self.assertTrue(all(len(checkpoint["revision"]) == 40 for checkpoint in xr1["checkpoints"]))

    def test_reproduction_targets_match_harness_pin(self) -> None:
        targets = reproduction_targets()
        self.assertEqual(targets["harness_revision"], "2680ab2fafe981c2dba63c6c1a4e7bb4415dbb56")
        robocasa365 = next(item for item in targets["targets"] if item["benchmark"] == "robocasa365")
        self.assertEqual(robocasa365["local_status"], "planned_backport")
