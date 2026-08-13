from __future__ import annotations

import unittest

from sim_benchmarks.registry import benchmark_manifests, dataset_catalog, model_manifests, reproduction_targets


class RegistryTests(unittest.TestCase):
    def test_first_wave_benchmarks_are_pinned(self) -> None:
        manifests = benchmark_manifests()
        by_id = {item["id"]: item for item in manifests}
        self.assertEqual(
            {item["id"] for item in manifests if item["status"] == "planned"},
            {"colosseum_v2", "vla_arena", "domino", "ebench"},
        )
        for item in manifests:
            self.assertEqual(len(item["revision"]), 40)
        self.assertEqual(by_id["vlabench"]["status"], "scaffolded")
        self.assertEqual(len(by_id["vlabench"]["tracks"]), 5)

    def test_dataset_catalog_is_limited_to_benchmark_evaluation_data(self) -> None:
        catalog = dataset_catalog()
        self.assertEqual({"train_id", "eval_id", "eval_ood"}, set(catalog["split_policy"]))
        self.assertTrue(catalog["benchmark_native"])
        self.assertNotIn("external_pretraining", catalog)
        self.assertTrue(all("evaluation_role" in item for item in catalog["benchmark_native"]))
        self.assertTrue(all("use" not in item for item in catalog["benchmark_native"]))

    def test_model_sources_are_pinned(self) -> None:
        manifests = model_manifests()
        self.assertEqual([item["id"] for item in manifests], ["pi05_libero", "xiaomi_robotics_1"])
        self.assertTrue(all(len(item["revision"]) == 40 for item in manifests))

        by_id = {item["id"]: item for item in manifests}
        pi05 = by_id["pi05_libero"]
        self.assertEqual(pi05["benchmark_runtime"]["suite"], "libero_object")
        self.assertTrue(all(len(checkpoint["revision"]) == 40 for checkpoint in pi05["checkpoints"]))
        self.assertTrue(all(len(weight["sha256"]) == 64 for weight in pi05["checkpoints"][0]["weight_files"]))

        xr1 = by_id["xiaomi_robotics_1"]
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
