from __future__ import annotations

import unittest

from sim_benchmarks.registry import benchmark_manifests, dataset_catalog, reproduction_targets


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

    def test_dataset_catalog_has_train_and_evaluation_split_policy(self) -> None:
        catalog = dataset_catalog()
        self.assertEqual({"train_id", "eval_id", "eval_ood"}, set(catalog["split_policy"]))
        self.assertTrue(catalog["benchmark_native"])
        self.assertTrue(catalog["external_pretraining"])

    def test_reproduction_targets_match_harness_pin(self) -> None:
        targets = reproduction_targets()
        self.assertEqual(targets["harness_revision"], "2680ab2fafe981c2dba63c6c1a4e7bb4415dbb56")
        self.assertTrue(any(item["benchmark"] == "robocasa365" for item in targets["targets"]))
