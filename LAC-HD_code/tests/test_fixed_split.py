"""Test actual paper membership and reject split leakage/legacy provenance."""
from __future__ import annotations
import copy
import json
import sys
import tempfile
import unittest
from pathlib import Path
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'Code'))
from fixed_split import DEFAULT_SPLIT_MANIFEST, fixed_split_indices, require_fixed_provenance

class FixedSplitTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.manifest = json.loads(Path(DEFAULT_SPLIT_MANIFEST).read_text(encoding='utf-8'))
        annotations = ROOT.parent / 'LAC-HD_data/annotations/dialogue_annotations.jsonl'
        cls.rows = [json.loads(line) for line in annotations.read_text(encoding='utf-8').splitlines() if line.strip()]

    def altered(self, mutate):
        manifest = copy.deepcopy(self.manifest)
        mutate(manifest)
        with tempfile.TemporaryDirectory() as tmp:
            file = Path(tmp) / 'manifest.json'
            file.write_text(json.dumps(manifest), encoding='utf-8')
            return fixed_split_indices(self.rows, file)

    def test_real_partition_counts_coverage_and_group_isolation(self):
        parts, info = fixed_split_indices(self.rows)
        self.assertEqual([len(parts[k]) for k in ('train', 'dev', 'test')], [17847, 991, 991])
        self.assertEqual(set(parts['train']) | set(parts['dev']) | set(parts['test']), set(range(19829)))
        self.assertFalse(set(parts['train']) & set(parts['test']))
        groups = [{self.rows[i]['id'].rsplit('_', 1)[0] for i in parts[k]} for k in ('train', 'dev', 'test')]
        self.assertEqual([len(g) for g in groups], [9000, 500, 500])
        for i in range(3):
            for j in range(i): self.assertFalse(groups[i] & groups[j])
        require_fixed_provenance(info)

    def test_membership_independent_of_record_order(self):
        p, _ = fixed_split_indices(self.rows)
        reverse = list(reversed(self.rows))
        q, _ = fixed_split_indices(reverse)
        for key in p:
            self.assertEqual({self.rows[i]['id'] for i in p[key]}, {reverse[i]['id'] for i in q[key]})

    def test_two_way_manifest_rejected(self):
        with self.assertRaises(ValueError): self.altered(lambda m: m['partitions'].pop('test'))

    def test_overlapping_group_rejected(self):
        def mutate(m): m['partitions']['dev']['group_ids'][0] = m['partitions']['train']['group_ids'][0]
        with self.assertRaises(ValueError): self.altered(mutate)

    def test_duplicate_ids_rejected(self):
        with self.assertRaises(ValueError): fixed_split_indices(self.rows + [self.rows[0]])

    def test_missing_response_rejected(self):
        with self.assertRaises(ValueError): fixed_split_indices(self.rows[:-1])

    def test_wrong_seed_rejected(self):
        with self.assertRaises(ValueError): self.altered(lambda m: m.update(seed=43))

    def test_old_checkpoint_provenance_rejected(self):
        with self.assertRaises(ValueError): require_fixed_provenance({'split_mode': 'group'})

if __name__ == '__main__': unittest.main()
