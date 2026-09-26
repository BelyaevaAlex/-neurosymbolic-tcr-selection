"""Boundary tests for integrity, information access and stopping mathematics."""
from pathlib import Path
import copy
import hashlib
import itertools
import json
import sys
import subprocess
import tempfile
import unittest

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT), str(ROOT/'src')]
from reproduce import verify_integrity
from policies import choose_action
from tool_kernel import parse_raw, result


class ReviewChecks(unittest.TestCase):
    def test_tampered_file_rejected(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            (root/'data.csv').write_bytes(b'1,2\n')
            (root/'checksums.json').write_text(json.dumps({'files':[
                {'path':'data.csv','bytes':4,'sha256':hashlib.sha256(b'1,2\n').hexdigest()}]}))
            self.assertEqual(verify_integrity(root), 1)
            (root/'data.csv').write_bytes(b'1,3\n')
            with self.assertRaisesRegex(ValueError, 'Integrity'):
                verify_integrity(root)

    def test_ambiguous_or_extended_tool_call_rejected(self):
        self.assertEqual(parse_raw('{"tool":"query_upper_bound"}'), {'tool':'query_upper_bound'})
        for raw in ['{"tool":"query_upper_bound","tool":"query_provisional_score"}',
                    '{"tool":"query_upper_bound","action":12}', '{"tool":"unknown"}',
                    '[{"tool":"query_upper_bound"}]', 'query_upper_bound']:
            self.assertIsNone(parse_raw(raw))

    def world(self):
        return dict(free=[.2,.4,.8], blocks=[[.1,.2,.3,.4,.5],[.7,.6,.5,.4,.3],[.2,.4,.6,.8,1.]],
                    blend=.5, k=2, reference_cost=5, labels=[0,1,1])

    def test_regret_equals_maximum_over_all_interval_corners(self):
        world = self.world()
        for actions in [[],[0],[0,0,1],[0]*5+[1]*5+[2]*5]:
            state, public = result(world, actions)
            losses = []
            for corners in itertools.product([0.,1.], repeat=3):
                scores = []
                for i, extreme in enumerate(corners):
                    count = public['counts'][i]
                    total = sum(world['blocks'][i][:count])+(5-count)*extreme
                    scores.append(.5*world['free'][i]+.5*total/5)
                losses.append(sum(sorted(scores)[-2:])-sum(scores[i] for i in state['selected']))
            self.assertAlmostEqual(state['regret'], max(losses), places=12)
            self.assertEqual(state['cost'], len(actions)+(5 if actions else 0))

    def test_policy_cannot_see_unrevealed_blocks_or_labels(self):
        original = self.world()
        changed = copy.deepcopy(original)
        changed['labels'] = [1,0,0]
        changed['blocks'][0][1:] = [1.,1.,1.,1.]
        changed['blocks'][1] = [0.]*5
        changed['blocks'][2] = [1.]*5
        _, first = result(original,[0])
        _, second = result(changed,[0])
        self.assertEqual(first,second)
        for rule in ['upper','provisional','uniform','fixed_sequence']:
            self.assertEqual(choose_action(first,None,rule),choose_action(second,None,rule))

    def test_fixed_sequence_uses_initial_scores_and_skips_exhausted_candidates(self):
        world = self.world()
        _, public = result(world, [])
        self.assertEqual(choose_action(public,None,'fixed_sequence'),2)
        _, public = result(world, [2]*5)
        self.assertEqual(choose_action(public,None,'fixed_sequence'),1)

    def test_data_only_integrity_does_not_require_bundled_source(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            (root/'data').mkdir()
            (root/'data/input.csv').write_bytes(b'1,2\n')
            (root/'checksums.json').write_text(json.dumps({'files':[
                {'path':'data/input.csv','bytes':4,'sha256':hashlib.sha256(b'1,2\n').hexdigest()},
                {'path':'private_source.py','bytes':0,'sha256':hashlib.sha256(b'').hexdigest()}]}))
            self.assertEqual(verify_integrity(root,data_only=True),1)
            (root/'data/input.csv').write_bytes(b'1,3\n')
            with self.assertRaisesRegex(ValueError,'Integrity'):
                verify_integrity(root,data_only=True)

    def test_private_inputs_are_explicitly_required(self):
        completed = subprocess.run([sys.executable,str(ROOT/'reproduce.py'),'--quick'],
                                   capture_output=True,text=True)
        self.assertEqual(completed.returncode,2)
        self.assertIn('--data-root',completed.stderr)
        self.assertNotIn('Traceback',completed.stderr)

    def test_empty_input_manifest_rejected(self):
        with tempfile.TemporaryDirectory() as directory:
            root=Path(directory)
            (root/'checksums.json').write_text('{"files":[]}')
            with self.assertRaises(ValueError):
                verify_integrity(root,data_only=True)


if __name__=='__main__':
    unittest.main()
