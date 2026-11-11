"""Small standalone API checks; synthetic inputs are not an accuracy evaluation."""

from pathlib import Path
import shutil
import subprocess
import sys
import tempfile
import unittest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import numpy as np

from predict import predict
from radar_model import inference
from radar_model.model import load_model
from radar_model.preprocessing import CLASSES, build_aligned_inputs, shift_without_wrap


def sample_inputs():
    """Return reproducible mixed piece lengths, sparse IDs and distinct within-piece times."""
    lengths = (1, 2, 5, 30)
    tracks = np.repeat(np.array([-100, 77, 50000, 9999]), lengths)
    times = np.concatenate([np.arange(length) * .1 for length in lengths])
    rng = np.random.default_rng(42)
    real = rng.normal(size=(len(tracks), 5, 256))
    imaginary = rng.normal(size=(len(tracks), 5, 256))
    raw = (real + 1j * imaginary).astype(np.complex64)
    return raw, np.full(len(raw), 50.), np.zeros(len(raw), dtype=bool), tracks, times


class ProcessingChecks(unittest.TestCase):
    """Checks that do not require a trained checkpoint."""

    def test_invalid_signal_floor(self):
        """Zero/non-finite segments receive the evaluated fixed input floor."""
        raw = np.zeros((2, 5, 256), dtype=np.complex64)
        raw[1, 0, 0] = np.nan
        np.testing.assert_array_equal(build_aligned_inputs(raw), -np.ones((2, 5, 256)))

    def test_alignment_and_relative_cell_power(self):
        """A known tone is centered while unequal cell powers remain unequal."""
        amplitudes = np.arange(1, 6)
        tone = np.exp(2j * np.pi * 11 * np.arange(256) / 256)
        raw = (amplitudes[:, None] * tone).astype(np.complex64)[None]
        values = build_aligned_inputs(raw)[0]
        self.assertEqual(int(values[2].argmax()), 128)
        fractions = 10 ** (values * 80 / 10)
        np.testing.assert_allclose(fractions.sum(axis=1), amplitudes ** 2 / np.sum(amplitudes ** 2), atol=3e-6)

    def test_no_circular_wrap(self):
        """Translated boundary energy is discarded instead of appearing at the other edge."""
        power = np.zeros((5, 256))
        power[:, -1] = 1
        self.assertEqual(float(shift_without_wrap(power, 1).sum()), 0)

    def test_probability_averaging(self):
        """A known interleaved grouping averages vectors and scatters them to original rows."""
        probabilities = np.zeros((4, 9))
        probabilities[:, :2] = [[.2, .8], [.9, .1], [.8, .2], [.1, .9]]
        tracks = np.array([100, -7, 100, 300])
        actual = inference.average_probabilities(probabilities, tracks, np.array([.1, 2., 0., 4.]))
        expected = probabilities.copy()
        expected[[0, 2], :2] = [.5, .5]
        np.testing.assert_allclose(actual, expected)

    def test_empty_and_malformed_inputs(self):
        """Empty calls return no labels; unsupported shapes and ID types are explicit errors."""
        empty = np.empty((0, 5, 256), dtype=np.complex64)
        self.assertEqual(predict(empty, [], [], np.array([], dtype=int), []), [])
        with self.assertRaises(ValueError):
            inference.predict_probabilities(np.zeros((1, 256)), np.array([1]), np.array([0.]))
        with self.assertRaises(ValueError):
            inference.predict_probabilities(np.zeros((1, 5, 256), dtype=complex), np.array([1.5]), np.array([0.]))


class ProductionChecks(unittest.TestCase):
    """Checks for the installed full-data model and standalone prediction API."""

    @classmethod
    def setUpClass(cls):
        """Load the actual submitted weights and prepare one synthetic API batch."""
        cls.model = load_model()
        cls.inputs = sample_inputs()

    def test_labels_cpu_and_cache(self):
        """The API returns N allowed strings and reuses the same CPU model."""
        labels = predict(*self.inputs)
        self.assertEqual(len(labels), len(self.inputs[0]))
        self.assertTrue(all(isinstance(label, str) and label in CLASSES for label in labels))
        self.assertEqual(CLASSES, ("D1", "D2", "D3", "D4", "D5", "D6", "human", "bird", "reflector"))
        self.assertTrue(all(parameter.device.type == "cpu" for parameter in self.model.parameters()))
        self.assertIs(load_model(), self.model)
        self.assertEqual(predict(*self.inputs), labels)

    def test_shuffle_ids_and_unused_metadata(self):
        """Output rows follow inputs; ID naming and unused metadata cannot supply evidence."""
        raw, ranges, edges, tracks, times = self.inputs
        expected = np.asarray(predict(*self.inputs))
        order = np.random.default_rng(21).permutation(len(raw))
        actual = np.asarray(predict(raw[order], ranges[order], edges[order], tracks[order], times[order]))
        np.testing.assert_array_equal(actual[np.argsort(order)], expected)
        names = {value: index * 19 + 7 for index, value in enumerate(np.unique(tracks))}
        renamed = np.array([names[value] for value in tracks])
        np.testing.assert_array_equal(predict(raw, np.full(len(raw), np.nan), ~edges, renamed, times + 300), expected)

    def test_piece_isolation(self):
        """Singletons and pieces up to 30 rows match their results in a combined call."""
        raw, ranges, edges, tracks, times = self.inputs
        expected = inference.predict_probabilities(raw, tracks, times)
        for identifier in np.unique(tracks):
            rows = np.flatnonzero(tracks == identifier)
            actual = inference.predict_probabilities(raw[rows], tracks[rows], times[rows])
            np.testing.assert_allclose(actual, expected[rows], atol=1e-6, rtol=1e-5)
            np.testing.assert_allclose(actual, np.repeat(actual[:1], len(rows), axis=0), atol=0)
        with self.assertRaises(ValueError):
            inference.predict_probabilities(raw, np.ones(len(raw), dtype=int), np.arange(len(raw)) * .1)

    def test_isolated_copy_and_working_directory(self):
        """A runtime-only copy imports and predicts from an unrelated working directory."""
        with tempfile.TemporaryDirectory() as temporary:
            isolated = Path(temporary) / "package"
            isolated.mkdir()
            shutil.copyfile(ROOT / "predict.py", isolated / "predict.py")
            for name in ("radar_model", "model"):
                shutil.copytree(ROOT / name, isolated / name, ignore=shutil.ignore_patterns("__pycache__"))
            command = """
import sys
sys.path.insert(0, sys.argv[1])
import numpy as np
from predict import predict
segments = np.zeros((1, 5, 256), dtype=np.complex64)
labels = predict(segments, [50.], [False], np.array([987]), np.array([0.]))
assert len(labels) == 1 and isinstance(labels[0], str)
"""
            result = subprocess.run([sys.executable, "-I", "-B", "-c", command, str(isolated)],
                                    cwd=temporary, capture_output=True, text=True)
            self.assertEqual(result.returncode, 0, result.stdout + result.stderr)


if __name__ == "__main__":
    unittest.main()
