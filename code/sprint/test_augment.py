import unittest

import numpy as np

from sprint.augment import (asr_variant, augment_batch, cluster_key, digitize, normalize_numbers,
                            procedural_oos, romanize, strip_fillers)


class AugmentTests(unittest.TestCase):
    def test_long_vowels_and_digits(self):
        self.assertEqual(romanize('सात बजे'), 'saat baje')
        self.assertEqual(digitize('saat baje kal'), '7 baje kal')
        self.assertEqual(digitize('do something'), 'do something')
        self.assertEqual(normalize_numbers('7:15 tomorrow'), 'saat baje pandrah minute tomorrow')
        self.assertEqual(normalize_numbers('30 percent volume'), 'tees percent volume')

    def test_boundary_fillers(self):
        self.assertEqual(strip_fillers('bhai list kholo na'), 'list kholo')
        self.assertEqual(strip_fillers('na list hatao'), 'na list hatao')
        self.assertEqual(cluster_key('bhai test this please'), cluster_key('test this'))

    def test_asr_and_seed(self):
        first = asr_variant('mujhe aap kal bulao', np.random.default_rng(1))
        self.assertEqual(first, 'mjhe ap kal bulao')
        values = ['सात बजे', 'mujhe kya chahiye']
        self.assertEqual(augment_batch(values), augment_batch(values))

    def test_procedural_novelty(self):
        rows = ['one alpha source sequence here', 'two beta other words there']
        oos = procedural_oos(rows, [0, 1], 12)
        self.assertEqual(len(oos), 12)
        self.assertTrue(all(text and text not in rows for text in oos))


if __name__ == '__main__':
    unittest.main()
