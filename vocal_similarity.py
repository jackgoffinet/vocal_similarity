from __future__ import print_function, division
"""
An implementation of the vocal similarity model of chord consonance described
in "Vocal similarity predicts the relative attraction of musical chords" by
Daniel L. Bowling, Dale Purves, and Kamraan Z. Gill (December 2017).

Reference:
    @article{bowling2017vocal,
      title={Vocal similarity predicts the relative attraction of musical
             chords},
      author={Bowling, Daniel L and Purves, Dale and Gill, Kamraan Z},
      journal={Proceedings of the National Academy of Sciences},
      pages={201713206},
      year={2017},
      publisher={National Acad Sciences}
      note={Available at: https://doi.org/10.1073/pnas.1713206115}
    }

Usage:
    $ python vocal_similarity.py

Notes:
    The main method is <get_consonance_score>.

"""

__author__ = "Jack Goffinet"
__date__ = "March 2018"


from fractions import gcd
from itertools import combinations


# Numerators & Denominators of the just intonation ratios used.
JI_NUMS =   [1, 16, 9, 6, 5, 4, 7, 3, 8, 5, 9, 15, 2]
JI_DENOMS = [1, 15, 8, 5, 4, 3, 5, 2, 5, 3, 5, 8,  1]
MIDDLE_C = 220.0 * 2.0 ** (3 / 12) # Middle C in Hertz



def get_gcd(numbers):
    """Return the greatest common divisor of the given integers"""
    return reduce(gcd, numbers)


def get_lcm(numbers):
    """Return the lowest common multiple of the given integers."""
    def helper(a, b):
        return (a * b) // gcd(a, b)
    return reduce(helper, numbers, 1)


def chord_to_freq_ratios(chord):
    """Return the frequency ratios of the pitches in <chord>

    Args:
        chord (tuple of ints): see <get_consonance_score>.

    Returns:
        list of ints:
    """
    numerators = [JI_NUMS[i] for i in chord]
    denoms = [JI_DENOMS[i] for i in chord]
    denominator = get_lcm(denoms)
    numerators = [(numerators[i] * denominator) // denoms[i] for i in \
                                                        range(len(numerators))]
    return numerators, denominator


def harmonic_metric(chord):
    """Calculate the harmonic metric described in (Bowling et al., 2017).

    Roughly, the portion of the pitches' GCD's harmonic spectrum coinciding with
    the harmonic spectra of the pitches.

    Args:
        chord (tuple of ints): see <get_consonance_score>.

    Returns:
        float: The harmonic metric of the given chord.

    """
    numerators, _ = chord_to_freq_ratios(chord)
    numerator_gcd = get_gcd(numerators)
    numerator_lcm = get_lcm(numerators)
    result_denom = numerator_lcm // numerator_gcd
    gcd_series = range(numerator_gcd, numerator_lcm + 1, numerator_gcd)
    result_num = 0
    # Count the overlap in GCD series & chord spectrum.
    for i in gcd_series:
        for j in numerators:
            if i % j == 0:
                result_num += 1
                break
    return result_num / result_denom


def interval_metric(chord):
    """Return the minimum fundamental frequency diff. among pitches in <chord>

    The chord is first transposed so that the average fundamental frequency of
    its pitches is <MIDDLE_C>.

    Args:
        chord (tuple of ints): see <get_consonance_score>.

    Returns:
        float: The minimum diff. among the pitches' fundamental frequencies.

    """
    numerators, denominator = chord_to_freq_ratios(chord)
    freqs = [i / denominator for i in numerators]
    avg_freq = sum(freqs) / len(freqs)
    freqs = [i / avg_freq * MIDDLE_C for i in freqs]
    diffs = [freqs[i+1] - freqs[i] for i in range(len(freqs)-1)]
    return min(diffs)


def get_consonance_score(chord, cutoff=50.0):
    """Combine the interval and harmonic metrics, outputing a single score.

    In the original algorithm description, the vocal similarity model is
    restricted to pairwise chord comparisons. Therefore the score returned by
    this method is only intended to be used for ranking and pairwise comparisons
    (what matters is relative score, not absolute differences in score). If the
    interval metric is used, the result will be negative and if the harmonic
    metric is used, the result will be positive.

    The argument <chord> is a sorted tuple of ints in the range [0,12]. The int
    7 respresents the 7th semitone, so (0, 4, 7) represents a major triad.

    Args:
        chord (tuple of ints): A sorted tuple representing a chord. (see above)
        cutoff (float, optional): Vocal range limit, in Hertz. Defaults to 50.0.

    Returns:
        float: The chord's consonance score.

    """
    min_diff = interval_metric(chord)
    if min_diff < cutoff:
        return (-cutoff - 1.0 + min_diff) / (cutoff + 1.0)
    return harmonic_metric(chord)



if __name__ == '__main__':
    major_triad = (0,4,7)
    minor_triad = (0,3,7)
    print(get_consonance_score(major_triad))
    print(get_consonance_score(minor_triad))
