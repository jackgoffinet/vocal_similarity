from __future__ import print_function, division
"""
Python script to accompany "Little Evidence for the Vocal Similarity Hypothesis"
by Jack Goffinet (2018), a letter in response to "Vocal similarity predicts the
relative attraction of musical chords" by Daniel L. Bowling, Dale Purves, and
Kamraan Z. Gill (December 2017). This script reproduces all the quantitative
results referenced in the letter.

Usage:
    $ python main.py

Output:
    Total number of tested chords:  298
    Total number of possible chords: 1000 (3.36 times the number tested)

    Tritones have modeled consonances ranging from 39th to 71st percentile among
    possible dyads.

    8321 pairs of chords tested out of:
    26301 pairs of experimentally tested chords (31.6%)
    268907 pairs of possible chords (3.09%)

    Maximum percent variance explained by models:
    Sethares, dyads: 84.4
    Vocal Similarity, dyads: 87.8
    Sethares, triads: 77.0
    Vocal Similarity, triads: 69.3
    Sethares, tetrads: 69.5
    Vocal Similarity, tetrads: 56.7

    Regressions outputted as pdf files.

Dependencies:
    1) An implementation of William Sethares' dissonance measure:
        sethares.py, see https://gist.github.com/endolith/3066664

    2) Mean consonance ratings reported in (Bowling et al., 2017):
        i) Go to https://doi.org/10.1073/pnas.1713206115
        ii) Click on the tab labeled 'Figures & SI'
        iii) Scroll to the bottom and download 'Dataset_S01 (XLSX)'
        iv) Copy and paste the "Combined Ratings" "mean" column from each tab
            (dyads, triads, and tetrads) into a text files named 
            'dyad_exp_means.txt', 'triad_exp_means.txt', and
            'tetrad_exp_means.txt'.

    3) Standard Python dependencies:
        NumPy, see http://www.numpy.org/
        SciPy library, see https://www.scipy.org/scipylib/index.html
        scikit-learn, see http://www.scikit-learn.org/
        matplotlib, see https://matplotlib.org/
        seaborn (optional), see https://seaborn.pydata.org/

References:
    @article{bowling2017vocal,
      title={Vocal similarity predicts the relative attraction of musical
             chords},
      author={Bowling, Daniel L and Purves, Dale and Gill, Kamraan Z},
      journal={Proceedings of the National Academy of Sciences},
      pages={201713206},
      year={2017},
      publisher={National Acad Sciences},
      note={Available at: https://doi.org/10.1073/pnas.1713206115}
    }

    @article{goffinet2018little,
      title={Little evidence for the vocal similarity hypothesis},
      author={Goffinet, Jack},
      year={2018},
      note={Letter submitted for publication.}
    }

"""

__author__ = "Jack Goffinet"
__data__ = "March 2018"


from itertools import combinations
from math import factorial
from fractions import gcd
import traceback
import numpy as np
from sklearn.isotonic import IsotonicRegression
from scipy.stats import rankdata
import matplotlib.pyplot as plt
try:
    import seaborn as sns
    sns.set()
except ImportError:
    pass
try:
    import sethares
except ImportError:
    print("ImportError: The file <sethares.py> can't be found.")
    print("Download it at https://gist.github.com/endolith/3066664")
    quit()
import vocal_similarity

plt.rc('text', usetex=True)
plt.rc('font', family='sans-serif')


# Just intonation numerators and denominators used by Bowling et al.
JI_NUMS =   np.array([1, 16, 9, 6, 5, 4, 7, 3, 8, 5, 9, 15, 2])
JI_DENOMS = np.array([1, 15, 8, 5, 4, 3, 5, 2, 5, 3, 5, 8,  1])
JI_RATIOS = JI_NUMS / JI_DENOMS
MIDDLE_C = 220.0 * 2.0 ** (3/12) # Middle C in Hertz
NUM_PARTIALS = 4 # Number of partials considered for Sethares model


def get_freq_ratio(chord):
    """Return chord's relative fundamental frequencies in reduced form.

    The argument <chord> is a sorted tuple of ints in the range [0,12]. The int
    7 respresents the 7th semitone, so that (0, 4, 7) represents a major triad.

    Args:
        chord (tuple of ints): A sorted tuple representing a chord. (see above)

    Returns:
        tuple of ints: The relative frequencies of the chord's fundamentals.

    """
    numerators = JI_NUMS[list(chord)]
    denoms = JI_DENOMS[list(chord)]
    denominator = vocal_similarity.get_lcm(denoms)
    numerators = [(numerators[i] * denominator) //
                        denoms[i] for i in range(len(numerators))]
    return tuple(numerators)


def parallel_sort(x, y):
    """Sort array <x> and apply the same permutation to <y>."""
    args = np.argsort(x)
    return x[args], y[args]


def choose(a, b):
    """Calculate <a> choose <b>."""
    if a == b:
        return 1
    temp = reduce(lambda x, y: x*y, range(a, a - b, -1))
    return  temp // factorial(b)


def zscore(a):
    """Calculate zscores for array <a>."""
    return (a - np.mean(a)) / np.std(a)


def sethares_wrapper(chord):
    """Wrapper function for <sethares.dissmeasure>."""
    freqs = np.zeros(len(chord) * NUM_PARTIALS)
    for i, pitch in enumerate(chord):
        for j in range(NUM_PARTIALS):
            freqs[i * NUM_PARTIALS + j] = (j + 1) * JI_RATIOS[pitch]
    freqs *= MIDDLE_C / np.mean(JI_RATIOS[list(chord)])
    return -1.0 * sethares.dissmeasure(freqs, np.ones(len(freqs)))



# Define tested dyads, triads, and tetrads.
pitches = range(1,13,1)
tested_dyads = [(0,) + i for i in combinations(pitches, 1)]
tested_triads = [(0,) + i for i in combinations(pitches, 2)]
tested_tetrads = [(0,) + i for i in combinations(pitches, 3)]

total_num_tested = len(tested_dyads) + len(tested_triads) + len(tested_tetrads)
print("Total number of tested chords: ", total_num_tested)

# Define possible dyads, triads, and tetrads.
pitches = range(0,13,1)
possible_dyads = list(combinations(pitches, 2))
possible_triads = list(combinations(pitches, 3))
possible_tetrads = list(combinations(pitches, 4))

# Determine how many unique chords are contained within these.
dyad_ratios = list(frozenset(map(get_freq_ratio, possible_dyads)))
triad_ratios = list(frozenset(map(get_freq_ratio, possible_triads)))
tetrad_ratios = list(frozenset(map(get_freq_ratio, possible_tetrads)))

total_num_possible = len(dyad_ratios) +  len(triad_ratios) + \
                                            len(tetrad_ratios)
to_print = "Total number of possible chords: " + str(total_num_possible) + " ("
to_print += "{0:.2f}".format(total_num_possible / total_num_tested) + " "
to_print += "times the number tested)"
print(to_print)

# Test modeled tritone consonances.
ratios_encountered = set()
to_delete = []
for i in range(len(possible_dyads)):
    ratio = get_freq_ratio(possible_dyads[i])
    if ratio in ratios_encountered:
        to_delete.append(i)
    else:
        ratios_encountered.add(ratio)
for i in reversed(to_delete):
    del possible_dyads[i]
b_dyad_scores = np.array(map( \
                        vocal_similarity.get_consonance_score, possible_dyads))
b_dyad_scores.sort()
tritones = [(i, i+6) for i in range(7)]
percentiles = []
for tritone in tritones:
    score = vocal_similarity.get_consonance_score(tritone)
    avg_index = 0.5 * ( np.searchsorted(b_dyad_scores, score, side='left') + \
                        np.searchsorted(b_dyad_scores, score, side='right') - 1)
    percentiles.append(100.0 * avg_index/len(possible_dyads))

to_print = "Tritones have modeled consonances ranging from "
to_print += str(int(round(min(percentiles)))) + "th to "
to_print += str(int(round(max(percentiles))))
to_print += "st percentile among possible dyads."
print("\n" + to_print)

# Calculate the number of pairs of chords.
groups = [tested_dyads, tested_triads, tested_tetrads]
num_exp_tested_pairs = sum(choose(len(group), 2) for group in groups)
groups = [dyad_ratios, triad_ratios, tetrad_ratios]
num_poss_tested_pairs = sum(choose(len(group), 2) for group in groups)

to_print = "8321 pairs of chords tested out of:\n" + str(num_exp_tested_pairs)
to_print += " pairs of experimentally tested chords ("
to_print += "{0:.1f}".format(100.0 * 8321 / num_exp_tested_pairs) + "%)\n"
to_print += str(num_poss_tested_pairs) + " pairs of possible chords ("
to_print += "{0:.2f}".format(100.0 * 8321 / num_poss_tested_pairs) + "%)"
print("\n" + to_print)

# Perform regressions.
groups = [tested_dyads, tested_triads, tested_tetrads]
model_names = ["Sethares", "Vocal Similarity"]
model_funcs = [sethares_wrapper, vocal_similarity.get_consonance_score]
print("\nMaximum percent variance explained by models:")

for group, group_name in zip(groups, ['dyad', 'triad', 'tetrad']):
    for func, model_name in zip(model_funcs, model_names):
        x_data = np.array([func(chord) for chord in group])
        rank_data = rankdata(x_data)
        try:
            y_data = np.loadtxt(group_name + '_exp_means.txt')
        except IOError:
            print("Cannot load file: "+group_name + '_exp_means.txt')
            print(traceback.format_exc())
            quit()
        y_data = zscore(y_data)
        sum_of_squares = np.sum(y_data ** 2)
        rank_data, y_data = parallel_sort(rank_data, y_data)
        ir = IsotonicRegression(increasing=True, out_of_bounds='clip')
        y_model = np.array(ir.fit_transform(rank_data, y_data))
        residual = np.sum((y_model - y_data) ** 2)
        r2 = 1.0 - residual / sum_of_squares

        to_print = model_name + ", " + group_name + "s: "
        to_print += "{0:.1f}".format(100.0 * r2)
        print(to_print)

        # Plot regression.
        plt.scatter(rank_data, y_data, marker='o', alpha=0.5)
        label = model_name + ", $R^2=" + "{0:.3f}".format(r2) + "$"
        plt.plot(rank_data, y_model, ls='-', label=label)
    plt.title("Model Comparsion, " + group_name.capitalize() + "s")
    plt.legend()
    plt.xlabel('Modeled Chord Consonance Rank')
    plt.ylabel('Average Consonance Rating (zscore)')
    plt.savefig(group_name + "_fits.pdf")
    plt.clf()

print("\nRegressions outputted as pdf files.")
