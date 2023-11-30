#!/usr/bin/env python
# coding: utf-8

# In[1]:


import math
from nltk import word_tokenize
from collections import Counter
from nltk.util import ngrams


class BLEU(object):
    @staticmethod
    def compute(candidate, references, weights):
        candidate = [c.lower() for c in candidate]
        references = [[r.lower() for r in reference] for reference in references]

        p_ns = (BLEU.modified_precision(candidate, references, i) for i, _ in enumerate(weights, start=1))
        s = math.fsum(w * math.log(p_n) for w, p_n in zip(weights, p_ns) if p_n)

        bp = BLEU.brevity_penalty(candidate, references)
        return bp * math.exp(s)

    @staticmethod
    def modified_precision(candidate, references, n):
        counts = Counter(ngrams(candidate, n))

        if not counts:
            return 0

        max_counts = {}
        for reference in references:
            reference_counts = Counter(ngrams(reference, n))
            for ngram in counts:
                max_counts[ngram] = max(max_counts.get(ngram, 0), reference_counts[ngram])

        clipped_counts = dict((ngram, min(count, max_counts[ngram])) for ngram, count in counts.items())

        return sum(clipped_counts.values()) / sum(counts.values())
    
    @staticmethod
    def brevity_penalty(candidate, references):
        c = len(candidate)
        # r = min(abs(len(r) - c) for r in references)
        r = min(len(r) for r in references)

        if c > r:
            return 1
        else:
            return math.exp(1 - r / c)
        

if __name__ == '__main__':
    scorer = BLEU()
    grount_truths = ["$ \sin ^ { 2 } \theta + \cos ^ { 2 } \theta = 1 $",
                     "$ \sum _ { { T \geq g } } { 8 . 2 } $",
                     "$ r = r ( \theta ) $"]


    # the predictions must be in the same format where each symbol is followed by a space
    predictions = ["$ \cos ^ { 2 } \theta + \cos ^ { 2 } \theta = 1 } } } $  ",
                   "$ \sum _ { { T \leq g } } { 0 . 2 } $",
                   "$ x = R ( \theta ) $"]


    overall = 0
    for gt, pred in zip(grount_truths, predictions):
        gt = gt.split()
        pred = pred.split()
        overall += BLEU.compute(pred,[gt], weights=[1/4, 1/4, 1/4, 1/4])

    print("Macro Bleu : ", overall/len(predictions))
# In[2]:


# scorer = BLEU()
# grount_truths = ["$ \sin ^ { 2 } \theta + \cos ^ { 2 } \theta = 1 $",
#                  "$ \sum _ { { T \geq g } } { 8 . 2 } $",
#                  "$ r = r ( \theta ) $"]


# # the predictions must be in the same format where each symbol is followed by a space
# predictions = ["$ \cos ^ { 2 } \theta + \cos ^ { 2 } \theta = 1 } } } $  ",
#                "$ \sum _ { { T \leq g } } { 0 . 2 } $",
#                "$ x = R ( \theta ) $"]


# overall = 0
# for gt, pred in zip(grount_truths, predictions):
#     gt = gt.split()
#     pred = pred.split()
#     overall += BLEU.compute(pred,[gt], weights=[1/4, 1/4, 1/4, 1/4])

# print("Macro Bleu : ", overall/len(predictions))


# In[ ]:




