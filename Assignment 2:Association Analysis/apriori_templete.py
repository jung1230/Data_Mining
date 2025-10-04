from __future__ import print_function
import sys
from itertools import combinations


def apriori(dataset, min_support=0.5, verbose=False):
    """Implements the Apriori algorithm.

    The Apriori algorithm will iteratively generate new candidate
    k-itemsets using the frequent (k-1)-itemsets found in the previous
    iteration.

    Parameters
    ----------
    dataset : list
        The dataset (a list of transactions) from which to generate
        candidate itemsets.

    min_support : float
        The minimum support threshold. Defaults to 0.5.

    Returns
    -------
    F : list
        The list of frequent itemsets.

    support_data : dict
        The support data for all candidate itemsets.

    References
    ----------
    .. [1] R. Agrawal, R. Srikant, "Fast Algorithms for Mining Association
           Rules", 1994.

    """
    C1 = create_candidates(dataset) ## [frozenset({'Apple'}), frozenset({'Corn'}), frozenset({'Doll'}), frozenset({'Eggs'}), frozenset({'Ice-cream'}), frozenset({'Key-chain'}), frozenset({'Mango'}), frozenset({'Nintendo'}), frozenset({'Onion'}), frozenset({'Umbrella'}), frozenset({'Yo-yo'})]

    # this is the dataset D as a list of sets
    D = list(map(set, dataset)) ## [{'Yo-yo', 'Mango', 'Ice-cream', 'Umbrella', 'Corn', 'Key-chain'}, {'Doll', 'Mango', 'Umbrella', 'Eggs', 'Onion', 'Key-chain'}, {'Yo-yo', 'Mango', 'Ice-cream', 'Eggs', 'Onion', 'Key-chain'}, {'Yo-yo', 'Eggs', 'Corn', 'Onion', 'Key-chain'}, {'Doll', 'Mango', 'Nintendo', 'Onion', 'Apple'}]

    F1, support_data = get_freq(D, C1, min_support, verbose=False) # get frequent 1-itemsets

    F = [F1] # list of frequent itemsets; initialized to frequent 1-itemsets

    k = 2 # the itemset cardinality
    while (len(F[k - 2]) > 0):
        Ck = apriori_gen(F[k-2], k) # generate candidate itemsets
        Fk, supK  = get_freq(D, Ck, min_support) # get frequent itemsets
        support_data.update(supK)# update the support counts to reflect pruning
        F.append(Fk)  # add the frequent k-itemsets to the list of frequent itemsets
        k += 1

    if verbose:
        # Print a list of all the frequent itemsets.
        for kset in F:
            for item in kset:
                print(""                     + "{"                     + "".join(str(i) + ", " for i in iter(item)).rstrip(', ')                     + "}"                     + ":  sup = " + str(round(support_data[item], 3)))

        # print all candidate itemsets
        print("\n\nAll candidate itemsets:")
        for key in support_data:
            print(""             + "{"             + "".join(str(i) + ", " for i in iter(key)).rstrip(', ')             + "}"             + ":  sup = "
                  + str(round(support_data[key], 3)))
    return F, support_data

def create_candidates(dataset, verbose=False):
    """Creates a list of candidate 1-itemsets from a list of transactions.

    Parameters
    ----------
    dataset : list
        The dataset (a list of transactions) from which to generate candidate
        itemsets.

    Returns
    -------
    The list of candidate itemsets (c1) passed as a frozenset (a set that is
    immutable and hashable).
    """
    c1 = [] # list of all items in the database of transactions
    for transaction in dataset:
        for item in transaction:
            if not [item] in c1:
                c1.append([item])
    c1.sort()

    if verbose:
        # Print a list of all the candidate items.
        print(""             + "{"             + "".join(str(i[0]) + ", " for i in iter(c1)).rstrip(', ')             + "}")

    # Map c1 to a frozenset because it will be the key of a dictionary.
    return list(map(frozenset, c1))

def get_freq(dataset, candidates, min_support, verbose=False):
    """

    This function separates the candidates itemsets into frequent itemset and infrequent itemsets based on the min_support,
	and returns all candidate itemsets that meet a minimum support threshold.

    Parameters
    ----------
    dataset : list
        The dataset (a list of transactions) from which to generate candidate
        itemsets.

    candidates : frozenset
        The list of candidate itemsets.

    min_support : float
        The minimum support threshold.

    Returns
    -------
    freq_list : list
        The list of frequent itemsets.

    support_data : dict
        The support data for all candidate itemsets.
    """

    # TODO
    freq_list = []
    support_data = {}

    # get support count for each candidate
    for transaction in dataset:
        for candidate in candidates:
            if candidate.issubset(transaction):
                if candidate in support_data:
                    support_data[candidate] += 1
                else:
                    # use update method to add new candidate to support_data
                    support_data.update({candidate: 1}) 

    # After getting the support count, calculate the support and filter out the infrequent itemsets
    for key in support_data:
        # support = (support count) / (total number of transactions)
        support_data[key] /= len(dataset)
        if support_data[key] >= min_support:
            freq_list.append(key)

    # print("support_data:")
    # print(support_data) ## {frozenset({'Corn'}): 0.4, frozenset({'Ice-cream'}): 0.4, frozenset({'Key-chain'}): 0.8, frozenset({'Mango'}): 0.8, frozenset({'Umbrella'}): 0.4, frozenset({'Yo-yo'}): 0.6, frozenset({'Doll'}): 0.4, frozenset({'Eggs'}): 0.6, frozenset({'Onion'}): 0.8, frozenset({'Apple'}): 0.2, frozenset({'Nintendo'}): 0.2}
    # print("frequent itemsets:")
    # print(freq_list) ## [frozenset({'Key-chain'}), frozenset({'Mango'}), frozenset({'Yo-yo'}), frozenset({'Eggs'}), frozenset({'Onion'})]


    return freq_list, support_data


def apriori_gen(freq_sets, k):
    """Generates candidate itemsets (via the F_k-1 x F_k-1 method).

    This part generates new candidate k-itemsets based on the frequent
    (k-1)-itemsets found in the previous iteration.

    The apriori_gen function performs two operations:
    (1) Generate length k candidate itemsets from length k-1 frequent itemsets
    (2) Prune candidate itemsets containing subsets of length k-1 that are infrequent

    Parameters
    ----------
    freq_sets : list
        The list of frequent (k-1)-itemsets.

    k : integer
        The cardinality of the current itemsets being evaluated.

    Returns
    -------
    candidate_list : list
        The list of candidate itemsets.
    """
    # TODO
    candidate_list = []
    # print(freq_sets) ## [frozenset({'Key-chain'}), frozenset({'Mango'}), frozenset({'Yo-yo'}), frozenset({'Eggs'}), frozenset({'Onion'})]
    
    # Generate length k candidate itemsets from length k-1 frequent itemsets (F_k-1 x F_k-1)
    for i in range(len(freq_sets)):
        for j in range(i+1, len(freq_sets)):
            first = list(freq_sets[i])
            second = list(freq_sets[j])
            # print(first)
            # print(second)

            # sort two list to ensure the order of items in the list is the same
            first.sort()
            second.sort()

            # if the first k-2 items are the same
            if first[:k-2] == second[:k-2]: 
                first.append(second[-1])
                candidate_list.append(first)
            #     print(first)
            # print("-----\n")

    # print("candidate_list:")
    # print(candidate_list) ## [['Key-chain', 'Mango'], ['Key-chain', 'Yo-yo'], ['Key-chain', 'Eggs'], ['Key-chain', 'Onion'], ['Mango', 'Yo-yo'], ['Mango', 'Eggs'], ['Mango', 'Onion'], ['Yo-yo', 'Eggs'], ['Yo-yo', 'Onion'], ['Eggs', 'Onion']]
    
    # Prune candidate itemsets containing subsets of length k-1 that are infrequent
    pruned_candidate_list = []
    freq_sets_list = []
    for s in freq_sets:
        s=list(s)
        s.sort()
        freq_sets_list.append(s)
    # print(freq_sets_list)

    for candidate in candidate_list:
        all_possible_candidate_subsets = combinations(candidate, k-1)

        passed = True

        # check if all the subsets of the candidate are in freq_sets_list
        for subset in all_possible_candidate_subsets:

            sorted_subset = list(subset)
            sorted_subset.sort()
            if (sorted_subset in freq_sets_list) and (sorted_subset not in pruned_candidate_list):
                continue
            else:
                passed = False
                break
        if passed:
            pruned_candidate_list.append(candidate)

    

    # turn list of frozensets
    pruned_candidate_list = list(map(frozenset, pruned_candidate_list))
    # print("pruned_candidate_list:")
    # print(pruned_candidate_list, "\n\n") ##
    return pruned_candidate_list



def loadDataSet(fileName, delim=','):
    fr = open(fileName)
    stringArr = [line.strip().split(delim) for line in fr.readlines()]
    return stringArr



def run_apriori(data_path, min_support, verbose=False):
    dataset = loadDataSet(data_path)
    # print(dataset) ## [['Corn', 'Ice-cream', 'Key-chain', 'Mango', 'Umbrella', 'Yo-yo'], ['Doll', 'Eggs', 'Key-chain', 'Mango', 'Onion', 'Umbrella'], ['Eggs', 'Ice-cream', 'Key-chain', 'Mango', 'Onion', 'Yo-yo'], ['Corn', 'Eggs', 'Key-chain', 'Onion', 'Yo-yo'], ['Apple', 'Doll', 'Mango', 'Nintendo', 'Onion']]
    F, support = apriori(dataset, min_support=min_support, verbose=verbose)
    return F, support



def bool_transfer(input):
    ''' Transfer the input to boolean type'''
    input = str(input)
    if input.lower() in ['t', '1', 'true' ]:
        return True
    elif input.lower() in ['f', '0', 'false']:
        return False
    else:
        raise ValueError('Input must be one of {T, t, 1, True, true, F, F, 0, False, false}')




if __name__ == '__main__':
    if len(sys.argv)==3:
        F, support = run_apriori(sys.argv[1], float(sys.argv[2]))
    elif len(sys.argv)==4:
        F, support = run_apriori(sys.argv[1], float(sys.argv[2]), bool_transfer(sys.argv[3]))
    else:
        raise ValueError('Usage: python apriori_templete.py <data_path> <min_support> <is_verbose>')

    '''
    Example: 
    
    python apriori_templete.py market_data_transaction.txt 0.5 
    
    python apriori_templete.py market_data_transaction.txt 0.5 True
    
    '''





