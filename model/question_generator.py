import spacy
import nltk.tree
import collections

import benepar
benepar.download('benepar_en3')

def find_word(tree, kind, parents=None):
    if parents is None:
        parents = []
    if not isinstance(tree, nltk.tree.Tree):
        return None, None
    if tree.label() == kind:
        return tree[0], parents
    parents.append(tree)
    for st in tree:
        n, p = find_word(st, kind, parents)
        if n is not None:
            return n, p
    parents.pop()
    return None, None

def find_subtrees(tree, kind, blocking_kinds=()):
    result = []
    if not isinstance(tree, nltk.tree.Tree):
        return result
    if tree.label() == kind:
        result.append(tree)
    if tree.label() not in blocking_kinds:
        for st in tree:
            result.extend(find_subtrees(st, kind))
    return result

def tree_to_str(tree, transform=lambda w: w):
    l = []
    def list_words(tree):
        if isinstance(tree, str):
            l.append(transform(tree))
        else:
            for st in tree:
                list_words(st)
    list_words(tree)
    if l[-1] == '.':
        l = l[:-1]
    return ' '.join(l)

def tree_to_nouns(tree, transform=lambda w: w):
    l = []
    def list_words(tree, noun=False):
        if isinstance(tree, str):
            if noun == True:
                l.append(transform(tree))
        else:
            for st in tree:
                if not isinstance(st, str):
                    if st.label() == 'NN':
                        noun = True
                    else:
                        noun = False
                list_words(st, noun)
    list_words(tree)
    if l[-1] == '.':
        l = l[:-1]
    return l

def make_determinate(w):
    if w.lower() in ('a', 'an'):
        return 'the'
    return w

def make_indeterminate(w):
    if w.lower() in ('the', 'his', 'her', 'their', 'its'):
        return 'a'
    return w

def pluralize(singular, plural, number):
    if number <= 1:
        return singular
    return plural

def count_labels(tree):
    counts = collections.defaultdict(int)
    def update_counts(node):
        counts[node.label()] += 1
        for child in node:
            if isinstance(child, nltk.tree.Tree):
                update_counts(child)
    update_counts(tree)
    return counts

def get_number(tree):
    if not isinstance(tree, nltk.tree.Tree):
        return 0
    if tree.label() == 'NN':
        return 1
    if tree.label() == 'NNS':
        return 2
    first_noun_number = None
    n_np_children = 0
    for subtree in tree:
        label = subtree.label() if isinstance(subtree, nltk.tree.Tree) else None
        if label == 'NP':
            n_np_children += 1
        if label in ('NP', 'NN', 'NNS') and first_noun_number is None:
            first_noun_number = get_number(subtree)
    if tree.label() == 'NP' and n_np_children > 1:
        return 2
    return first_noun_number or 0

def is_present_continuous(verb):
    return verb.endswith('ing')

class QuestionGenerator:
    def __init__(self):
        self.parser = benepar.Parser("benepar_en3")

    def generate_what_question(self, s):
        tree = self.parser.parse(s)[0]
        questions = []
        try:
            if len(tree) >= 2 and tree[0].label() == 'NP' and tree[1].label() == 'VP':
                np = tree[0]
                verb = None
                vp = tree[1]
                vnp = None

                while True:
                    verb, verb_parents = find_word(vp, 'VBG')
                    if verb is None:
                        break
                    if is_present_continuous(verb):
                        if len(verb_parents[-1]) > 1:
                            vnp = verb_parents[-1][1]
                        break
                    else:
                        vp = verb_parents[-1][1]
                to_be = pluralize('is', 'are', get_number(np))
                if vnp is not None and vnp.label() == 'NP':
                    questions.append(('What {} {} {}?'
                                     .format(to_be, 
                                             tree_to_str(np, make_determinate).lower(), 
                                             verb),
                                     tree_to_nouns(vnp)[-1]))
        except Exception as e:
            print(e)
        return questions

    def generate_is_there_question(self, s):
        tree = self.parser.parse(s)
        questions = []
        nps = find_subtrees(tree, 'NP', ('PP',))
        for np in nps:
            only_child_label = len(np) == 1 and next(iter(np)).label()
            if only_child_label in ('PRP', 'EX'):
                continue
            try:
                to_be = pluralize('Is', 'Are', get_number(np))
                questions.append('{} there {}?'
                                 .format(to_be, 
                                         tree_to_str(np, make_indeterminate).lower()))
            except Exception as e:
                print(e)
        return questions