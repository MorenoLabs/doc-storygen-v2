# Copyright (c) Meta Platforms, Inc. and affiliates.

from contextlib import contextmanager
import logging
import re
import signal


import roman
import Levenshtein
from transformers import AutoTokenizer
from scipy.special import log_softmax

from storygen.common.llm.prompt import TemplatePromptBuilder


tokenizers = {}


def init_logging(logging_level):
    logging_level = logging_level.upper()
    assert logging_level in ['DEBUG', 'INFO', 'WARNING', 'ERROR', 'CRITICAL']
    logging.basicConfig(format='%(asctime)s %(levelname)-8s %(message)s',
                        level=logging_level,
                        datefmt='%Y-%m-%d %H:%M:%S')


class TimeoutException(Exception): pass

@contextmanager
def time_limit(seconds):
    def signal_handler(signum, frame):
        raise TimeoutException("Timed out!")
    signal.signal(signal.SIGALRM, signal_handler)
    signal.alarm(seconds)
    try:
        yield
    finally:
        signal.alarm(0)


def num_to_char(num, newline=False):
    if num > 26:
        return num_to_char(num // 26) + num_to_char(num % 26)
    new_char = chr(num-1 + ord('a')) # 1:a 2:b etc
    if newline:
        return '\n' + new_char
    else:
        return new_char


def num_to_roman(num, newline=False):
    new_num = roman.toRoman(num)
    if newline:
        return '\n' + new_num.lower()
    else:
        return new_num.lower()


class Filter:
    def __init__(self, filter_func):
        self.filter_func = filter_func

    @staticmethod
    def wrap_preprocessor(preprocessor, filter):
        return Filter(lambda s: filter(preprocessor(s)))
    
    def __call__(self, *args, **kwargs):
        try:
            return self.filter_func(*args, **kwargs)
        except:
            return self.filter_func(*args) # for any functions that don't take extra kwargs

    def __add__(self, other):
        return Filter(lambda s: self.filter_func(s) and other.filter_func(s))


def min_max_tokens_filter(min_tokens, max_tokens, tokenizer_model_string='gpt2', filter_empty=True):
    # the tokenizer model doesn't really matter. we're just counting tokens for filtering purposes
    global tokenizers
    if tokenizer_model_string in tokenizers:
        tokenizer = tokenizers[tokenizer_model_string]
    else:
        tokenizer = AutoTokenizer.from_pretrained(tokenizer_model_string)
        tokenizers[tokenizer_model_string] = tokenizer
    filter = Filter(lambda s: min_tokens <= len(tokenizer.encode(s.strip())) <= max_tokens)
    if filter_empty:
        filter = filter + Filter(lambda s: len(s.strip()) > 0)
    return filter


def levenshtein_ratio_filter(passages_to_match, threshold=0.8):
    # check if any subpassage of the generated passage are too similar to any passage in passages_to_match
    return Filter(lambda s: all([all([Levenshtein.ratio(sub_s, passage) < threshold for passage in passages_to_match]) for sub_s in s.split()]))


def word_filter(word_list):
    return Filter(lambda s: all([word not in s for word in word_list]))


def list_next_number_format_filter():
    # check if any list numbering e.g. "4." is preceded by a newline
    bad_regex = re.compile(r'[^\n]\d+\.')
    return Filter(lambda s: not bad_regex.search(s))


def wrap_filter_for_tuple(filter, index=0):
    return Filter(lambda s: filter(s[index]))


""" def extract_choice_logprobs(full_completion, choices=['yes', 'no'], default_logprobs=[-1e8, -1e8], case_sensitive=False):
    batch_logprobs = []
    for choice in full_completion['choices']:
        all_logprobs = choice['logprobs']['top_logprobs']
        found = False
        logprobs = [l for l in default_logprobs]
        for token_logprobs in all_logprobs: # look for the first position that has yes or no in the top few logprobs
            for key, value in token_logprobs.items():
                for i, choice in enumerate(choices):
                    if choice in key or (not case_sensitive and choice.lower() in key.lower()):
                        found = True
                        logprobs[i] = value
            if found:
                break
        batch_logprobs.append(log_softmax(logprobs))
    return batch_logprobs """


""" def extract_choice_logprobs(completion_response, choices=['yes', 'no'], default_logprobs=[-1e8, -1e8]):
Extracts the log probabilities for specified choices from the API completion.
    batch_logprobs = []
    all_logprobs = completion_response.choices[0].logprobs
    logprobs = [l for l in default_logprobs]
    for token_logprobs in all_logprobs:
        for key, value in token_logprobs.items():
            if key in choices:  # Exact match to 'yes' or 'no'
                idx = choices.index(key)
                logprobs[idx] = value
    
    batch_logprobs.append(log_softmax(logprobs))
    return batch_logprobs """


""" def extract_choice_logprobs(completion_response, choices=['yes', 'no'], default_logprobs=[-1e8, -1e8], case_sensitive=False):
    Extracts the log probabilities for specified choices from the API completion.
    print("entering extract_choice_logprobs")
    batch_logprobs = []
    content_logprobs = completion_response.choices[0].logprobs.content[0].top_logprobs  # Access the correct path to logprobs in the response
    print("content_logprobs: ", content_logprobs)

    # Iterate over each token's logprobs
    logprobs = default_logprobs.copy()

    # Iterate over each token's top_logprobs to extract specific choices 'yes' and 'no'
    for top_logprob in content_logprobs:
        token = top_logprob.token.lower()  # Normalize the token text to lowercase for comparison
        if token in choices:
            print("token: ", token)
            idx = choices.index(token)
            logprobs[idx] = top_logprob.logprob  # Update the log probability for the found choice

    batch_logprobs.append(log_softmax(logprobs))  # Apply log softmax to normalize the probabilities
    return batch_logprobs
 """



def extract_choice_logprobs(full_completion, choices=['yes', 'no'], default_logprobs=[-1e8, -1e8], case_sensitive=False):
    """ Extracts the log probabilities for specified choices from the API completion, considering case sensitivity. """
    #print("Entering extract_choice_logprobs")
    batch_logprobs = []
    
    for index, choice_data in enumerate(full_completion.choices):
        #print(f"Processing choice {index} with data: {choice_data}")
        all_logprobs = choice_data.logprobs.content  # Directly access the content
        #print(f"all_logprobs length: {len(all_logprobs)}")
        
        found = False
        logprobs = default_logprobs.copy()
        #print(f"Initial logprobs: {logprobs}")

        # Iterate over each token's logprobs to extract specific choices 'yes' and 'no'
        for token_idx, token_logprob in enumerate(all_logprobs):
            top_logprobs = token_logprob.top_logprobs  # Access the top_logprobs directly
            #print(f"Token {token_idx} top_logprobs: {top_logprobs}")
            
            for top_logprob in top_logprobs:
                token = top_logprob.token
                logprob = top_logprob.logprob
                #print(f"Inspecting token: {token} with logprob: {logprob}")
                
                if not case_sensitive:
                    token = token.lower()
                
                for idx, choice in enumerate(choices):
                    compare_choice = choice if case_sensitive else choice.lower()
                    if compare_choice == token:
                        logprobs[idx] = logprob
                        found = True
                        #print(f"Match found for '{token}'. Updating logprob at index {idx}: {logprobs}")
                        break
                if found:
                    break
            if found:
                break

        batch_logprobs.append(log_softmax(logprobs))
        #print(f"Normalized logprobs for batch index {index}: {batch_logprobs[-1]}")

    return batch_logprobs
