# Copyright (c) Meta Platforms, Inc. and affiliates.

import logging
import os
import time

from openai import OpenAI
import openai
import backoff
from dotenv import load_dotenv

load_dotenv()  # take environment variables from .env.
cliento = OpenAI(api_key=os.environ['OPENAI_API_KEY'])

import anthropic

clienta = anthropic.Anthropic(
    # defaults to os.environ.get("ANTHROPIC_API_KEY")
    api_key=os.environ.get("ANTHROPIC_API_KEY")
)


import ollama


from storygen.common.server import ServerConfig
from storygen.common.util import *


models = {} # model string -> model object


class SamplingConfig:
    def __init__(self, 
                 server_config,
                 prompt_format,
                 max_tokens=None,
                 temperature=None,
                 top_p=None,
                 frequency_penalty=None,
                 presence_penalty=None,
                 stop=None,
                 n=None,
                 logit_bias=None,
                 logprobs=None,
                 top_logprobs=None):
        self.server_config = server_config
        self.prompt_format = prompt_format
        self.max_tokens = max_tokens
        self.temperature = temperature
        self.top_p = top_p
        self.frequency_penalty = frequency_penalty
        self.presence_penalty = presence_penalty
        self.stop = stop
        self.n = n
        self.logit_bias = logit_bias
        self.logprobs = logprobs
        self.top_logprobs = top_logprobs
    
    @staticmethod
    def from_config(config):
        return SamplingConfig(
            server_config=ServerConfig.from_config(config),
            prompt_format=config['prompt_format'],
            max_tokens=config.get('max_tokens', None),
            temperature=config.get('temperature', None),
            top_p=config.get('top_p', None),
            frequency_penalty=config.get('frequency_penalty', None),
            presence_penalty=config.get('presence_penalty', None),
            stop=config.get('stop', None),
            n=config.get('n', None),
            logit_bias=config.get('logit_bias', None),
            logprobs=config.get('logprobs', None),
            top_logprobs=config.get('top_logprobs', None)
        )
    
    def __getitem__(self, key):
        return getattr(self, key)
    
    def dict(self):
        d = {'model': self.server_config.engine}
        for attr in ['max_tokens', 'temperature', 'top_p', 'frequency_penalty', 'presence_penalty', 'stop', 'n', 'logit_bias', 'logprobs', 'top_logprobs']:
            if getattr(self, attr) is not None:
                d[attr] = getattr(self, attr)
        return d


class LLMClient:
    def __init__(self):
        self.warned = {'vllm_logit_bias': False}

    def call_with_retry(self, prompt_builder, sampling_config, postprocessor=None, filter=lambda s: len(s.strip()) > 0, max_attempts=5, **kwargs):
        for _ in range(max_attempts):
            try:
                completions, full_completion_object = self(prompt_builder, sampling_config, **kwargs)
            except Exception as e:
                logging.error(f"Atempt failed with Error: {e}")
                continue
            if postprocessor is not None:
                completions = postprocessor(completions, full_completion_object=full_completion_object)
            completions = [c for c in completions if filter(c)]
            if len(completions) > 0 or kwargs.get('empty_ok', False):
                if kwargs.get('return_full_completion', False):
                    return completions, full_completion_object
                else:
                    return completions
        logging.error(f"Failed to get a valid completion after {max_attempts} attempts.")
        raise RuntimeError(f"Failed to get a valid completion after {max_attempts} attempts.")
    
    #@backoff.on_exception(backoff.expo, openai.RateLimitError)
    def __call__(self, prompt_builder, sampling_config, **kwargs):
        if sampling_config.server_config['server_type'] == 'openai':
            cliento.api_key = os.environ['OPENAI_API_KEY']
            cliento.api_base = 'https://api.openai.com/v1'
        elif sampling_config.server_config['server_type'] == 'vllm':
            cliento.api_key = "EMPTY"
            cliento.api_base = sampling_config.server_config['host'] + ':' + str(sampling_config.server_config['port']) + '/v1'
            if 'logit_bias' in sampling_config.dict():
                if not self.warned['vllm_logit_bias']:
                    logging.warning(f"Logit bias is not supported for vllm server.")
                    self.warned['vllm_logit_bias'] = True
        elif sampling_config.server_config['server_type'] == 'ollama':
            pass
        else:
            raise NotImplementedError(f"Engine type {self.sampling_config.server_config['server_type']} not implemented.")
        
        prompt = prompt_builder.render_for_llm_format(sampling_config.prompt_format)
        #print("this is the prompt", prompt)
        logging.debug(f"Prompt: {prompt}")

        if sampling_config['prompt_format'] == 'openai-chat':
            with time_limit(kwargs.get('time_limit', 30)):
                """ completion = clienta.messages.create(messages=prompt, model="claude-3-opus-20240229",
                max_tokens=1000,
                temperature=0.0,
                system="Respond only in Yoda-speak.") 
                print(completion.content) #! ANTHROPIC """
                #completion = ollama.chat(model='llama2', messages=prompt) #! OLLAMA
                #print(completion['message']['content'])
                try:
                    #print("this is the full prompt send to with time limit function", prompt)
                    #print("config", sampling_config.dict())
                    completion = cliento.chat.completions.create(messages=prompt, **sampling_config.dict())
                    #print("this is the completion", completion.choices[0].message.content)
                except Exception as e:
                    logging.error(f"API call failed with error: {e}")
                    raise
            logging.debug(f"Completion: {completion.choices[0].message.content}")
            texts = [c.message.content for c in completion.choices]
            
            #texts = [completion['message']['content']]
            # text_block = completion.content[0]  # Access the TextBlock from the list
            # texts = text_block.text
            # #print(texts)
            # strip response prefix
            if prompt_builder.response_prefix is not None:
                for i, text in enumerate(texts):
                    if text.startswith(prompt_builder.response_prefix.format()):
                        texts[i] = text[len(prompt_builder.response_prefix.format()):]
        else:
            params = sampling_config.dict()
            if 'logit_bias' in params:
                del params['logit_bias'] # vllm doesn't yet support logit bias
            with time_limit(kwargs.get('time_limit', 30)):
                completion = cliento.chat.completions.create(prompt=prompt, **params) 
            #logging.debug(f"Completion: {completion.choices[0].text}")
            texts = [c.message.content for c in completion.choices]
        
        if prompt_builder.output_prefix is not None:
            for i, text in enumerate(texts):
                texts[i] = prompt_builder.output_prefix.rstrip() + ' ' + text.lstrip()
        from colorama import Fore
        #print(f"{Fore.RED}OUTPUT{Fore.WHITE}")
        #print("this is the text", texts)
        #print("this is the completion", completion.choices[0].message.content)
        #print(f"{Fore.RED}OUTPUT END{Fore.WHITE}")
        #print("exit LLM call function with two outputs---------------------------")
        return texts, completion