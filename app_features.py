import time
import json
from preprocessing import FileIO
from typing import List
import tiktoken 
from loguru import logger
from prompt_templates import context_block, question_answering_prompt_series
import streamlit as st  

@st.cache_data
def load_content_cache(data_path: str):
    data = FileIO().load_parquet(data_path)
    content_data = {d['doc_id']: d['content'] for d in data}
    return content_data

@st.cache_data
def load_data(data_path: str):
    with open(data_path, 'r') as f:
        data = json.load(f)
    return data

def convert_seconds(seconds: int):
    """
    Converts seconds to a string of format Hours:Minutes:Seconds
    """
    return time.strftime("%H:%M:%S", time.gmtime(seconds))

def generate_prompt_series(query: str, results: List[dict]) -> str:
    """
    Generates a prompt for the OpenAI API by joining the context blocks of the top results.
    Provides context to the LLM by supplying the summary, guest, and retrieved content of each result.

    Args:
    -----
        query : str
            User query
        results : List[dict]
            List of results from the Weaviate client
    """
    context_series = '\n'.join([context_block.format(summary=res['summary'], guest=res['guest'], \
                                         transcript=res['content']) for res in results]).strip()
    prompt = question_answering_prompt_series.format(question=query, series=context_series)
    return prompt

def validate_token_threshold(ranked_results: List[dict], 
                             base_prompt: str,
                             query: str,
                             tokenizer: tiktoken.Encoding, 
                             token_threshold: int,
                             content_field: str='content', 
                             verbose: bool = False
                             ) -> List[dict]:
        """
        Validates that prompt is below the set token threshold by adding lengths of:
            1. Base prompt
            2. User query
            3. Context material
        If threshold is exceeded, context results are reduced incrementally until the 
        combined prompt tokens are below the threshold. This function does not take into
        account every token passed to the LLM, but it is a good approximation.
        """
        overhead_len = len(tokenizer.encode(base_prompt.format(question=query, series='')))
        context_len = _get_batch_length(ranked_results, tokenizer, content_field=content_field)
    
        token_count = overhead_len + context_len
        if token_count > token_threshold:
            print('Token count exceeds token count threshold, reducing size of returned results below token threshold')
            
            while token_count > token_threshold and len(ranked_results) > 1:
                num_results = len(ranked_results)
                
                # remove the last ranked (most irrelevant) result
                ranked_results = ranked_results[:num_results-1]
                # recalculate new token_count
                token_count = overhead_len + _get_batch_length(ranked_results, tokenizer, content_field=content_field)

        if verbose:
            logger.info(f'Total Final Token Count: {token_count}')
        return ranked_results

def _get_batch_length(ranked_results: List[dict], 
                      tokenizer: tiktoken.Encoding, 
                      content_field: str='content'
                      ) -> int:
    '''
    Convenience function to get the length in tokens of a batch of results 
    '''
    contexts = tokenizer.encode_batch([r[content_field] for r in ranked_results])
    context_len = sum(list(map(len, contexts)))
    return context_len

def search_result(i: int, 
                  url: str, 
                  title: str, 
                  content: str,
                  guest: str,
                  length: str,
                  space: str='&nbsp; &nbsp;'
                 ) -> str:
    
    '''
    HTML to display search results.

    Args:
    -----
    i: int
        index of search result
    url: str
        url of YouTube video 
    title: str
        title of episode 
    content: str
        content chunk of episode
    '''
    return f"""
        <div style="font-size:120%;">
            {i + 1}.<a href="{url}">{title}</a>
        </div>

        <div style="font-size:95%;">
            <p>Episode Length: {length} {space}{space} Guest: {guest}</p>
            <div style="color:grey;float:left;">
                ...
            </div>
            {content}
        </div>
    """


# Additional function added to expand the window of the context passed to the LLM. 

from itertools import groupby

@st.cache_data
def prepare_expanded_content(cache_data_path: str, window_size: int=2) -> dict:
    '''
    Augments the content for each document with the content from the preceeding and 
    following documents from the same window based the window_size argument. 
    '''

    data = FileIO().load_parquet(cache_data_path)

    episodes = []
    for key, group in groupby(data, lambda x: x['video_id']):
        episode = [chunk for chunk in group]
        episodes.append(episode)

    parent_chunks = {}
    for episode in episodes:
        contents = [d['content'] for d in episode]
        for i, d in enumerate(episode):
            doc_id = d['doc_id']
            start = max(0, i-window_size)
            end = i+window_size+1
            chunk = ' '.join(contents[start:end])
            parent_chunks[doc_id] = chunk
    return parent_chunks


def expand_response_content(response: List, expanded_content_map: dict) :
    '''
    Uses the expanded content map to expand the content in each of the search responses 
    to be sent to the LLM. 
    '''

    for doc in response:
        doc['content'] = expanded_content_map[doc['doc_id']]

    return response