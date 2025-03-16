import os
import time
import random
import openai
import logging
from packaging.version import parse as parse_version

IS_OPENAI_V1 = parse_version(openai.__version__) >= parse_version('1.0.0')

if IS_OPENAI_V1:
    from openai import APIError, APIConnectionError, RateLimitError
else:
    from openai.error import APIError, APIConnectionError, RateLimitError

class ClientError(RuntimeError):
    pass
def get_content(query, base_url, model_name):
    API_KEY = os.environ.get("OPENAI_API_KEY", "EMPTY")
    API_REQUEST_TIMEOUT = int(os.getenv('OPENAI_API_REQUEST_TIMEOUT', '99999'))
    if IS_OPENAI_V1:
        import httpx
        client = openai.OpenAI(
            api_key=API_KEY,
            base_url=base_url,
            timeout=httpx.Timeout(API_REQUEST_TIMEOUT),
        )
    else:
        client = None
    messages = [{'role': 'user', 'content': query}]
    call_args = dict(
            model=model_name,
            messages=messages,
            temperature=0.6,
            top_p=0.95,
            max_tokens=32768,
        )
    if IS_OPENAI_V1:
            call_args['extra_body'] = {}
            extra_args_dict = call_args['extra_body']
    else:
        extra_args_dict = call_args
    extra_args_dict.update({
        'top_k': 40,
    })
    
    if IS_OPENAI_V1:
        call_func = client.chat.completions.create
        call_args['timeout'] = API_REQUEST_TIMEOUT
    else:
        call_func = openai.ChatCompletion.create
        call_args['api_key'] = API_KEY
        call_args['api_base'] = base_url
        call_args['request_timeout'] = API_REQUEST_TIMEOUT

    result = ''
    try:
        completion = call_func(**call_args, )
        result = completion.choices[0].message.content              
    except AttributeError as e:
        err_msg = getattr(completion, "message", "")
        if err_msg:
            time.sleep(random.randint(25, 35))
            raise ClientError(err_msg) from e
        raise ClientError(err_msg) from e
    except (APIConnectionError, RateLimitError) as e:
        err_msg = e.message if IS_OPENAI_V1 else e.user_message
        time.sleep(random.randint(25, 35))
        raise ClientError(err_msg) from e
    except APIError as e:
        err_msg = e.message if IS_OPENAI_V1 else e.user_message
        if "maximum context length" in err_msg:  # or "Expecting value: line 1 column 1 (char 0)" in err_msg:
            logging.warn(f"max length exceeded. Error: {err_msg}")
            return {'gen': "", 'end_reason': "max length exceeded"}
        time.sleep(1)
        raise ClientError(err_msg) from e
    return result

if __name__ == "__main__":
    conversation_history = []
    user_input = "Hello!"
    res = get_content(user_input, "http://10.77.249.36:8030/v1", "Qwen/QwQ")
    print(f"Response: {res}")

    user_input = "How are you?"
    res = get_content(user_input, "http://10.77.249.36:8030/v1", "Qwen/QwQ")
    print(f"Response: {res}")