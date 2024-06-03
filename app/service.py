import logging
import time
from datetime import datetime
from threading import Thread

import torch
import urllib3
from config import (
    APP_NAME,
    LLM_ENVIRONMENT,
    LLM_MODEL_NAME,
    MODEL_LOAD,
    NEW_TOKENS_DEFAULT,
    STREAM_DEFAULT,
    SYSTEM_PROMPT,
    TEMPERATURE_DEFAULT,
    TOP_K_DEFAULT,
    TOP_P_DEFAULT,
    WAIT_RESPONSE,
)
from fastapi.responses import JSONResponse, StreamingResponse
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    BitsAndBytesConfig,
    TextIteratorStreamer,
)

################################################################################
# CONSTANTS
################################################################################

# os.environ["CUDA_VISIBLE_DEVICES"] = "0"

model = None
tokenizer = None
model_ready = False
prompt_config = {}
prompt_template = ""

################################################################################
# STATIC INITIALIZATION
################################################################################

urllib3.disable_warnings(urllib3.exceptions.InsecureRequestWarning)  # disable invalid ssl warnings
logger = logging.getLogger(APP_NAME)
quantization_config = BitsAndBytesConfig(
    load_in_4bit=True,
    # bnb_4bit_use_double_quant=True,
    # bnb_4bit_quant_type="nf4",
    bnb_4bit_compute_dtype=torch.bfloat16,
)


def check_for_new_model():
    # immediately return and do not load model if LLM_MODEL_LOAD is false. false is set for int and uat envs
    if not MODEL_LOAD:
        logger.info("MODEL_LOAD is set false, so model won't be loaded.")
        return

    if LLM_ENVIRONMENT.upper() == "DEV":
        return

    load_model(LLM_MODEL_NAME)


def load_model(model_path):
    global model, tokenizer, model_ready

    logger.info(f"Loading LLM Model and Adapter from {model_path}.")

    new_model = AutoModelForCausalLM.from_pretrained(model_path, device_map="auto", torch_dtype=torch.float16)
    new_tokenizer = AutoTokenizer.from_pretrained(model_path, legacy=True)
    new_model.eval()

    # Update the global model, tokenizer, and model_ready flag
    model = new_model
    tokenizer = new_tokenizer
    model_ready = True

    logger.info("LLM Model is locked and loaded.")


def is_model_ready():
    return model_ready


def get_generic_message(message: str, stream: bool):
    def stream_message():
        yield from message

    if stream:
        return StreamingResponse(stream_message(), media_type="text/event-stream")
    else:
        return JSONResponse({"response": [{"role": "assistant", "content": message}]})


def completions_response(content):
    stream = content.get("stream", STREAM_DEFAULT)
    max_new_tokens = content.get("max_tokens", NEW_TOKENS_DEFAULT)
    temperature = content.get("temperature", TEMPERATURE_DEFAULT)
    top_p = content.get("top_p", TOP_P_DEFAULT)
    top_k = content.get("top_k", TOP_K_DEFAULT)

    # If model is not loaded in test environments, return a place holder.
    if not model_ready or not MODEL_LOAD:
        logger.warning("Model is not ready.")
        return get_generic_message(WAIT_RESPONSE, stream=stream)

    # Is used by Continue to generate a relevant title corresponding to the
    # model's response, however, the current prompt passed by Continue is not
    # good at obtaining a title from Code Llama's completion feature so we
    # use chat completion instead.
    messages = [{"role": "user", "content": content["prompt"]}]

    # Send back the response.
    return run_chat_completion(
        messages, stream=stream, max_new_tokens=max_new_tokens, temperature=temperature, top_p=top_p, top_k=top_k
    )


def chat_completions_response(content):
    messages = content["messages"]
    stream = content.get("stream", STREAM_DEFAULT)
    max_new_tokens = content.get("max_tokens", NEW_TOKENS_DEFAULT)
    temperature = content.get("temperature", TEMPERATURE_DEFAULT)
    top_p = content.get("top_p", TOP_P_DEFAULT)
    top_k = content.get("top_k", TOP_K_DEFAULT)

    # If model is not loaded in test environments, return a place holder.
    if not model_ready or not MODEL_LOAD:
        logger.warning("Model is not ready.")
        return get_generic_message(WAIT_RESPONSE, stream=stream)

    # Process messages
    if messages[0]["role"] == "assistant":
        messages[0]["role"] = "system"

    last_role = None
    remove_elements = []
    for i in range(len(messages)):
        if messages[i]["role"] == last_role:
            messages[i - 1]["content"] += "\n\n" + messages[i]["content"]
            remove_elements.append(i)
        else:
            last_role = messages[i]["role"]

    # remove messages in remove_elements
    final_messages = []
    for i in range(len(messages)):
        if i not in remove_elements:
            final_messages.append(messages[i])

    response = run_chat_completion(
        final_messages, stream=stream, max_new_tokens=max_new_tokens, temperature=temperature, top_p=top_p, top_k=top_k
    )

    # return {"response": [{"role": "assistant", "content": "".join(outputs)}]}
    return response


def get_prompt(messages: list[dict], system_prompt: str) -> str:
    texts = [f"### System Prompt\n{system_prompt}\n\n"]
    do_strip = False
    for message in messages:
        messageContent = message["content"].strip() if do_strip else message["content"]
        if message["role"] == "user":
            texts.append(f"### User Message\n{messageContent}\n\n")
        else:
            texts.append(f"### Assistant Response\n{messageContent}\n\n")
        do_strip = True
    texts.append("### Assistant Response\n")
    logger.info(f"REQUEST | User asks: {messages[-1]['content']}")
    return "".join(texts)


def run_chat_completion(
    messages: list[dict],
    stream: bool = STREAM_DEFAULT,
    max_new_tokens: int = NEW_TOKENS_DEFAULT,
    temperature: float = TEMPERATURE_DEFAULT,
    top_p: float = TOP_P_DEFAULT,
    top_k: int = TOP_K_DEFAULT,
) -> str:
    system_prompt: str = SYSTEM_PROMPT

    # get system prompt from messages
    for message in messages:
        if message["role"] == "system":
            system_prompt = message["content"]
            messages.remove(message)
            break

    prompt = get_prompt(messages, system_prompt)

    inputs = tokenizer([prompt], return_tensors="pt", add_special_tokens=False).to("cuda")

    streamer = TextIteratorStreamer(tokenizer, timeout=1000.0, skip_prompt=True, skip_special_tokens=True)

    generation_kwargs = {
        "inputs": inputs["input_ids"].to("cuda"),
        "attention_mask": inputs["attention_mask"].to("cuda"),
        "pad_token_id": tokenizer.pad_token_id,
        "eos_token_id": tokenizer.eos_token_id,
        "streamer": streamer,
        "max_new_tokens": max_new_tokens,
        "do_sample": True,
        "top_p": top_p,
        "top_k": top_k,
        "temperature": temperature,
        "repetition_penalty": 1.2,
        "num_return_sequences": 1,
    }

    if not stream:
        start = time.time()
        outputs = model.generate(**generation_kwargs)
        response = tokenizer.decode(outputs[0][inputs.input_ids.shape[1] :], skip_special_tokens=True)
        gen_duration = time.time() - start
        logger.info(f"Generation completed in {gen_duration:.2f} seconds")
        return JSONResponse({"response": [{"role": "assistant", "content": response}]})

    def generate_with_caution():
        # Since generate works in a thread, this function is used to catch exceptions raised from the thread, aka generation errors
        try:
            return model.generate(**generation_kwargs)
        except Exception as ex:  # burda skip prompt kısmı çalışmıyo nedense
            streamer.text_queue.put(ex)  # if an exception occurs, it is added to the stream queue to be handled later

    start = time.time()
    thread = Thread(target=generate_with_caution)
    thread.start()  # buraya exception handling

    def stream_generation():
        c = 0
        for next_word in streamer:
            if next_word:
                if isinstance(next_word, Exception):
                    logger.error("An Exception has been caught during generation")
                    raise next_word
                c += 1
                yield next_word
        gen_duration = time.time() - start
        logger.info(f"Generation completed in {gen_duration:.2f} seconds")
        logger.info(f"Generation speed: {c / gen_duration:.2f} words per second")

    return StreamingResponse(stream_generation(), media_type="text/event-stream")
