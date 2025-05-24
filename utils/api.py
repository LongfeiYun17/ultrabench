
def api_price(model="gpt-4.1", input_tokens=0, completion_tokens=0):
    INPUT_TOKEN_PRICE_GPT_4_1 = 2. / 1e6
    OUTPUT_TOKEN_PRICE_GPT_4_1 = 8. / 1e6
    INPUT_TOKEN_PRICE_GPT_4_1_MINI = 0.4 / 1e6
    OUTPUT_TOKEN_PRICE_GPT_4_1_MINI = 1.6 / 1e6
    INPUT_TOKEN_PRICE_GPT_4_1_NANO = 0.10 / 1e6
    OUTPUT_TOKEN_PRICE_GPT_4_1_NANO = 0.4 / 1e6
    if model == "gpt-4.1":
        return input_tokens * INPUT_TOKEN_PRICE_GPT_4_1 + completion_tokens * OUTPUT_TOKEN_PRICE_GPT_4_1
    elif model == "gpt-4.1-mini":
        return input_tokens * INPUT_TOKEN_PRICE_GPT_4_1_MINI + completion_tokens * OUTPUT_TOKEN_PRICE_GPT_4_1_MINI
    elif model == "gpt-4.1-nano":
        return input_tokens * INPUT_TOKEN_PRICE_GPT_4_1_NANO + completion_tokens * OUTPUT_TOKEN_PRICE_GPT_4_1_NANO
    else:
        raise ValueError(f"Invalid model: {model}")