def createResponse(messages, llm, tokenizer, temperature=0.0):
    text = tokenizer.apply_chat_template(
        messages, tokenize=False, add_generation_prompt=True
    )

    model_inputs = tokenizer([text], return_tensors="pt").to("cuda")

    generated_ids = llm.generate(
        model_inputs.input_ids, max_new_tokens=512, temperature=temperature
    )

    generated_ids = [
        output_ids[len(input_ids) :]
        for input_ids, output_ids in zip(model_inputs.input_ids, generated_ids)
    ]

    response = tokenizer.batch_decode(generated_ids, skip_special_tokens=True)[0]

    return response


def promptData(
    song_descriptions: str | list[str], lyrics: str, image_style: str | None
) -> str:
    request_text = (
        "I will provide a song description and lyrics in English and Hebrew. "
        + "Your task is to generate a 77-token maximum image description in English that captures the core emotions and messages of the song. "
        + "This description will be used for text-to-image generation. Focus on emotions and key messages."
    )

    prompt = request_text

    if image_style is not None and len(image_style) > 0:
        prompt += "Requirements: " + image_style

    if len(song_descriptions) > 0:
        prompt += " Song descpriptions:"

        for index, description in enumerate(song_descriptions):
            prompt += f" {index + 1}- {description}"

    if len(lyrics) > 0:
        prompt += " Lyrics: " + lyrics

    print(f"{prompt=}")

    return prompt
