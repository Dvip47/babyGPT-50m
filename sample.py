def generate_text(prompt, max_tokens=80, temperature=0.5, top_k=40):
    # --- 1. normalise prompt ---
    prompt = prompt.rstrip()
    if not prompt.endswith("A:"):
        prompt += "\nA:"

    # --- 2. encode & sample ---
    idx = torch.tensor(encode(prompt), dtype=torch.long)[None, ...]
    out = model.generate(
        idx,
        max_new_tokens=max_tokens,
        temperature=temperature,
        top_k=top_k
    )[0].tolist()

    full = decode(out)

    # --- 3. post‑process: cut off at new block or new Q: --------------
    # everything *after* the original prompt
    answer = full[len(prompt):]

    # stop at first unwanted marker
    for marker in ["\n[", "\nQ:", "\n\n"]:
        if marker in answer:
            answer = answer.split(marker, 1)[0]
    answer = answer.strip()

    # --- 4. fallback if answer too short / nonsense ------------------
    # naive heuristic: <4 tokens or still empty
    if len(answer.split()) < 4:
        return "Sorry, I don't know."

    return answer
