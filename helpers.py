def load_system_prompt_template(filename):
    return open(f"system_prompt_templates/{filename}.txt", "r", encoding="utf-8").read()
