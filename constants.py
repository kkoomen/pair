from helpers import load_system_prompt_template

# Specifies the approaches, attack system prompts and judge system prompts for
# a given mode.

hist_rp_template = load_system_prompt_template("historical-roleplay")
hist_rp_lang_template = load_system_prompt_template("historical-roleplay-lang")

judge_prompt = load_system_prompt_template("judge")
judge_lang_prompt = load_system_prompt_template("judge-lang")

MODES = {
    # Complete benchmark for all approaches using the misuse behaviors.
    "misuse": {
        "approaches": [
            "Roleplay",
            "Hist-Roleplay",
            "Hist-Roleplay-NL",
            "Hist-Roleplay-DE",
            "Hist-Roleplay-FR",
        ],
        "system_prompts": [
            load_system_prompt_template("roleplay"),
            hist_rp_template,
            hist_rp_lang_template.replace("[[LANG]]", "Dutch"),
            hist_rp_lang_template.replace("[[LANG]]", "German"),
            hist_rp_lang_template.replace("[[LANG]]", "French"),
        ],
        "judge_system_prompts": [
            judge_prompt,
            judge_prompt,
            judge_lang_prompt.replace("[[LANG]]", "Dutch"),
            judge_lang_prompt.replace("[[LANG]]", "German"),
            judge_lang_prompt.replace("[[LANG]]", "French"),
        ],
    },

    # When --raw-benchmark is passed for benchmark without attack model
    "misuse-raw": {
        "approaches": ["Baseline"],
        "system_prompts": [None],
        "judge_system_prompts": [judge_prompt],
    },

    # This is only used for raw benchmarks (i.e., without attack model)
    # on the benign behaviors.
    "benign": {
        "approaches": ["Benign"],
        "system_prompts": [None],
        "judge_system_prompts": [load_system_prompt_template("judge-benign")],
    },

    # This is only used for raw benchmarks (i.e., without attack model)
    # on the benign behaviors.
    "defense": {
        "approaches": ["Defense"],
        "system_prompts": [load_system_prompt_template("defense")],
        "judge_system_prompts": [load_system_prompt_template("judge-defense")],
    },
}
