# Thesis BSc Artificial Intelligence

Trying to jailbreak LLMs.

# Getting started

Clone this repository:

```
$ git clone https://github.com/kkoomen/bsc-thesis && bsc-thesis
```

Setup a virtual env and install dependencies:

```
$ python3 -m venv env
$ source env/bin/activate
$ pip3 install -r requirements.txt
```

Next, run `cp .env.example .env` and get yourself the following API keys:
- (paid) [OpenAI API key](https://platform.openai.com/settings/organization/api-keys) to use GPT-4.1 as a judge model.
- (paid) [TogetherAI API key](https://api.together.ai/settings/api-keys) to use Mixtral-8x7B-Instruct as the attack LLM.
- (free) [Hugging Face API key](https://huggingface.co/settings/tokens) to use the LLaMA-2-7B and Vicuna-13B as target models.
