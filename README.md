# nq2sq - Natural Questions to Search Queries
This is a language model made to convert natural language question to search queries.

Upon first glance at the problem, using an LLM is the obvious solution. I then started to look into small models that would fit the requirement of running in 100ms or less on a consumer grade cpu, and the small variant of the T5 family of models was the one that met that. The "flan" versions of the T5 family are enhanced versions of the T5 models that have been finetuned in a mixture of tasks.

The next step, and arguably the most important, was to source the data. I initially searched around for a dataset mapping natural langauge questions to search queries, but came up empty handed. This lead me to the conclusion that I had to generate these mappings myself. I decided to pull natural language questions from QA datasets, and generating the adequate queries using a state of the art LLM, from which our smaller model can learn from.

I played around with different models to test their performance and cost effectiveness, and settled with OpenAI's GPT3.5 Turbo model. I began with 2000 questions to finetune the flan-t5-small model with, but that turned out terribly. I then slowly increased again and again until I hit about 10,000 data entries, and realized I would need a lot more data, which would end up being costly with GPT3.5 Turbo providing the mappings. This took me back to assessing cheaper models, and found Google's Gemini Pro model to have great accuracy while being a fraction of the cost. With this new model, I slowly increased the number of mappings until I hit about 80,000 mappings.

[Live Demo](https://nq2sq.rehanzo.com)
[API](https://nq2sq.rehanzo.com/api/)

The 'index.html' file is what the live demo runs off of.

## Usage
The api uses the [ctranslate]() library to interface with the model. Therefore, one needs to run the following prior to starting up the api, with the ctranslate module installed:
```bash
ct2-transformers-converter --quantization int8 --model rehanzo/nq2sq --output_dir nq2sq-ct
```

Then, we can start the api by running the command:
```bash
uvicorn api:app --host 0.0.0.0 --port 8000
```
Which will start the api on 127.0.0.1:8000.

## Dataset
The following is a breakdown of the composition of the final set of questions that were sent to Gemini to produce mappings for:
- [HotpotQA](https://hotpotqa.github.io/) (~39,000) - Question answering dataset featuring natural, multi-hop questions
- [Natural Questions](https://ai.google.com/research/NaturalQuestions) (~15,000) - NL questions pulled from real google queries
- [FreebaseQA](https://github.com/kelvin-jiang/FreebaseQA) (~20,000) - Data set for open-domain QA over the Freebase knowledge graph
- [GraphQuestions](https://github.com/ysu1989/GraphQuestions) (~5,000) - GraphQuestions is a characteristic-rich dataset for factoid question answering
- A handful of my own generated questions using GPT4 (~800) - aimed at generating questions that are more LLM focused (ex. Generate a table that..., Give me a list of the top five...., etc.)
Which gives us a total of ~80,000 entries.

My aim was to bring in a diverse number of questions by pulling from many sources. Some sources contain more simple questions (Natural Questions), while others contains more complex questions (HotpotQA). Using python, I pulled the data from their source json and jsonl files and put them into csv files. I then used another python script to fill the csv's with search query mappings generated from Gemini, and then combined the resulting files into one to finetune the model on.

The final file is contained in the `train.csv` file.

## Training
Given that it would likely take forever to finetune the model on my local hardware (6 year old second hand laptop), I used HuggingFace Spaces for finetuning. I played around with the training hyperparameters to obtain the best performing end model. Here is a list of the hyperparameters used:
```
"seed": 42,
"lr": 0.0001,
"epochs": 1,
"max_seq_length": 128,
"max_target_length": 128,
"batch_size": 8,
"warmup_ratio": 0.1,
"gradient_accumulation": 1,
"optimizer": "adamw_torch",
"scheduler": "constant_with_warmup",
"weight_decay": 0,
"max_grad_norm": 1,
"logging_steps": -1,
"evaluation_strategy": "epoch",
"auto_find_batch_size": false,
"save_total_limit": 1,
"save_strategy": "epoch",
"peft": false,
"quantization": null,
"lora_r": 16,
"lora_alpha": 32,
"lora_dropout": 0.05,
"target_modules": [
  "all-linear"
]
```
