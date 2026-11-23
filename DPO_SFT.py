"""Run educational Phi-2 LoRA SFT followed by DPO, or validate the examples offline."""

import argparse


SFT_DATA = [{'messages': [{'role': 'system', 'content': 'You are a helpful, concise, and accurate AI assistant.'},
               {'role': 'user', 'content': 'What is the capital of France?'},
               {'role': 'assistant', 'content': 'The capital of France is Paris.'}]},
 {'messages': [{'role': 'system', 'content': 'You are a helpful, concise, and accurate AI assistant.'},
               {'role': 'user',
                'content': 'Write a Python function that returns the factorial of a number using '
                           'recursion.'},
               {'role': 'assistant',
                'content': 'def factorial(n):\n'
                           '    if n <= 1:\n'
                           '        return 1\n'
                           '    return n * factorial(n - 1)'}]},
 {'messages': [{'role': 'system', 'content': 'You are a helpful, concise, and accurate AI assistant.'},
               {'role': 'user',
                'content': 'Explain the difference between supervised and unsupervised learning in one '
                           'paragraph.'},
               {'role': 'assistant',
                'content': 'Supervised learning trains a model on labeled input/output pairs, learning '
                           'to map inputs to known outputs. Unsupervised learning finds patterns in '
                           'unlabeled data without any target outputs. Supervised learning is used for '
                           'classification and regression; unsupervised learning is used for '
                           'clustering, dimensionality reduction, and anomaly detection.'}]},
 {'messages': [{'role': 'system', 'content': 'You are a helpful, concise, and accurate AI assistant.'},
               {'role': 'user', 'content': "Translate 'Good morning, how are you?' into Spanish."},
               {'role': 'assistant', 'content': 'Buenos días, ¿cómo estás?'}]},
 {'messages': [{'role': 'system', 'content': 'You are a helpful, concise, and accurate AI assistant.'},
               {'role': 'user',
                'content': 'What are three key advantages of using transformers over RNNs for NLP '
                           'tasks?'},
               {'role': 'assistant',
                'content': 'Three key advantages of transformers over RNNs are: first, parallel '
                           'processing of sequences rather than sequential computation, enabling much '
                           'faster training; second, the self-attention mechanism captures long-range '
                           'dependencies without the vanishing gradient problems that affect RNNs; and '
                           'third, transformers scale more effectively with larger datasets and model '
                           'sizes, producing better performance as compute increases.'}]},
 {'messages': [{'role': 'system', 'content': 'You are a helpful, concise, and accurate AI assistant.'},
               {'role': 'user', 'content': 'I need to cancel my subscription. What should I do?'},
               {'role': 'assistant',
                'content': 'To cancel your subscription, go to your Account Settings, select Billing or '
                           'Subscription, and click Cancel Subscription. You will keep access until the '
                           'end of your current billing period. If you have trouble finding this '
                           'option, contact support with your account email and they will assist '
                           'you.'}]}]


DPO_DATA = [{'prompt': 'System: You are a helpful, concise, and accurate AI assistant.\n'
            'Instruct: Explain what machine learning is.\n'
            'Output: ',
  'chosen': 'Machine learning is a subset of artificial intelligence where systems learn from data to '
            'improve their performance on tasks without being explicitly programmed. Models identify '
            'patterns in training data and use them to make predictions or decisions on new, unseen '
            'data.',
  'rejected': 'Machine learning is a thing where computers learn stuff from data. It is pretty '
              'complicated and involves a lot of math. There are many types of machine learning like '
              'supervised and unsupervised and they all work differently in various ways that are hard '
              'to explain briefly.'},
 {'prompt': 'System: You are a helpful, concise, and accurate AI assistant.\n'
            'Instruct: What is the best way to learn Python?\n'
            'Output: ',
  'chosen': 'The most effective approach is: start with the official Python tutorial to learn syntax, '
            "then work through a structured course like CS50P or Python.org's beginner guide, practice "
            "daily by building small projects, and read other people's code on GitHub. Consistency "
            'matters more than intensity.',
  'rejected': 'You should definitely learn Python because it is a great language. There are many '
              'resources online. Some people like books and some people like videos. You should try '
              'different things and see what works for you. The main thing is to practice a lot and '
              'never give up even when it gets hard.'},
 {'prompt': 'System: You are a helpful, concise, and accurate AI assistant.\n'
            'Instruct: How do I reverse a string in Python?\n'
            'Output: ',
  'chosen': 'You can reverse a string in Python using slice notation:\n'
            '\n'
            '```python\n'
            'reversed_str = original_str[::-1]\n'
            '```\n'
            '\n'
            'This creates a new string with all characters in reverse order. For example, '
            "`'hello'[::-1]` returns `'olleh'`.",
  'rejected': 'Reversing a string is something you can do in Python. You would need to use some kind of '
              'loop or maybe a built-in function. Python has many ways to work with strings. You could '
              'also convert it to a list first. There are different approaches and they all work in '
              'their own way.'},
 {'prompt': 'System: You are a helpful, concise, and accurate AI assistant.\n'
            'Instruct: What should I do if my code has a bug?\n'
            'Output: ',
  'chosen': 'Debug systematically: first reproduce the bug reliably, then isolate where it occurs by '
            'adding print statements or using a debugger (pdb for Python). Read any error messages '
            'carefully, check your assumptions about what the code does, and search for the error '
            'message online if it is unclear. Fix one thing at a time and test after each change.',
  'rejected': 'Bugs happen to everyone so do not feel bad. You should look at your code carefully and '
              'try to find where the problem is. Sometimes restarting your computer helps or closing '
              'and reopening your editor. You could also try asking a friend or posting on Stack '
              'Overflow.'},
 {'prompt': 'System: You are a helpful, concise, and accurate AI assistant.\n'
            'Instruct: What is a REST API?\n'
            'Output: ',
  'chosen': 'A REST API (Representational State Transfer Application Programming Interface) is a '
            'standard way for systems to communicate over HTTP. It uses standard HTTP methods (GET, '
            'POST, PUT, DELETE) to perform operations on resources identified by URLs. REST APIs return '
            'data, typically JSON, allowing different applications and languages to exchange '
            'information in a consistent, stateless way.',
  'rejected': 'REST API stands for something with the letters REST. It is a way that computers talk to '
              'each other on the internet. APIs are important in modern software development. Many big '
              'companies use APIs. You can use them to get data from websites or do things on other '
              'services.'}]

ROLE_NAMES = {"system": "System", "user": "Instruct", "assistant": "Output"}


def format_messages(messages):
    return "\n".join(f"{ROLE_NAMES[item['role']]}: {item['content']}" for item in messages)


def validate_data():
    for example in SFT_DATA:
        messages = example["messages"]
        if [item["role"] for item in messages] != ["system", "user", "assistant"]:
            raise ValueError("Expected one system, user, and assistant message in each SFT example.")
        if any(not item["content"].strip() for item in messages):
            raise ValueError("SFT messages must have nonempty content.")
    for example in DPO_DATA:
        if any(not example[field].strip() for field in ("prompt", "chosen", "rejected")):
            raise ValueError("Preference examples require a nonempty prompt and two responses.")
        if example["chosen"] == example["rejected"]:
            raise ValueError("The two responses in each preference example must differ.")


def train():
    import torch
    from datasets import Dataset
    from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training
    from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
    from trl import DPOConfig, DPOTrainer, SFTConfig, SFTTrainer

    if not torch.cuda.is_available():
        raise SystemExit("4-bit Phi-2 training requires a supported CUDA GPU. Use --validate-data for an offline check.")

    model_id = "microsoft/phi-2"
    use_bf16 = torch.cuda.is_bf16_supported()
    quantization = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=torch.bfloat16 if use_bf16 else torch.float16,
        bnb_4bit_use_double_quant=True,
    )
    tokenizer = AutoTokenizer.from_pretrained(model_id)
    tokenizer.pad_token = tokenizer.pad_token or tokenizer.eos_token
    tokenizer.padding_side = "right"
    model = AutoModelForCausalLM.from_pretrained(
        model_id, quantization_config=quantization, device_map={"": torch.cuda.current_device()}
    )
    model.config.use_cache = False
    model = prepare_model_for_kbit_training(model)
    model = get_peft_model(
        model,
        LoraConfig(
            r=16, lora_alpha=32, target_modules=["q_proj", "k_proj", "v_proj", "dense"],
            lora_dropout=0.05, bias="none", task_type="CAUSAL_LM",
        ),
    )
    common = dict(
        per_device_train_batch_size=1, lr_scheduler_type="cosine",
        warmup_ratio=0.1, optim="paged_adamw_32bit", fp16=not use_bf16,
        bf16=use_bf16, logging_steps=1, report_to="none",
    )
    sft = SFTTrainer(
        model=model,
        args=SFTConfig(
            output_dir="./sft_output", num_train_epochs=3,
            gradient_accumulation_steps=8, learning_rate=2e-4, max_length=512,
            save_steps=100, **common,
        ),
        train_dataset=Dataset.from_list([{"text": format_messages(row["messages"])} for row in SFT_DATA]),
        processing_class=tokenizer,
    )
    sft_result = sft.train()
    print(f"SFT loss: {sft_result.training_loss:.4f}")
    sft.save_model("./sft_model")
    del sft

    # Compare the trainable policy to the saved SFT adapter on the same base model.
    model.load_adapter("./sft_model", adapter_name="sft_reference", is_trainable=False)
    model.set_adapter("default")
    dpo = DPOTrainer(
        model=model,
        args=DPOConfig(
            output_dir="./dpo_output", num_train_epochs=2,
            gradient_accumulation_steps=4, learning_rate=5e-5, beta=0.1,
            max_length=512, max_prompt_length=256, model_adapter_name="default",
            ref_adapter_name="sft_reference", **common,
        ),
        train_dataset=Dataset.from_list(DPO_DATA),
        processing_class=tokenizer,
    )
    dpo_result = dpo.train()
    print(f"DPO loss: {dpo_result.training_loss:.4f}")
    model.save_pretrained("./dpo_model", selected_adapters=["default"])
    tokenizer.save_pretrained("./dpo_model")

    model.eval()
    for question in (
        "What is the difference between a list and a tuple in Python?",
        "How does gradient descent work?",
        "What are three best practices for writing clean code?",
    ):
        prompt = f"System: You are a helpful, concise, and accurate AI assistant.\nInstruct: {question}\nOutput:"
        inputs = tokenizer(prompt, return_tensors="pt").to(model.device)
        with torch.inference_mode():
            output = model.generate(**inputs, max_new_tokens=200, do_sample=False)
        response = tokenizer.decode(output[0, inputs.input_ids.shape[1]:], skip_special_tokens=True).strip()
        print(f"\nQ: {question}\nA: {response}")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--validate-data", action="store_true", help="Check examples without loading ML libraries")
    args = parser.parse_args()
    validate_data()
    print(f"Validated {len(SFT_DATA)} SFT examples and {len(DPO_DATA)} preference pairs.")
    if not args.validate_data:
        train()


if __name__ == "__main__":
    main()
