from transformers import AutoModelForCausalLM, AutoTokenizer
from huggingface_hub import login
import torch
import gradio as gr

login()

model_id = "meta-llama/Llama-3.2-1B-Instruct"
model = AutoModelForCausalLM.from_pretrained(
  model_id,
  torch_dtype=torch.bfloat16
)

tokenizer = AutoTokenizer.from_pretrained(model_id)

if torch.cuda.is_available():
    device = torch.device("cuda")
    print(f"Usando GPU: {torch.cuda.get_device_name(device)}")
else:
    device = torch.device("cpu")
    print("Usando CPU")

model = model.to(device)

def respond(
        message,
        history,
        system_message,
        max_tokens,
        temperature,
        top_p
):
    messages = [{"role": "system", "content": system_message}]

    for val in history:
        if val[0]:
            messages.append({"role": "user", "content": val[0]})
        if val[1]:
            messages.append({"role": "assistant", "content": val[1]})

    messages.append({"role": "user", "content": message})

    input_ids = tokenizer.apply_chat_template(
        messages,
        add_generation_prompt=True,
        return_tensors='pt'
    ).to(model.device)

    terminators = [
        tokenizer.eos_token_id,
        tokenizer.convert_tokens_to_ids("<|eot_id|>")
    ]

    outputs = model.generate(
        input_ids,
        max_new_tokens=max_tokens,
        eos_token_id=terminators,
        do_sample=True,
        temperature=temperature,
        top_p=top_p
    )

    response = ""

    for message in tokenizer.decode(
        outputs[0][input_ids.shape[-1]:],
        skip_special_tokens=True
    ):
        response += message
        yield response

demo = gr.ChatInterface(
    respond,
    additional_inputs=[
        gr.Textbox(value="Tu eres un asistente amigable", label="System Message"),
        gr.Slider(minimum=1, maximum=2048, value=512, step=1, label="Max new tokens"),
        gr.Slider(minimum=0.1, maximum=3, value=0.7, step=0.1, label="Temperature"),
        gr.Slider(minimum=0.1, maximum=1, value=0.95, step=0.05, label="Top p")
    ]
)

if __name__ == "__main__":
    demo.launch()
