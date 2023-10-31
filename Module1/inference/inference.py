import argparse
import json, os
import pandas as pd
import numpy as np
parser = argparse.ArgumentParser()
parser.add_argument('--base_model', default='path/to/llama-7b', type=str, required=False)
parser.add_argument('--lora_model', default='path/to/lora', type=str)
parser.add_argument('--tokenizer_path',default='path/to/tokenizer',type=str)
parser.add_argument('--gpus', default="0", type=str)
parser.add_argument('--only_cpu',action='store_true',help='only use CPU for inference')
parser.add_argument('--alpha',type=str,default="1.0", help="The scaling factor of NTK method, can be a float or 'auto'. ")
parser.add_argument('--load_in_8bit',default=True, action='store_true', help="Load the LLM in the 8bit mode")

args = parser.parse_args()
if args.only_cpu is True:
    args.gpus = ""
os.environ["CUDA_VISIBLE_DEVICES"] = args.gpus
import torch
from transformers import LlamaForCausalLM, LlamaTokenizer
from peft import  PeftModel

from patches import apply_attention_patch, apply_ntk_scaling_patch
apply_attention_patch(use_memory_efficient_attention=True)
apply_ntk_scaling_patch(args.alpha)

generation_config = dict(
    temperature=0.2,
    top_k=40,
    top_p=0.9,
    do_sample=True,
    num_beams=1,
    repetition_penalty=1.1,
    max_new_tokens=400
    )

prompt_input = (

    #You are a professional doctor. Based on the management guidelines, provide comprehensive management recommendations for the following patient with diabetes:
    "### Instruction:\n\n{instruction}\n\n### Response:\n\n"
)

def generate_prompt(instruction):
    return prompt_input.format_map({'instruction': instruction})


if __name__ == '__main__':
    load_type = torch.float16
    if torch.cuda.is_available():
        device = torch.device(0) 
    else:
        device = torch.device('cpu')
    print(device)
    if args.tokenizer_path is None:
        args.tokenizer_path = args.lora_model
        if args.lora_model is None:
            args.tokenizer_path = args.base_model
    tokenizer = LlamaTokenizer.from_pretrained(args.tokenizer_path)
    base_model = LlamaForCausalLM.from_pretrained(
        args.base_model,
        load_in_8bit=args.load_in_8bit,
        torch_dtype=load_type,
        low_cpu_mem_usage=True,
        device_map='auto',
        )

    model_vocab_size = base_model.get_input_embeddings().weight.size(0)
    tokenzier_vocab_size = len(tokenizer)
    if model_vocab_size!=tokenzier_vocab_size:
        assert tokenzier_vocab_size > model_vocab_size
        print("Resize model embeddings to fit tokenizer")
        base_model.resize_token_embeddings(tokenzier_vocab_size)
    if args.lora_model is not None:
        print("loading peft model")
        model = PeftModel.from_pretrained(base_model, args.lora_model,torch_dtype=load_type,device_map='auto',)
    else:
        model = base_model

    if device==torch.device('cpu'):
        model.float()
    model.eval()
    with torch.no_grad():
        while True:
            raw_input_text = input("Input:")
'''example:
{Sex: Female, Age: 57, BMI: 24.4080691, SBP: 130, DBP: 78, heart rate: 72, Smoking: occasionally smoking;, 
Drinking: Drinking alcohol when socializing;, History of blood pressure: In 2005, elevated blood pressure 
was detected, which once reached 180/110mmHg. Now taking valsartan treatment, the usual blood pressure is 
around 130/85mmHg. He denies the history of stroke;, Medical history of circulatory system:  In 2014, color 
Doppler ultrasound showed rough carotid intima and thickened carotid intima-media layer, and rough carotid 
intima from 2015 to 2017;, Family history:  Father has high blood pressure, one brother has diabetes, and 
the other has a history of rectal cancer and kidney cancer;, Eating habits:  balanced meat and vegetables;, 
Physical activity: Body exercise frequency: every day; average running time: 60 minutes; exercise items: 
walking;, Medical history of endocrine system: Elevated fasting blood sugar was detected in 2005, and then 
he was diagnosed with type 2 diabetes. Now he is injecting insulin (15U Lantus at night), and taking 
repaglinide orally, with a slightly controlled diet. Follow-up fasting blood sugar is 7-8mmol/l; in 2000 
Left and right dyslipidemia, triglyceride 4.63mmol/l in 2013, 6.75mmol/l in 2015, 4.92mmol/l in 2016; 
cholesterol 6mmol/l in 2013, 6.28mmol/l in 2015, 6.27mmol/l in 2016; Low-density lipoprotein 3.58mmol/l, 
taking fenofibrate and atorvastatin intermittently, with a slightly controlled diet, normal blood lipids 
in 2017; thyroid nodules detected in 2013, left 6*4mm, right 10*6mm, left in 2014 6.1*4.8mm, right 10.2*5.8mm, 
left 6*4mm, right 10*9mm in 2015, left 6*5mm, right 10*8mm in 2016, left 6*5mm, right 13*7mm in 2017; 
bleeding in 2010 Elevated uric acid, 513.8umol/L in 2013, 502.8umol/l in 2014, 566.9umol/l in 2015, 
430.1umol/l in 2016, denied gout attack, did not take medicine, uncontrolled diet, normal blood uric acid 
in 2017;, Serum triglycerides: 3.21, Total cholesterol: 4.95, High-density lipoprotein (HDL): 1.14, 
Low-density lipoprotein (LDL): 2.72, Urine Albumin-to-Creatinine Ratio (UACR, mg/g): 33.4112990996339, 
Estimated glomerular filtration rate (eGFR): 41.44014187, Fasting serum glucose: 8.94, Glycated hemoglobin 
(HbA1c): 10.5, Aspartate aminotransferase (AST): 19.7, Alanine aminotransferase (ALT): 16.5, Gamma-glutamyl 
transferase (GGT): 21, DR grade: 0, DME grade: 0}
''' 
            if len(raw_input_text.strip())==0:
                break
            input_text = generate_prompt(instruction=raw_input_text)
            inputs = tokenizer(input_text,return_tensors="pt")
            generation_output = model.generate(
                input_ids = inputs["input_ids"].to(device),
                attention_mask = inputs['attention_mask'].to(device),
                eos_token_id=tokenizer.eos_token_id,
                pad_token_id=tokenizer.pad_token_id,
                **generation_config
            )
            s = generation_output[0]
            output = tokenizer.decode(s,skip_special_tokens=True)
            if args.with_prompt:
                response = output.split("### Response:")[1].strip()
            else:
                response = output
            print("Response: ",response)
            print("\n")