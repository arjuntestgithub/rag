# from vllm import LLM, SamplingParams


# llm = LLM(model="facebook/opt-125m")  


# sampling_params = SamplingParams(
#     temperature=0.7,
#     max_tokens=100
# )

# output = llm.generate("Tell me a joke about AI", sampling_params)

# print(output[0].outputs[0].text)





# from mlx_lm import load, generate


# model, tokenizer = load("mlx-community/Meta-Llama-3.1-8B-Instruct-4bit")


# response = generate(
#     model,
#     tokenizer,
#     prompt="What is AI?",
#     max_tokens=200,
#     verbose=True
# )
model_id = "meta-llama/Llama-3.1-8B-Instruct"

# print(response)


from smolagents import CodeAgent, InferenceClientModel

model = InferenceClientModel(model_id)  


agent = CodeAgent(tools=[], model=model)
while True:
    abc = input() 
    result = agent.run(abc)
    print(result)




c7f75510442f4aa18d94a3f5e738511e