from huggingface_hub import InferenceClient

def AIFOROUTE(gender, age, allergies, area):
	client = InferenceClient(
		provider="together",
		api_key=""
	)

	messages = [
		{
			"role": "user",
			"content": f"You help an oncologist to plan a optimal chemotherapy route for a {gender} patient, age {age}, with {allergies}, the tumor is in the {area}. Make sure to list possible drugs, and there side affects."
		}
	]

	completion = client.chat.completions.create(
		model="deepseek-ai/DeepSeek-R1", 
		messages=messages, 
		max_tokens=500,
	)

	# print(completion.choices[0].message)

	return completion.choices[0].message