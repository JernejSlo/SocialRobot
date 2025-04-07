import re

from transformers import T5Tokenizer, T5ForConditionalGeneration


class LLMUtils():
    def __init__(self):
        self.tokenizer = T5Tokenizer.from_pretrained("google/flan-t5-large")
        self.llm_model = T5ForConditionalGeneration.from_pretrained("google/flan-t5-large")
        pass


    def parse_llm_output(self,text):
        # Remove <pad> and </s> tokens if present
        clean_text = text.replace("<pad>", "").replace("</s>", "").strip()

        # Use regex to extract key-value pairs
        pattern = r'(\w+)\s*:\s*([\w\s]+)'
        matches = re.findall(pattern, clean_text)

        return {key.strip(): value.strip() for key, value in matches}

    def parse_goal(self,text):
        prompt = """
                You are an assistant for a robot arm. The user says: "{text}"
                Extract the intended action as a JSON object with keys: action, object, and target.

                Examples:

                Input: "Can you give me the lemon?"
                {{"action": "give", "object": "lemon", "target": "user"}} 

                Input: "Grab the apple from the table"
                {{"action": "grab", "object": "apple", "target": "table"}}

                Input: "Move the lemon to the right"
                {{"action": "move", "object": "lemon", "target": "right"}}
                
                Input: "hello my name is Michael"
                {{"action": None", "object": None", "target": None"}}
                
                The result MUST include keys action, object and target. If any of the values are not in the query replace the values by writing "None" in place of the value.

                Now do it for:\n
                """
        prompt += f"{text}"
        # Replace with actual LLM call (e.g., OpenAI Chat)
        result = self.call_llm(prompt)
        print(result)
        output_dict = self.parse_llm_output(result)
        return output_dict

    def call_llm(self,text):

        input_text = text
        input_ids = self.tokenizer(input_text, return_tensors="pt").input_ids

        outputs = self.llm_model.generate(input_ids)
        return self.tokenizer.decode(outputs[0])

#print(LLMUtils().parse_goal("Give"))
