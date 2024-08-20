from langchain_openai import ChatOpenAI
from langchain_google_genai import ChatGoogleGenerativeAI



class LLMFactory:
    @staticmethod
    def get_llm(model=None, temperature=None):
        if model is None or temperature is None:
            raise ValueError("Both 'model' and 'temperature' must be specified.")

        if model.startswith('gpt'):
            return ChatOpenAI(model=model, temperature = temperature, streaming=True)

        if model.startswith('gemini'):
            return ChatGoogleGenerativeAI(model=model, temperature = temperature, streaming=True)

        raise ValueError(f"Model {model} is not supported.")