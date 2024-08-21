from langchain_openai import ChatOpenAI
from langchain_google_genai import ChatGoogleGenerativeAI



class LLMFactory:
    @staticmethod
    def get_llm(model=None, temperature=None):
        if model is None or temperature is None:
            raise ValueError("Both 'model' and 'temperature' must be specified.")

        if model.startswith('gpt'):
            return ChatOpenAI(model=model, temperature = temperature, streaming=True)

        # Notice that currently only gemini-1.5-flash-latest and gemini-1.5-pro-latest support function-calling
        if model.startswith('gemini'):
            return ChatGoogleGenerativeAI(model=model, temperature = temperature, streaming=True)

        raise ValueError(f"Model {model} is not supported.")