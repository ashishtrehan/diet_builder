import configparser
from langchain.llms import OpenAI
from langchain.chat_models import ChatOpenAI

#API TOKEN
config = configparser.ConfigParser()
config.read('llm.ini')


class LLM:
    def __init__(self) -> None:
        self.open_api = config['OPEN_AI']['API_TOKEN']

print(LLM().open_api)
# llm = OpenAI(openai_api_key=)
# chat_model = ChatOpenAI()

# llm.predict("hi!")
# chat_model.predict("hi!")
# print(chat_model.predict("hi!"))
# print(chat_model)