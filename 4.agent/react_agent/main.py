import json
import re
from dotenv import load_dotenv
from openai import OpenAI
from op_llm_client import OllamaClient
from agent import CustomerServiceAgent
from tools.query_product_data import query_by_product_name
from tools.read_store_promotions import read_store_promotions
from tools.calc import calculate


load_dotenv()


def load_config():
    with open("config.json", "r") as f:
        return json.load(f)


def get_client(config):
    if config["openai"].get("use_model", True):
        return OpenAI()
    else:
        return OllamaClient()


def get_max_iterations(config):
    # 选择使用的模型来确定最大迭代次数
    if config["ollama"]["use_model"]:
        return config["ollama"]["max_iterations"]
    elif config["openai"]["use_model"]:
        return config["openai"]["max_iterations"]
    else:
        return 10  # 如果没有启用任何模型，可以设置一个默认的迭代次数


def main():
    config = load_config()

    try:
        client = get_client(config)
        agent = CustomerServiceAgent(client, config)
    except Exception as e:
        print(f"Error initializing the AI client: {str(e)}")
        print("Please check your configuration and ensure the AI service is running.")
        return

    tools = {
        "query_by_product_name": query_by_product_name,
        "read_store_promotions": read_store_promotions,
        "calculate": calculate,
    }

    while True:
        query = input("📋 请输入问题或者如 'quit' 退出: ")
        if query.lower() == "quit":
            break
        if query.lower() == "messages":
            print(agent.messages)
            continue
        iteration = 0
        max_iterations = get_max_iterations(config)
        while iteration < max_iterations:
            try:
                result = agent(query)
                action_re = re.compile(r"^Action: (\w+): (.*)$")
                actions = [
                    action_re.match(line)
                    for line in result.split("\n")
                    if action_re.match(line)
                ]
                if actions:
                    tool_name = actions[0].group(1)
                    tool_args = actions[0].group(2)
                    if tool_name in tools:
                        print(f"ℹ️ 正在执行工具 {tool_name}: {tool_args} 并等待结果...")
                        try:
                            observation = tools[tool_name](tool_args)
                            query = f"Observation: {observation}"
                            print(f"ℹ️ 工具 {tool_name} 执行结果：{observation}")
                        except Exception as e:
                            query = f"Observation: Error executing tool {tool_name}: {str(e)}"
                    else:
                        query = f"Observation: Tool {tool_name} not found"
                elif "Answer:" in result:
                    print(f"🤖 客服回复：{result.split('Answer:')[-1].strip()}")
                    break
                else:
                    query = "Observation: No valid action or answer found. Please provide a clear action or answer."
            except Exception as e:
                print(f"An error occurred while processing the query: {str(e)}")
                print(
                    "Please check your configuration and ensure the AI service is running."
                )
                break

            iteration += 1

        if iteration >= max_iterations:
            print("Reached maximum number of iterations without a final answer.")


if __name__ == "__main__":
    main()
