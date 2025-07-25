import requests


class OllamaClient:
    def __init__(self, base_url="http://localhost:11434") -> None:
        self.base_url = base_url

    def chat_completion_create(self, model, messages, temperature=0.7):
        url = f"{self.base_url}/api/generate"
        payload = {
            "model": model,
            "prompt": self._format_messages(messages),
            "stream": False,
            "temperature": temperature,
        }

        try:
            response = requests.post(url, json=payload)
            if response.status_code == 200:
                return response.json()["response"]
            else:
                print(f"Error: {response.status_code} {response.text}")
        except requests.exceptions.ConnectionError:
            raise ConnectionError(
                f"Failed to connect to Ollama server at {self.base_url}. "
                f"Make sure Ollama is running and accessible."
            )
        except requests.exceptions.Timeout:
            raise TimeoutError(
                f"Request to Ollama server at {self.base_url} timed out. "
                f"The server might be overloaded or not responding."
            )
        except requests.exceptions.HTTPError as e:
            if e.response.status_code == 404:
                raise ValueError(
                    f"The specified model might not be available on the Ollama server. "
                    f"Error: {str(e)}"
                )
            else:
                raise

    def _format_messages(self, messages):
        formatted_prompt = ""
        for message in messages:
            if message["role"] == "system":
                formatted_prompt += f"System: {message['content']}\n"
            elif message["role"] == "user":
                formatted_prompt += f"Human: {message['content']}\n"
            elif message["role"] == "assistant":
                formatted_prompt += f"Assistant: {message['content']}\n"
        return formatted_prompt.strip()

    def _parse_response(self, response):
        return {
            "choices": [
                {
                    "message": {
                        "role": "assistant",
                        "content": response.get("response", ""),
                    }
                }
            ]
        }
