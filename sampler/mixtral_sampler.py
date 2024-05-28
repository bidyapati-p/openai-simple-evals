import asyncio
import time


from .http_wrapper import HttpWrapper
from .request_resp_handler import RequestRespHandler
from types1 import MessageList, SamplerBase
from transformers import AutoTokenizer

# reference: https://github.com/lm-sys/FastChat/blob/7899355ebe32117fdae83985cf8ee476d2f4243f/fastchat/conversation.py#L894


class MixtralInstructCompletionSampler(SamplerBase):
    """
    Sample from NOWLLM API
    """

    def __init__(
        self,
        model: str = "Mixtral",
        system_message: str | None = None,
        temperature: float = 0.0,  # default in Anthropic example
        max_tokens: int = 1024,
    ):
        self.api = "<MODEL_URL>"
        self.auth = "<TOKEN>"
        self.model = model
        self.system_message = system_message
        self.temperature = temperature
        self.max_tokens = max_tokens
        self.image_format = "base64"
        self.formatter = HFChatTemplateFormatter("mistralai/Mistral-7B-Instruct-v0.1", has_system_turn=False)

        # set client now
        self.inf_typ = "tgi"
        self.req_resp_hndlr = RequestRespHandler("tgi", {"model": "NowLLM"})
        if self.inf_typ == "tgi" or self.inf_typ == "triton":
            # ssl should be verified by default for triton
            verify_ssl = True if self.inf_typ == "triton" else False
            httpReq = HttpWrapper(timeout=30, verify_ssl=verify_ssl)
            self.req_resp_hndlr.set_client(httpReq)
        else:
            raise Exception(f"Invalid inference type at NowLLM object creation : {self.inf_typ}")

    def name(self):
        return "NowLLM"
    def _handle_image(
        self, image: str, encoding: str = "base64", format: str = "png", fovea: int = 768
    ):
        new_image = {
            "type": "image",
            "source": {
                "type": encoding,
                "media_type": f"image/{format}",
                "data": image,
            },
        }
        return new_image

    def _handle_text(self, text):
        return {"type": "text", "text": text}

    def _pack_message(self, role, content):
        return {"role": str(role), "content": content}

    
    def convert_to_text(self, message_list: MessageList):
        new_message_list = []
        for m in message_list:
            new_message_list.append({m.get("role"):m.get("content")})
        text = self.build_conversation_text(new_message_list)
        return text
    
    def convert_to_plaintext(self, message_list: MessageList):
        text = ""
        for m in message_list:
            text = text + m.get("content") + "\n"
        return text.strip()

    
    def build_conversation_text(self, inputs_pretok):
        """
        Build the conversation text from the input dictionary
        """
        conversation_text = ""
        if isinstance(inputs_pretok, str):
            conversation_text = inputs_pretok
        elif isinstance(inputs_pretok, list):
            # if list formatted data with role based text, convert based on model
            conversation_text += self.formatter.build_conversation_text(inputs_pretok)
        else:
            raise Exception(f"Invalid input type: {type(inputs_pretok)}")

        return conversation_text
    
    def replace_special_tokens(self, text):
        special_tokens= ["</s>","<|end|>", "<|endoftext|>","<|user|>", "<|assistant|>"]
        for token in special_tokens:
            text = text.replace(token, "")
        return text

    def __call__(self, message_list: MessageList) -> str:
        trial = 0
        msgBody = self.req_resp_hndlr.get_input_msg(self.convert_to_text(message_list),
                                                    {"temperature": self.temperature, "max_new_token": self.max_tokens, "stop":["<|end|>", "</s>"]})
        print(f"{self.name()} input message body: {str(msgBody)}")
        header_json = self.req_resp_hndlr.get_header_json(self.auth)

        while True:

            try:
                resp_status, resp_text =  self.req_resp_hndlr.request_server(url=self.api, auth=self.auth,
                                                                                  header_json=header_json,
                                                                                  msg_body=msgBody)
                if resp_status == 200:
                    print(f"{self.name()} model response: {str(resp_text)}")
                    text_resp = self.req_resp_hndlr.get_response_text(resp_status, resp_text)
                    text_resp = self.replace_special_tokens(text_resp).strip()
                    return text_resp if len(text_resp) > 0 else str(" ")
                else:
                    print(f"Error: error in the request. Code: {resp_status}.")
                    raise Exception(f"Error in server code:{resp_status}")
            except Exception as e:
                exception_backoff = 2**trial  # expontial back off
                print(
                    f"LLM server exception so wait and retry {trial} after {exception_backoff} sec",
                    e,
                )
                time.sleep(exception_backoff)
                trial += 1
                if trial > 10:
                    print("SYSTEM IS OVERLOADED")
                    return " "
            # unknown error shall throw exception

class HFChatTemplateFormatter:
    ROLE_KEY_SYSTEM = "system"
    ROLE_KEY_USER = "user"
    ROLE_KEY_ASSISTANT = "assistant"
    def __init__(self, hf_chat_template_model_id: str, has_system_turn: bool = True):
        self.hf_chat_template_model_id = hf_chat_template_model_id
        self.tokenizer = AutoTokenizer.from_pretrained(hf_chat_template_model_id)
        self.has_system_turn = has_system_turn

    def _get_next_item(self, inputs_pretok, idx):
        input = inputs_pretok[idx]
        # assumption is that the dict will have only one item
        role = next(iter(input))
        content = input[role]
        return role, content, idx + 1

    def build_conversation_text(self, inputs_pretok: list):
        hf_formatted_messages = []
        idx = 0
        while idx < len(inputs_pretok):
            role, content, idx = self._get_next_item(inputs_pretok, idx)
            # append system turn to user turn since model does not support system turn
            if role == HFChatTemplateFormatter.ROLE_KEY_SYSTEM and not self.has_system_turn:
                next_role, next_content, idx = self._get_next_item(inputs_pretok, idx)
                assert (
                    next_role == HFChatTemplateFormatter.ROLE_KEY_USER
                ), f"Applying chat template not possible for {self.hf_chat_template_model_id} since it does not support system turn and input does not contain system turn followed by user. Input: {inputs_pretok}"
                if next_role == HFChatTemplateFormatter.ROLE_KEY_USER:
                    role = next_role
                    content = content + "\n\n" + next_content
            hf_formatted_messages.append({"role": role, "content": content})

        chat_formatted_text = self.tokenizer.apply_chat_template(
            hf_formatted_messages, tokenize=False, add_generation_prompt=True
        )
        return chat_formatted_text