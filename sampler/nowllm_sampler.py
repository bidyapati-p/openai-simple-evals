import asyncio
import time

import anthropic

from .http_wrapper import HttpWrapper
from .request_resp_handler import RequestRespHandler
from types1 import MessageList, SamplerBase

# reference: https://github.com/lm-sys/FastChat/blob/7899355ebe32117fdae83985cf8ee476d2f4243f/fastchat/conversation.py#L894


class NowLLMCompletionSampler(SamplerBase):
    """
    Sample from NOWLLM API
    """

    def __init__(
        self,
        model: str = "NowLLM",
        system_message: str | None = None,
        temperature: float = 0.0,  # default in Anthropic example
        max_tokens: int = 1024,
    ):
        self.api = "<NOWLLM_URL>"
        self.auth = "<AUTH_BEARER_TOKEN>"
        self.model = model
        self.system_message = system_message
        self.temperature = temperature
        self.max_tokens = max_tokens
        self.image_format = "base64"

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
        text = ""
        for m in message_list:
            if m.get("role") == "system":
                text = text + "<|system|>" + m.get("content") + "<|end|>\n"
            if m.get("role") == "user":
                text = text + "<|user|>" + m.get("content") + "<|end|>\n"
            if m.get("role") == "assistant":
                text = text + "<|assistant|>" + m.get("content") + "<|end|>\n"
        return text

    def __call__(self, message_list: MessageList) -> str:
        trial = 0
        msgBody = self.req_resp_hndlr.get_input_msg(self.convert_to_text(message_list),
                                                    {"temperature": self.temperature, "max_new_token": self.max_tokens})
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
