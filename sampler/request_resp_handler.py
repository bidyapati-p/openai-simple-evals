import asyncio
import json
from typing import Any

class RequestRespHandler:
    '''
    This class is responsible for creating request and processing response for each type of inference server
    '''

    def __init__(self, inference_type="tgi", properties={}):
        if not self.supported_type(inference_type):
            raise Exception("Unsupported inference server: " + inference_type)
        self.inference_type = inference_type
        self.properties = properties
        self.client = None
    def supported_type(self, inference_type):
        # ALL:"tgi", "vllm", "triton", "azure", "hf"
        supported = ["tgi", "vllm",
                     "triton"]
        return inference_type in supported

    def set_client(self, client:Any):
        '''
        Set this with Http if it is tgi or triton
        Set this with OpenAI client for vllm
        :param client: HttpWrapper object if tgi or triton
        :return:
        '''
        self.client = client

    def get_header_json(self, auth):
        '''
        Get the header json for http call
        :param auth: auth to be added into header
        :return: header json for http call
        '''
        if self.inference_type == "tgi" \
                or self.inference_type == "triton":
            return {'Content-Type': 'application/json', 'formatter': 'dummy_formatter',
             'chat_formatter': 'dummy_formatter', 'server': 'huggingface_server',
             'Authorization': auth}
        else:
            return None

    def get_input_msg(self, input_text, params):
        if self.inference_type == "tgi":
            return {
                "inputs": input_text,
                "parameters": params
            }
        elif self.inference_type == "triton":
            request_json = json.dumps({"prompt": input_text})
            return self._create_triton_request(request_json, options_json=json.dumps(params))
        elif self.inference_type == "vllm":
            return {"prompt":[input_text], "params":params}
        else:
            raise Exception(f"Invalid inference server type: {self.inference_type}")

    def request_server(self, url, auth, header_json, msg_body):
        '''
        Request TGI or Triton using http request or call vllm with openai client
        Http request needs url, header_json and msg_body
        OpenAI client need url, auth and msg_body. header_json will be None
        '''
        if self.inference_type == "tgi" \
                or self.inference_type == "triton":
            code, retmsg = self.client.post(url=url, headers=header_json, msg_body=msg_body)
            return code, retmsg
        elif self.inference_type == "vllm":
            params = msg_body.get("params", {})
            mxt = params.get("max_new_tokens", 500)
            tmp = params.get("temperature", 0.001)
            del params["max_new_tokens"]
            del params["temperature"]
            model_name = self.properties.get("model", "")

            try:
                # passing single prompt as an array
                prediction = self.client.completions.create(
                    model=model_name,
                    prompt=msg_body["prompt"],
                    max_tokens=mxt,
                    temperature=tmp,
                    extra_body={
                        **params
                    }
                )
                results = [p.dict()["text"].strip() for p in prediction.choices]
                # return only single result, we dont have batch at this level right now
                return 200, results[0]
            except Exception as e:
                print(e)
                return 429, f"vllm server error: {e}"
        else:
            raise Exception(f"Invalid inference server type: {self.inference_type}")


    def get_response_text(self, resp_status, resp_text):
        if self.inference_type == "tgi":
            json_resp = json.loads(resp_text)
            text_resp = json_resp['generated_text']
        elif self.inference_type == "triton":
            json_resp = json.loads(resp_text)
            text_resp = json_resp['outputs'][0]['data'][0]
            # check if strip() needed
            text_resp = json.loads(text_resp)['model_output']
        elif self.inference_type == "vllm":
            text_resp = resp_text
        else:
            raise Exception("Invalid inference server type for get_response_text() : "+ self.inference_type)

        return text_resp

    def _create_triton_request(self, request_json, options_json):
        return {
            "id": "42",
            "inputs": [
                {
                    "name": "request",
                    "shape": [
                        1,
                        1
                    ],
                    "datatype": "BYTES",
                    "data": [
                        request_json
                    ]
                },
                {
                    "name": "options",
                    "shape": [
                        1,
                        1
                    ],
                    "datatype": "BYTES",
                    "data": [ options_json ]
                }
            ],
            "outputs": [
                {
                    "name": "response"
                }
            ]
    }