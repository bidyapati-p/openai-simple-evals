import requests


class HttpWrapper:
    # timeout in seconds for each request
    timeout: int = 30

    def __init__(self, timeout, verify_ssl=True):        
        self.timeout = timeout
        self.verify_ssl = verify_ssl

    def get(self, url, headers):
        resp_text = " "        
        try:                

            resp = requests.get(url, headers=headers, timeout=self.timeout)
            resp_text = resp.text
            if resp.status_code != 200:
                print(
                    f"Error: HTTP request failed with code: {resp.status_code} and error: {resp_text}"
                )
        except Exception as x:
            print(f"Error: Http request failed {x}")
            return 408, " "
        return resp.status_code, resp_text
    
    def post(self, url, headers, msg_body, send_body_as_data=False):
        resp_text = " "        
        try:
            if send_body_as_data:
                resp = requests.post(
                    url, data=msg_body, headers=headers, timeout=self.timeout
                )
            else:
                resp = requests.post(
                    url, json=msg_body, headers=headers, timeout=self.timeout
                )
            if resp.status_code == 200:
                resp_text = resp.text
            else:
                print(f"Error: HTTP request failed with code: {resp.status_code}")
        except Exception as x:
            print(f"Error: Http request failed {x}")
            return 408, " "
        return resp.status_code, resp_text