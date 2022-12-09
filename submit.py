import requests
from bs4 import BeautifulSoup
import time

url = "http://10.77.110.234:5000/result"
# headers = {
#     'Connection': 'keep-alive',
#     'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/108.0.0.0 Safari/537.36 Edg/108.0.1462.46',
#     'Content-Type': 'multipart/form-data; boundary=----WebKitFormBoundaryyKya8PtfIarck4RL',
#     'Accept-Encoding': 'gzip, deflate'
# }

def wait(sec):
    time.sleep(sec)

def post(filePath, times, wait_time):
    file = {
        'input_file': open(filePath, 'rb')
    }

    for i in range(times):
        try:
            r = requests.post(url, files=file)
            if(r.status_code == 200):
                return r.text
            else:
                raise Exception('Fail to access the server')
        except Exception:
            wait(time)

    raise Exception('Fail to access the server')

def submit(filePath, times=3, wait_time=3):
    text = post(filePath, times, wait_time)
    soup = BeautifulSoup(text, "html.parser")
    ps = soup.find_all('p')
    result = ps[1].text.split(' ')[1]
    return result

value = submit('test_output.csv')
print(value)