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

def post(filePath):
    file = {
        'input_file': open(filePath, 'rb')
    }
    r = requests.post(url, files=file)
    if(r.status_code == 200):
        return r.text
    else:
        raise Exception('Fail to access the server')

def submit(filePath):
    text = post(filePath)
    soup = BeautifulSoup(text, "html.parser")
    ps = soup.find_all('p')
    result = ps[1].text.split(' ')[1]
    return result

def wait(sec):
    time.sleep(sec)

value = submit('test_output.csv')
print(value)