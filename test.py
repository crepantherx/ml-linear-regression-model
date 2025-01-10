import requests


# # flask
# if __name__ == '__main__':
#     url = "http://127.0.0.1:5000/predict"
#     data = {"input": [5, 7, 10]}
#     response = requests.post(url, json=data)
#     print(response.json())



if __name__ == '__main__':
    url = "http://127.0.0.1:8000/predict"
    data = {"input": [5, 7, 10]}
    response = requests.post(url, json=data)
    print(response.json())