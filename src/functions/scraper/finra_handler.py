import base64
import requests
import os

from constant import Finra
from dotenv import load_dotenv
from model.RequestModel import GenericResponse

load_dotenv()


def authenticate() -> str:
    """
    Authenticates the FinraAPI
    :return:
    """
    api_client_id = os.environ.get("FINRA_CLIENT_ID")
    api_client_secret = os.environ.get("FINRA_CLIENT_SECRET")
    fip_endpoint = os.environ.get("FIP_ENDPOINT")

    credentials = f"{api_client_id}:{api_client_secret}".encode("ascii")
    encoded_credentials = base64.b64encode(credentials).decode("ASCII")
    headers = {
        "Authorization": f"Basic {encoded_credentials}"
    }

    response = requests.post(fip_endpoint, headers=headers)

    print(response.content)
    print(credentials, encoded_credentials)


def get_request(group: str, dataset: str) -> GenericResponse:
    """
    Performs a get request to Finra API

    :param group: The desired group
    :param dataset: The data set
    :return: A Generic Response
    """
    url = f"{Finra.BASE_URL}/data/group/{group}/name/{dataset}"
    api_client_id = os.environ.get("FINRA_CLIENT_ID")
    api_client_secret = os.environ.get("FINRA_CLIENT_SECRET")
    headers = {
        "Authorization": f"Basic {api_client_id}:{api_client_secret}"
    }
    print(headers)

    response = requests.get(url, headers=headers)
    # print(response.text)


def post_request(group: str, dataset: str, payload: dict) -> GenericResponse:
    """
    Performs a post request to Finra API

    :param group: The desired group
    :param dataset: The dataset
    :param payload: A payload to the post request
    :return: A Generic Response
    """


if __name__ == '__main__':
    authenticate()
    # get_request("otcMarket", "consolidatedShortInterest")

