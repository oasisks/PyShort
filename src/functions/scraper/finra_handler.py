import base64
import json
import requests
import os

from constant import Finra
from datetime import datetime
from dotenv import load_dotenv
from errors.finra_errors import FinraAuthInvalid
from loguru import logger
from model.RequestModel import GenericResponse
from requests.exceptions import RequestException

load_dotenv()


def authenticate() -> bytes:
    """
    Authenticates the FinraAPI
    :return:
    """
    # first check and see if the token is cached
    cache_dir = os.environ.get("CACHE_DIRECTORY")
    cache_file = os.path.join(cache_dir, "oath_cache.json")
    logger.info("CHECKING_OATH_CACHE_EXISTS")
    if os.path.isfile(cache_file):
        with open(cache_file, "r", encoding="utf-8") as oath_cache:
            data = json.load(oath_cache)
            # the cache is valid
            if data["expires_in"] > datetime.now().timestamp():
                logger.info("CACHE_GRABBED")
                return data["access_token"]

    api_client_id = os.environ.get("FINRA_CLIENT_ID")
    api_client_secret = os.environ.get("FINRA_CLIENT_SECRET")
    fip_endpoint = os.environ.get("FIP_ENDPOINT")

    credentials = f"{api_client_id}:{api_client_secret}".encode("ascii")
    encoded_credentials = base64.b64encode(credentials).decode("ASCII")
    headers = {
        "Authorization": f"Basic {encoded_credentials}"
    }

    logger.info("REQUESTING_AUTH")
    response = requests.post(fip_endpoint, headers=headers)

    if not response.ok:
        logger.error("AUTH_ERROR")
        raise FinraAuthInvalid

    response = response.json()

    with open(cache_file, "w") as oath_cache:
        response["expires_in"] = int(response["expires_in"]) + datetime.now().timestamp()
        json.dump(response, oath_cache, indent=4)

    return response


def get_request(group: str, dataset: str, use_async: bool = False) -> GenericResponse:
    """
    Performs a get request to Finra API

    :param group: The desired group
    :param dataset: The data set
    :param use_async: Async if True
    :return: A Generic Response
    """
    access_token = authenticate()
    url = f"{Finra.BASE_URL}/data/group/{group}/name/{dataset}{'?=async' if use_async else ''}"
    headers = {
        "Authorization": f"Bearer {access_token}"
    }

    response = requests.get(url, headers=headers)

    if not response.ok:
        raise RequestException

    data = {
        "content": response.content,
        "json_response": response.json() if 'application/json' in response.headers.get('Content-Type', '') else None,
        "status_link": response.headers.get("Location") if use_async else None
    }
    return GenericResponse.validate(data)


def post_request(group: str, dataset: str, payload: dict, use_async: bool = False) -> GenericResponse:
    """
    Performs a post request to Finra API

    :param group: The desired group
    :param dataset: The dataset
    :param payload: A payload to the post request
    :param use_async: Async if True
    :return: A Generic Response
    """
    access_token = authenticate()
    url = f"{Finra.BASE_URL}/data/group/{group}/name/{dataset}"

    headers = {
        "Authorization": f"Bearer {access_token}"
    }
    payload["async"] = use_async
    response = requests.post(url, headers=headers, json=payload)

    if not response.ok:
        raise RequestException

    data = {
        "content": response.content,
        "json_response": response.json() if 'application/json' in response.headers.get('Content-Type', '') else None,
        "status_link": response.headers.get("Location") if use_async else None
    }

    return GenericResponse.validate(data)


if __name__ == '__main__':
    group = "otcMarket"
    dataset = "consolidatedShortInterest"
    payload = {
        "compareFilters": [
            {"compareType": "equal", "fieldName": "marketClassCode", "fieldValue": "NNM"},
        ]
    }
    # response = post_request(group, dataset, payload)
    # print(response)
    # response = get_request(group, dataset)
    # dats = response.content.decode().split("\n")
    # print(dats)
    # print(len(dats))
    # for key in access_token:
    #     print(key, access_token[key])
    # print(access_token)
    # get_request("otcMarket", "consolidatedShortInterest")

