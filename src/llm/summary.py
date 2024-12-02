import json
import os

from deprecated import deprecated
from dotenv import load_dotenv
from enum import Enum
from openai import OpenAI
from openai.types import Batch
from typing import List, Tuple

load_dotenv()

API_KEY = os.environ.get("OPENAI_API_KEY")
client = OpenAI(
    api_key=API_KEY
)

model = "gpt-4o"


class SummaryStatus(Enum):
    COMPLETED = 0
    ERROR = 1
    PROCESSING = 2


@deprecated(reason="This was used for testing purposes, use create_batch_summarize for cost efficiency")
def summarize(prompt: str) -> str:
    """
    :param prompt: the prompt
    Given a prompt, it will attempt to summarize whatever was given
    :return: The summarized results
    """
    messages = [
        {
            "role": "system",
            "content": "You are an expert in financial literacy and sentiment analysis,"
                       " specializing in news and data related to stocks and the stock market. "
                       "Your primary focus is to analyze financial news, trends, and data to provide insights and"
                       " context specific to stock market performance, investor sentiment,"
                       " and related financial metrics. Your responses should remain grounded "
                       "in the context of stocks and the stock market."
        },
        {
            "role": "user",
            "content": prompt
        }
    ]
    response = client.chat.completions.create(
        messages=messages,
        model=model
    )
    chat_response = response.choices[0].message.content

    if not chat_response:
        return "Chat didn't return a valid response"

    return chat_response


def create_file(inputs: List[Tuple[str, str]], file_name: str) -> None:
    """
    Creates a jsonl file
    :param inputs: a list of tuples where the first is the ticker_id and the second is the prompt
    :param file_name: the name of the output file
    :return: None
    """
    file = open(file_name, "w", encoding="utf-8")
    for _ in inputs:
        ticker_id, prompt = _
        body = {
            "model": model,
            "messages": [
                {
                    "role": "system",
                    "content": "You are an expert in financial literacy and sentiment analysis,"
                               " specializing in news and data related to stocks and the stock market. "
                               "Your primary focus is to analyze financial news, trends, and data to provide insights "
                               "and context specific to stock market performance, investor sentiment,"
                               " and related financial metrics. Your responses should remain grounded "
                               "in the context of stocks and the stock market."
                },
                {
                    "role": "user",
                    "content": prompt
                }
            ]
        }
        line = {"custom_id": ticker_id, "method": "POST", "url": "/v1/chat/completions", "body": body}
        file.write(json.dumps(line) + "\n")

    file.close()


def list_batches(limit=20) -> List[Tuple[str, str]]:
    """
    Returns a list of tuples where the first element is the batch_id and the second is the filename.
    It is ordered from most recent batch to the least recent batch (i.e. first element most recent)
    :param limit: the maximum number of batches to retrieve (max 100)
    :return: List of tuples
    """
    assert limit <= 100, "Limit is above 100. Range is only from [0, 100]."
    batches = client.batches.list(limit=limit).data
    sorted_models = sorted(batches, key=lambda x: x.created_at)
    return [(batch.id, client.files.retrieve(file_id=batch.input_file_id).filename) for batch in sorted_models]


def create_batch_summarize(file_name: str) -> Batch:
    """
    The purpose of this function is to create batch requests to OpenAI's GPT Chat/Completion endpoint.

    This will not check if a request for a batch with the file_name has already been requested.
    It will create a brand-new request.

    :param file_name: the name of the jsonl file that contains the batched messages
    :return: The batch model
    """
    # upload the file
    file = file_name, open(file_name, "rb")
    upload_model = client.files.create(
        file=file,
        purpose="batch"
    )

    input_file_id = upload_model.id

    # creates the batch
    batch_model = client.batches.create(
        input_file_id=input_file_id,
        endpoint="/v1/chat/completions",
        completion_window="24h"
    )

    return batch_model


def retrieve_batch_summarize(batch_id: str) -> SummaryStatus:
    """
    This retrieves the batch given the batch_id. If the batch is not finished then return the batch model.

    :param batch_id: the batch id
    :return: Optional[bytes]
    """
    response = client.batches.retrieve(
        batch_id=batch_id
    )

    if response.status == "completed":
        # grab the file on the server
        try:
            file_info = client.files.retrieve(
                file_id=response.input_file_id
            )

            file_response = client.files.content(
                file_id=response.output_file_id
            )
            file_name = "out_" + file_info.filename
            file_response.write_to_file(file_name)
            print(f"Successfully retrieved and written to file: {file_name}")
            return SummaryStatus.COMPLETED
        except Exception as e:
            print(f"Couldn't retrieve the output file from the batch: {response.output_file_id}. Error msg: {e}")
            return SummaryStatus.ERROR

    return SummaryStatus.PROCESSING


if __name__ == '__main__':
    #     prompt = """Instruction: Please summarize the following noisy but possible tweet posts extracted from StockTwits for TSLA stock, and extract keywords from the tweets. The tweets' text can be very noisy due to it being user-generated. Provide a separate summary for each tweet and extract keywords for all. Format the answer as: Summary: Tweet 1: …, …, Tweet N: …, Keywords: … You may put ’N/A’ if the noisy text does not have relevant information to extract.
    #
    # Tweets: {
    # Randomly Sample Tweets are structured as:
    # 1. $TSLA Why dropped so quickly?  Any news?
    # 2. $TSLA alright dip buyers let’s see if you exist
    # 3. $TSLA dumping Tesla for $DOGE.X
    # Elon named his new department after doge instead of his company. This says how much he cares about his garbage ev company
    # 4. $TSLA when layoffs at Tesla?
    # 5. $SPY $TSLA $NVDA $AMD
    #
    # AMD laying off employees.
    # Soon will be Tesla.
    # Too much hype with no revenues.
    # }"""
    #     summary = summarize(prompt)
    #
    #     print("This is the summary")
    #     print(summary)
    # create_batch_summarize("example.jsonl")
    #     inputs = [("aapl", """Instruction: Please summarize the following noisy but possible tweet posts extracted from StockTwits for AAPL stock, and extract keywords from the tweets. The tweets' text can be very noisy due to it being user-generated. Provide a separate summary for each tweet and extract keywords for all. Format the answer as: Summary: Tweet 1: …, …, Tweet N: …, Keywords: … You may put ’N/A’ if the noisy text does not have relevant information to extract.
    #
    # Tweets: {
    # 1. $AAPL retard bears think apple makes walkie-talkies
    #
    # The latest attack, which affected hundreds of walkie-talkies across the country, killed at least 14 people and wounded 450 others, the Lebanese health ministry said.
    #
    # 2. $AAPL after the info is digested which way does this go tomorrow?
    #
    # 3. $AAPL iOS 18 just killed all satellite phones. As hiker and backpacker, this is great reason to own iPhone.
    #
    # 4. $AAPL  the beauty about trading with half of the calls bought at $0.57 and booked already $1.4 in profits is playing &quot;free money&quot;; if SPY goes to $572, then here like Friday Aug 2 (after FOMC Jul 31) will tag $125.6 = $3; today those calls closed at $1.46
    # StockReturn: Positive
    #
    # 5. AAPL king Apple takes the crown like usual hahahah✅📈📈📈📈‼️
    # }"""), ("tsla", """Instruction: Please summarize the following noisy but possible tweet posts extracted from StockTwits for TSLA stock, and extract keywords from the tweets. The tweets' text can be very noisy due to it being user-generated. Provide a separate summary for each tweet and extract keywords for all. Format the answer as: Summary: Tweet 1: …, …, Tweet N: …, Keywords: … You may put ’N/A’ if the noisy text does not have relevant information to extract.
    #
    # Tweets: {
    # Randomly Sample Tweets are structured as:
    # 1. $TSLA Why dropped so quickly?  Any news?
    # 2. $TSLA alright dip buyers let’s see if you exist
    # 3. $TSLA dumping Tesla for $DOGE.X
    # Elon named his new department after doge instead of his company. This says how much he cares about his garbage ev company
    # 4. $TSLA when layoffs at Tesla?
    # 5. $SPY $TSLA $NVDA $AMD
    #
    # AMD laying off employees.
    # Soon will be Tesla.
    # Too much hype with no revenues.
    # }"""), ("msft", """Instruction: Please summarize the following noisy but possible tweet posts extracted from StockTwits for MSFT stock, and extract keywords from the tweets. The tweets' text can be very noisy due to it being user-generated. Provide a separate summary for each tweet and extract keywords for all. Format the answer as: Summary: Tweet 1: …, …, Tweet N: …, Keywords: … You may put ’N/A’ if the noisy text does not have relevant information to extract.
    #
    # Tweets: {
    # 1. $MSFT https://www.techradar.com/computing/windows/sfc-bug-detected-in-new-windows-11-24h2-update-microsoft-sure-to-issue-a-patch
    #
    # What exactly is so magnificent about Microsoft? Other than it makes people rich because people are too reliant on it and can no longer function on their own accord.
    #
    # Even the software programmers can’t get it right
    # 2. $MSFT Microsoft announced a series of new health-care data and artificial intelligence tools on Thursday.
    #
    # The tools are designed to help health systems build AI applications more quickly and save clinicians time on administrative tasks.
    #
    # It’s the latest example of latest Microsoft’s efforts to establish itself as a leader in health-care AI.
    # 3. $MSFT funny thing is we need to break $420 for a breakout
    #
    # $TLRY $CGC $TLRY
    # 4. $MSFT damn brutal.
    #
    # Hopefully just a. Retail shakeout. That $420 calls for tomorrow not looking real good
    #
    # 5. MSFT watching this now to test and retest key levels and we’ve got action. Will post soon.
    # }""")]
    # create_file(inputs, "example.jsonl")
    # batch = create_batch_summarize("example.jsonl")
    #
    # print(batch.model_dump())
    # print(json.dumps(batch.model_dump(), indent=4))
    # print(retrieve_batch_summarize("batch_674d18eb530c81909b63e87d8111dded"))
    # batch_list = list_batches()
    #
    # for batch in batch_list:
    #     print(batch)
    pass
