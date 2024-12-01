import asyncio

import googleapiclient
from googleapiclient.discovery import build
from google.oauth2 import service_account
from googleapiclient.http import MediaFileUpload
import os
from typing import Optional, Dict
import io

service_account_file_name = "../service_account_key.json"
creds = service_account.Credentials.from_service_account_file(
    service_account_file_name,
    scopes=['https://www.googleapis.com/auth/drive']
)

drive_service = build('drive', 'v3', credentials=creds)


def find_shared_folder(drive_service: googleapiclient.discovery, folder_name: str):
    """
    This returns the folder we are interested in
    :param drive_service: the drive service
    :param folder_name: the name of the folder
    :return:
    """
    query = f"name = '{folder_name}' and mimeType = 'application/vnd.google-apps.folder' and trashed = false"

    # Execute the query
    response = drive_service.files().list(
        q=query,
        fields="files(id, name)"
    ).execute()

    folders = response.get('files', [])

    # Display the results
    if not folders:
        print(f"No folder found with the name: {folder_name}")
        return None
    else:
        for folder in folders:
            print(f"Folder Name: {folder['name']}, Folder ID: {folder['id']}")
        return folders[0]


def delete_file(drive_service: googleapiclient.discovery, file_id: str):
    """
    Deletes a file from the drive
    :param drive_service: the service
    :param file_id: the file id
    :return: None
    """
    try:
        drive_service.files().delete(fileId=file_id).execute()
    except Exception as e:
        print(f"An error occurred {e}")


async def delete_file_async(drive_service: googleapiclient.discovery, file_id: str):
    """
    Deletes a file asynchronously
    :param drive_service: the service
    :param file_id: the file id
    :return: None
    """
    await asyncio.to_thread(delete_file, drive_service, file_id)


async def remove_all_duplicates(drive_service: googleapiclient.discovery, folder_id: str, file_name: str) -> str:
    """
    In our drive, we only want unique filenames. Any files that exist in the drive that shares the same filename will
    be deleted.
    :param drive_service: the service
    :param folder_id: the root folder
    :param file_name: the name of the file
    :return: Bool
    """
    query = f"'{folder_id}' in parents and name = '{file_name}' and trashed = false"
    response = drive_service.files().list(q=query, fields="files(id, name)").execute()
    files = response.get('files', [])

    tasks = [delete_file_async(drive_service, file['id']) for file in files]
    await asyncio.gather(*tasks)


async def upload_file_to_folder(drive_service: googleapiclient.discovery, folder_id: str,
                                file_path: str) -> Optional[Dict]:
    """
    Uploads a file to a folder on the drive
    :param drive_service: service
    :param folder_id: the folder id
    :param file_path: the path of the file
    :return: A file object | None if failed to find the file
    """

    if not os.path.isfile(file_path):
        return

    file_name = os.path.basename(file_path)

    await remove_all_duplicates(drive_service, folder_id, file_name)
    # Specify the metadata for the file, including the parent folder ID
    file_metadata = {
        'name': file_name,
        'parents': [folder_id]
    }
    media = MediaFileUpload(file_path, resumable=True)

    # Upload the file to the specified folder
    file = drive_service.files().create(
        body=file_metadata,
        media_body=media,
        fields='id'
    ).execute()

    return file


async def upload_filebytes_to_folder(drive_service: googleapiclient.discovery, folder_id: str, file_name: str,
                                     file_bytes: bytes) -> Dict:
    """
    Uploads the bytes of the file directory to the drive
    :param drive_service: service
    :param folder_id: the folder we are writing to
    :param file_name: the name of the file
    :param file_bytes: the content of the file
    :return: a file object
    """
    await remove_all_duplicates(drive_service, folder_id, file_name)

    file_meta = {'name': file_name}

    if folder_id:
        file_meta['parent'] = [folder_id]

    file_stream = io.BytesIO(file_bytes)
    media = MediaFileUpload(file_stream, mimetype='application/octet-stream', resumable=True)

    file = drive_service.files().create(
        body=file_meta,
        media_body=media,
        fields='id'
    ).execute()

    return file


def main():
    file_path = 'example.txt'  # Local path to the file you want to upload
    folder = find_shared_folder(drive_service, "Data")
    folder_id = folder["id"]
    file = asyncio.run(upload_file_to_folder(drive_service, folder_id, file_path))

    print(file)


if __name__ == '__main__':
    main()
